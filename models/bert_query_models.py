from apex.normalization import FusedLayerNorm
import torch
from torch import nn
from transformers.modeling_bert import BertConfig, BertPreTrainedModel, BertModel
from copy import deepcopy

from general_util.utils import LogMetric
from modules import layers
from modules.layers import check_nan

r"""
Copied from models.roberta_cquery_d_models.py

- Change the base model from `RobertaPreTrainedModel` to `BertPreTrainedModel`.
- Add full pre-trained config.

"""


class BertQueryPreTrainedConfig(BertConfig):
    added_configs = [
        'query_dropout', 'cls_type', 'sr_query_dropout', 'lm_query_dropout', 'pos_emb_size'
    ]

    def __init__(self, query_dropout=0.1, cls_type=0,
                 sr_query_dropout=0.1, lm_query_dropout=0.1,
                 pos_emb_size=200, **kwargs):
        super().__init__(**kwargs)

        self.query_dropout = query_dropout
        self.cls_type = cls_type
        self.sr_query_dropout = sr_query_dropout
        self.lm_query_dropout = lm_query_dropout
        self.pos_emb_size = pos_emb_size

    def expand_configs(self, *args):
        self.added_configs.extend(list(args))


class QueryBertModel(BertPreTrainedModel):
    config_class = BertQueryPreTrainedConfig
    model_prefix = 'query_bert'

    def __init__(self, config: BertQueryPreTrainedConfig):
        super().__init__(config)

        self.config = config
        self.bert = BertModel(config)

        self.query = layers.MultiHeadSelfTokenAttention(
            config,
            attn_dropout_p=config.query_dropout,
            dropout_p=config.query_dropout
        )

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                sentence_index=None, sentence_mask=None, sent_word_mask=None,
                **kwargs):
        seq_output = self.bert(input_ids=input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids)[0]

        batch, sent_num, seq_len = sent_word_mask.size()
        sentence_index = sentence_index.unsqueeze(-1).expand(
            -1, -1, -1, self.config.hidden_size
        ).reshape(batch, sent_num * seq_len, self.config.hidden_size)

        sent_word_hidden = seq_output.gather(dim=1, index=sentence_index).reshape(
            batch, sent_num, seq_len, -1)

        hidden_sent = self.query(sent_word_hidden, sent_word_mask)

        return hidden_sent, seq_output, sent_word_hidden


class QueryBertModelForSRAndMLM(QueryBertModel):
    model_prefix = 'query_bert_sr_mlm'

    def __init__(self, config: BertQueryPreTrainedConfig):
        super().__init__(config)

        self.sr_sent_sum = nn.Linear(config.hidden_size, config.hidden_size)
        self.lm_sent_sum = nn.Linear(config.hidden_size, config.hidden_size)

        word_embedding_weight = self.bert.get_input_embeddings().weight
        self.vocab_size = word_embedding_weight.size(0)

        config.layer_norm_eps = 1e-5  # avoid fp16 underflow
        self.lm_prediction_head = layers.MaskedLMPredictionHead(config, word_embedding_weight)

        self.sr_pooler = layers.Pooler(config.hidden_size)
        self.sr_prediction_head = nn.Linear(config.hidden_size, 1)

        self.lm_dropout = nn.Dropout(config.lm_query_dropout)
        self.sr_dropout = nn.Dropout(config.sr_query_dropout)

        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-1)

        self.init_weights()

        # metric
        self.eval_metrics = LogMetric("sr_acc", "sr_loss", "mlm_loss", "mlm_acc")

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                sentence_index=None, sentence_mask=None, sent_word_mask=None,
                mlm_ids=None, true_sent_ids=None, reverse_sentence_index=None,
                answers=None, pre_answers=None, **kwargs):

        hidden_sent, seq_output, sent_word_hidden = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            sentence_index=sentence_index,
            sentence_mask=sentence_mask,
            sent_word_mask=sent_word_mask
        )
        batch, sent_num, word_num = sent_word_mask.size()
        seq_length = seq_output.size(1)

        query_num = answers.size(1)
        query_h = hidden_sent[:, :query_num]

        attention_mask = attention_mask.to(query_h.dtype)

        # SR
        sr_query_h = self.sr_sent_sum(query_h)

        q_rel_d_sent_h, _ = layers.mul_sentence_sum(
            sr_query_h, sent_word_hidden, sent_word_mask
        )

        sr_scores = self.sr_prediction_head(
            self.sr_dropout(self.sr_pooler(q_rel_d_sent_h))).squeeze(-1)

        # MLM
        lm_query_h = self.lm_sent_sum(query_h)
        query_token_num = mlm_ids.size(1)

        q_rel_d_h, _ = layers.mul_weighted_sum(
            lm_query_h, seq_output, 1 - attention_mask
        )
        q_rel_d_h = self.lm_dropout(q_rel_d_h)

        aligned_sent_hidden = q_rel_d_h.gather(
            dim=1,
            index=reverse_sentence_index.unsqueeze(-1).expand(-1, -1, seq_output.size(-1))
        )

        concat_word_hidden = torch.cat([seq_output[:, :query_token_num], aligned_sent_hidden], dim=-1)

        mlm_scores = self.lm_prediction_head(concat_word_hidden)

        output_dict = {}

        if mlm_ids is not None and answers is not None and pre_answers is not None:

            sent_mask = sentence_mask
            sent_mask = sent_mask.unsqueeze(1).expand(-1, query_num, -1)

            sr_scores = sr_scores + sent_mask * -10000.0

            fol_masked_scores = layers.mask_scores_with_labels(sr_scores, answers).contiguous()
            sr_loss1 = self.loss_fct(fol_masked_scores.view(batch * query_num, -1),
                                     pre_answers.view(-1))

            pre_masked_scores = layers.mask_scores_with_labels(sr_scores, pre_answers).contiguous()
            sr_loss2 = self.loss_fct(pre_masked_scores.view(batch * query_num, -1),
                                     answers.view(-1))

            mlm_loss = self.loss_fct(mlm_scores.view(-1, self.config.vocab_size),
                                     mlm_ids.view(-1))

            print(sr_loss1, sr_loss2, mlm_loss)

            loss = sr_loss1 + sr_loss2 + mlm_loss

            output_dict["loss"] = loss

            if not self.training:
                _, mlm_pred = mlm_scores.max(dim=-1)
                mlm_valid_num = (mlm_ids != -1).sum().item()
                mlm_acc = (mlm_pred == mlm_ids).sum().to(loss.dtype).item() / mlm_valid_num

                self.eval_metrics.update("mlm_loss", mlm_loss.item(), mlm_valid_num)
                self.eval_metrics.update("mlm_acc", mlm_acc, mlm_valid_num)

                valid_num1 = (answers != -1).sum().item()
                valid_num2 = (pre_answers != -1).sum().item()
                valid_num = valid_num1 + valid_num2

                _, pred = torch.topk(sr_scores, k=2, dim=-1, largest=True)

                acc1 = (pred == answers.unsqueeze(-1)).sum()
                acc2 = (pred == pre_answers.unsqueeze(-1)).sum()

                acc = (acc1 + acc2).to(dtype=sr_scores.dtype) / (valid_num * 1.0)

                output_dict["acc"] = acc
                output_dict["valid_num"] = valid_num

                self.eval_metrics.update("sr_acc", acc.item(), valid_num)
                self.eval_metrics.update("sr_loss", loss.item(), valid_num)

        return output_dict

    def get_eval_log(self, reset=False):
        _eval_metric_log = self.eval_metrics.get_log()
        _eval_metric_log = '\t'.join([f"{k}: {v}" for k, v in _eval_metric_log.items()])

        if reset:
            self.eval_metrics.reset()

        return _eval_metric_log


class QueryBertModelForBiSRAndMLM(QueryBertModel):
    model_prefix = 'query_bert_bi_sr_mlm'

    def __init__(self, config: BertQueryPreTrainedConfig):
        super().__init__(config)

        self.sr_sent_sum = nn.Linear(config.hidden_size, config.hidden_size)
        self.lm_sent_sum = nn.Linear(config.hidden_size, config.hidden_size)

        word_embedding_weight = self.bert.get_input_embeddings().weight
        self.vocab_size = word_embedding_weight.size(0)

        config.layer_norm_eps = 1e-5  # avoid fp16 underflow
        self.lm_prediction_head = layers.MaskedLMPredictionHead(config, word_embedding_weight)

        self.pre_sr_pooler = layers.Pooler(config.hidden_size)
        self.pre_sr_prediction_head = nn.Linear(config.hidden_size, 1)
        
        self.fol_sr_pooler = layers.Pooler(config.hidden_size)
        self.fol_sr_prediction_head = nn.Linear(config.hidden_size, 1)

        self.lm_dropout = nn.Dropout(config.lm_query_dropout)
        self.sr_dropout = nn.Dropout(config.sr_query_dropout)

        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-1)

        self.init_weights()

        # metric
        self.eval_metrics = LogMetric("sr_acc", "sr_loss", "mlm_loss", "mlm_acc")

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                sentence_index=None, sentence_mask=None, sent_word_mask=None,
                mlm_ids=None, true_sent_ids=None, reverse_sentence_index=None,
                answers=None, pre_answers=None, **kwargs):

        hidden_sent, seq_output, sent_word_hidden = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            sentence_index=sentence_index,
            sentence_mask=sentence_mask,
            sent_word_mask=sent_word_mask
        )
        batch, sent_num, word_num = sent_word_mask.size()
        seq_length = seq_output.size(1)

        query_num = answers.size(1)
        query_h = hidden_sent[:, :query_num]

        attention_mask = attention_mask.to(query_h.dtype)

        # SR
        sr_query_h = self.sr_sent_sum(query_h)

        q_rel_d_sent_h, _ = layers.mul_sentence_sum(
            sr_query_h, sent_word_hidden, sent_word_mask
        )

        # sr_scores = self.sr_prediction_head(
        #     self.sr_dropout(self.sr_pooler(q_rel_d_sent_h))).squeeze(-1)

        pre_sr_scores = self.pre_sr_prediction_head(
            self.sr_dropout(self.pre_sr_pooler(q_rel_d_sent_h))
        ).squeeze(-1)
        
        fol_sr_scores = self.fol_sr_prediction_head(
            self.sr_dropout(self.fol_sr_pooler(q_rel_d_sent_h))
        ).squeeze(-1)

        # MLM
        lm_query_h = self.lm_sent_sum(query_h)
        query_token_num = mlm_ids.size(1)

        q_rel_d_h, _ = layers.mul_weighted_sum(
            lm_query_h, seq_output, 1 - attention_mask
        )
        q_rel_d_h = self.lm_dropout(q_rel_d_h)

        aligned_sent_hidden = q_rel_d_h.gather(
            dim=1,
            index=reverse_sentence_index.unsqueeze(-1).expand(-1, -1, seq_output.size(-1))
        )

        concat_word_hidden = torch.cat([seq_output[:, :query_token_num], aligned_sent_hidden], dim=-1)

        mlm_scores = self.lm_prediction_head(concat_word_hidden)

        output_dict = {}

        if mlm_ids is not None and answers is not None and pre_answers is not None:

            sent_mask = sentence_mask
            sent_mask = sent_mask.unsqueeze(1).expand(-1, query_num, -1)

            fol_sr_scores = fol_sr_scores + sent_mask * -10000.0
            pre_sr_scores = pre_sr_scores + sent_mask * -10000.0

            sr_loss1 = self.loss_fct(pre_sr_scores.view(batch * query_num, -1),
                                     pre_answers.view(-1))

            sr_loss2 = self.loss_fct(fol_sr_scores.view(batch * query_num, -1),
                                     answers.view(-1))

            mlm_loss = self.loss_fct(mlm_scores.view(-1, self.config.vocab_size),
                                     mlm_ids.view(-1))

            print(sr_loss1, sr_loss2, mlm_loss)

            loss = sr_loss1 + sr_loss2 + mlm_loss

            output_dict["loss"] = loss

            if not self.training:
                _, mlm_pred = mlm_scores.max(dim=-1)
                mlm_valid_num = (mlm_ids != -1).sum().item()
                mlm_acc = (mlm_pred == mlm_ids).sum().to(loss.dtype).item() / mlm_valid_num

                self.eval_metrics.update("mlm_loss", mlm_loss.item(), mlm_valid_num)
                self.eval_metrics.update("mlm_acc", mlm_acc, mlm_valid_num)

                valid_num1 = (answers != -1).sum().item()
                valid_num2 = (pre_answers != -1).sum().item()
                valid_num = valid_num1 + valid_num2

                # _, pred = torch.topk(sr_scores, k=2, dim=-1, largest=True)
                _, pre_pred = torch.max(pre_sr_scores, dim=-1)
                _, fol_pred = torch.max(fol_sr_scores, dim=-1)

                acc1 = (fol_pred == answers).sum()
                acc2 = (pre_pred == pre_answers).sum()

                acc = (acc1 + acc2).to(dtype=pre_sr_scores.dtype) / (valid_num * 1.0)

                output_dict["acc"] = acc
                output_dict["valid_num"] = valid_num

                self.eval_metrics.update("sr_acc", acc.item(), valid_num)
                self.eval_metrics.update("sr_loss", loss.item(), valid_num)

        return output_dict

    def get_eval_log(self, reset=False):
        _eval_metric_log = self.eval_metrics.get_log()
        _eval_metric_log = '\t'.join([f"{k}: {v}" for k, v in _eval_metric_log.items()])

        if reset:
            self.eval_metrics.reset()

        return _eval_metric_log


class QueryBertModelForSRAndPosMLM(QueryBertModel):
    model_prefix = 'query_bert_sr_pos_mlm'

    def __init__(self, config: BertQueryPreTrainedConfig):
        super().__init__(config)

        self.sr_sent_sum = nn.Linear(config.hidden_size, config.hidden_size)
        self.lm_sent_sum = nn.Linear(config.hidden_size, config.hidden_size)

        word_embedding_weight = self.bert.get_input_embeddings().weight
        self.vocab_size = word_embedding_weight.size(0)

        config.layer_norm_eps = 1e-5  # avoid fp16 underflow
        # self.lm_prediction_head = layers.MaskedLMPredictionHead(config, word_embedding_weight)
        self.lm_prediction_head = layers.MaskedLMPredictionHeadPos(config,
                                                                   word_embedding_weight,
                                                                   config.pos_emb_size)

        self.sr_pooler = layers.Pooler(config.hidden_size)
        self.sr_prediction_head = nn.Linear(config.hidden_size, 1)

        self.lm_dropout = nn.Dropout(config.lm_query_dropout)
        self.sr_dropout = nn.Dropout(config.sr_query_dropout)

        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-1)

        self.init_weights()

        # metric
        self.eval_metrics = LogMetric("sr_acc", "sr_loss", "mlm_loss", "mlm_acc")

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                sentence_index=None, sentence_mask=None, sent_word_mask=None,
                mlm_ids=None, true_sent_ids=None, reverse_sentence_index=None,
                answers=None, pre_answers=None, **kwargs):

        hidden_sent, seq_output, sent_word_hidden = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            sentence_index=sentence_index,
            sentence_mask=sentence_mask,
            sent_word_mask=sent_word_mask
        )
        batch, sent_num, word_num = sent_word_mask.size()
        seq_length = seq_output.size(1)

        query_num = answers.size(1)
        query_h = hidden_sent[:, :query_num]

        attention_mask = attention_mask.to(query_h.dtype)

        # SR
        sr_query_h = self.sr_sent_sum(query_h)

        q_rel_d_sent_h, _ = layers.mul_sentence_sum(
            sr_query_h, sent_word_hidden, sent_word_mask
        )

        sr_scores = self.sr_prediction_head(
            self.sr_dropout(self.sr_pooler(q_rel_d_sent_h))).squeeze(-1)

        # MLM
        lm_query_h = self.lm_sent_sum(query_h)
        query_token_num = mlm_ids.size(1)

        q_rel_d_h, _ = layers.mul_weighted_sum(
            lm_query_h, seq_output, 1 - attention_mask
        )
        q_rel_d_h = self.lm_dropout(q_rel_d_h)

        aligned_sent_hidden = q_rel_d_h.gather(
            dim=1,
            index=reverse_sentence_index.unsqueeze(-1).expand(-1, -1, seq_output.size(-1))
        )

        # concat_word_hidden = torch.cat([seq_output[:, :query_token_num], aligned_sent_hidden], dim=-1)

        mlm_scores = self.lm_prediction_head(aligned_sent_hidden)

        output_dict = {}

        if mlm_ids is not None and answers is not None and pre_answers is not None:

            sent_mask = sentence_mask
            sent_mask = sent_mask.unsqueeze(1).expand(-1, query_num, -1)

            sr_scores = sr_scores + sent_mask * -10000.0

            fol_masked_scores = layers.mask_scores_with_labels(sr_scores, answers).contiguous()
            sr_loss1 = self.loss_fct(fol_masked_scores.view(batch * query_num, -1),
                                     pre_answers.view(-1))

            pre_masked_scores = layers.mask_scores_with_labels(sr_scores, pre_answers).contiguous()
            sr_loss2 = self.loss_fct(pre_masked_scores.view(batch * query_num, -1),
                                     answers.view(-1))

            mlm_loss = self.loss_fct(mlm_scores.view(-1, self.config.vocab_size),
                                     mlm_ids.view(-1))

            print(sr_loss1, sr_loss2, mlm_loss)

            loss = sr_loss1 + sr_loss2 + mlm_loss

            output_dict["loss"] = loss

            if not self.training:
                _, mlm_pred = mlm_scores.max(dim=-1)
                mlm_valid_num = (mlm_ids != -1).sum().item()
                mlm_acc = (mlm_pred == mlm_ids).sum().to(loss.dtype).item() / mlm_valid_num

                self.eval_metrics.update("mlm_loss", mlm_loss.item(), mlm_valid_num)
                self.eval_metrics.update("mlm_acc", mlm_acc, mlm_valid_num)

                valid_num1 = (answers != -1).sum().item()
                valid_num2 = (pre_answers != -1).sum().item()
                valid_num = valid_num1 + valid_num2

                _, pred = torch.topk(sr_scores, k=2, dim=-1, largest=True)

                acc1 = (pred == answers.unsqueeze(-1)).sum()
                acc2 = (pred == pre_answers.unsqueeze(-1)).sum()

                acc = (acc1 + acc2).to(dtype=sr_scores.dtype) / (valid_num * 1.0)

                output_dict["acc"] = acc
                output_dict["valid_num"] = valid_num

                self.eval_metrics.update("sr_acc", acc.item(), valid_num)
                self.eval_metrics.update("sr_loss", loss.item(), valid_num)

        return output_dict

    def get_eval_log(self, reset=False):
        _eval_metric_log = self.eval_metrics.get_log()
        _eval_metric_log = '\t'.join([f"{k}: {v}" for k, v in _eval_metric_log.items()])

        if reset:
            self.eval_metrics.reset()

        return _eval_metric_log


class QueryBertModelForPosQMLM(QueryBertModel):
    model_prefix = 'query_bert_pos_q_mlm'

    def __init__(self, config: BertQueryPreTrainedConfig):
        super().__init__(config)

        self.lm_sent_sum = nn.Linear(config.hidden_size, config.hidden_size)

        word_embedding_weight = deepcopy(self.bert.get_input_embeddings().weight)
        self.vocab_size = word_embedding_weight.size(0)

        config.layer_norm_eps = 1e-5  # avoid fp16 underflow
        self.lm_prediction_head = layers.PositionBasedSummaryForMLM(config,
                                                                    word_embedding_weight,
                                                                    config.pos_emb_size)

        self.lm_dropout = nn.Dropout(config.lm_query_dropout)

        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-1)

        self.init_weights()

        # metric
        self.eval_metrics = LogMetric("mlm_loss", "mlm_acc")

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                sentence_index=None, sentence_mask=None, sent_word_mask=None,
                mlm_ids=None, true_sent_ids=None, reverse_sentence_index=None,
                answers=None, pre_answers=None, **kwargs):

        hidden_sent, seq_output, sent_word_hidden = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            sentence_index=sentence_index,
            sentence_mask=sentence_mask,
            sent_word_mask=sent_word_mask
        )
        batch, sent_num, word_num = sent_word_mask.size()
        seq_length = seq_output.size(1)

        query_num = answers.size(1)
        query_h = hidden_sent[:, :query_num]

        attention_mask = attention_mask.to(query_h.dtype)

        # MLM
        lm_query_h = self.lm_sent_sum(query_h)
        query_token_num = mlm_ids.size(1)

        src_scores = lm_query_h.bmm(seq_output.transpose(1, 2))
        alighed_src_scores = src_scores.gather(
            dim=1,
            index=reverse_sentence_index.unsqueeze(-1).expand(-1, -1, src_scores.size(-1))
        )

        mlm_scores = self.lm_prediction_head(alighed_src_scores, seq_output, 1 - attention_mask, self.lm_dropout)

        output_dict = {}

        if mlm_ids is not None:

            mlm_loss = self.loss_fct(mlm_scores.view(-1, self.config.vocab_size),
                                     mlm_ids.view(-1))

            print(mlm_loss)

            loss = mlm_loss

            output_dict["loss"] = loss

            if not self.training:
                _, mlm_pred = mlm_scores.max(dim=-1)
                mlm_valid_num = (mlm_ids != -1).sum().item()
                mlm_acc = (mlm_pred == mlm_ids).sum().to(loss.dtype).item() / mlm_valid_num

                self.eval_metrics.update("mlm_loss", mlm_loss.item(), mlm_valid_num)
                self.eval_metrics.update("mlm_acc", mlm_acc, mlm_valid_num)

                output_dict["acc"] = mlm_acc
                output_dict["valid_num"] = mlm_valid_num

        return output_dict

    def get_eval_log(self, reset=False):
        _eval_metric_log = self.eval_metrics.get_log()
        _eval_metric_log = '\t'.join([f"{k}: {v}" for k, v in _eval_metric_log.items()])

        if reset:
            self.eval_metrics.reset()

        return _eval_metric_log


class QueryBertModelForSRAndPosQMLM(QueryBertModel):
    model_prefix = 'query_bert_sr_pos_q_mlm'

    def __init__(self, config: BertQueryPreTrainedConfig):
        super().__init__(config)

        self.sr_sent_sum = nn.Linear(config.hidden_size, config.hidden_size)
        self.lm_sent_sum = nn.Linear(config.hidden_size, config.hidden_size)

        word_embedding_weight = deepcopy(self.bert.get_input_embeddings().weight)
        self.vocab_size = word_embedding_weight.size(0)

        config.layer_norm_eps = 1e-5  # avoid fp16 underflow
        self.lm_prediction_head = layers.PositionBasedSummaryForMLM(config,
                                                                    word_embedding_weight,
                                                                    config.pos_emb_size)

        self.sr_pooler = layers.Pooler(config.hidden_size)
        self.sr_prediction_head = nn.Linear(config.hidden_size, 1)

        self.lm_dropout = nn.Dropout(config.lm_query_dropout)
        self.sr_dropout = nn.Dropout(config.sr_query_dropout)

        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-1)

        self.init_weights()

        # metric
        self.eval_metrics = LogMetric("sr_acc", "sr_loss", "mlm_loss", "mlm_acc")

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                sentence_index=None, sentence_mask=None, sent_word_mask=None,
                mlm_ids=None, true_sent_ids=None, reverse_sentence_index=None,
                answers=None, pre_answers=None, **kwargs):

        hidden_sent, seq_output, sent_word_hidden = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            sentence_index=sentence_index,
            sentence_mask=sentence_mask,
            sent_word_mask=sent_word_mask
        )
        batch, sent_num, word_num = sent_word_mask.size()
        seq_length = seq_output.size(1)

        query_num = answers.size(1)
        query_h = hidden_sent[:, :query_num]

        attention_mask = attention_mask.to(query_h.dtype)

        # SR
        sr_query_h = self.sr_sent_sum(query_h)

        q_rel_d_sent_h, _ = layers.mul_sentence_sum(
            sr_query_h, sent_word_hidden, sent_word_mask
        )

        sr_scores = self.sr_prediction_head(
            self.sr_dropout(self.sr_pooler(q_rel_d_sent_h))).squeeze(-1)

        # MLM
        lm_query_h = self.lm_sent_sum(query_h)
        query_token_num = mlm_ids.size(1)

        src_scores = lm_query_h.bmm(seq_output.transpose(1, 2))
        alighed_src_scores = src_scores.gather(
            dim=1,
            index=reverse_sentence_index.unsqueeze(-1).expand(-1, -1, src_scores.size(-1))
        )

        mlm_scores = self.lm_prediction_head(alighed_src_scores, seq_output, 1 - attention_mask, self.lm_dropout)

        output_dict = {}

        if mlm_ids is not None:

            sent_mask = sentence_mask
            sent_mask = sent_mask.unsqueeze(1).expand(-1, query_num, -1)

            sr_scores = sr_scores + sent_mask * -10000.0

            fol_masked_scores = layers.mask_scores_with_labels(sr_scores, answers).contiguous()
            sr_loss1 = self.loss_fct(fol_masked_scores.view(batch * query_num, -1),
                                     pre_answers.view(-1))

            pre_masked_scores = layers.mask_scores_with_labels(sr_scores, pre_answers).contiguous()
            sr_loss2 = self.loss_fct(pre_masked_scores.view(batch * query_num, -1),
                                     answers.view(-1))

            mlm_loss = self.loss_fct(mlm_scores.view(-1, self.config.vocab_size),
                                     mlm_ids.view(-1))

            print(sr_loss1, sr_loss2, mlm_loss)

            loss = sr_loss1 + sr_loss2 + mlm_loss

            output_dict["loss"] = loss

            if not self.training:
                _, mlm_pred = mlm_scores.max(dim=-1)
                mlm_valid_num = (mlm_ids != -1).sum().item()
                mlm_acc = (mlm_pred == mlm_ids).sum().to(loss.dtype).item() / mlm_valid_num

                self.eval_metrics.update("mlm_loss", mlm_loss.item(), mlm_valid_num)
                self.eval_metrics.update("mlm_acc", mlm_acc, mlm_valid_num)

                valid_num1 = (answers != -1).sum().item()
                valid_num2 = (pre_answers != -1).sum().item()
                valid_num = valid_num1 + valid_num2

                _, pred = torch.topk(sr_scores, k=2, dim=-1, largest=True)

                acc1 = (pred == answers.unsqueeze(-1)).sum()
                acc2 = (pred == pre_answers.unsqueeze(-1)).sum()

                acc = (acc1 + acc2).to(dtype=sr_scores.dtype) / (valid_num * 1.0)

                output_dict["acc"] = acc
                output_dict["valid_num"] = valid_num

                self.eval_metrics.update("sr_acc", acc.item(), valid_num)
                self.eval_metrics.update("sr_loss", loss.item(), valid_num)

        return output_dict

    def get_eval_log(self, reset=False):
        _eval_metric_log = self.eval_metrics.get_log()
        _eval_metric_log = '\t'.join([f"{k}: {v}" for k, v in _eval_metric_log.items()])

        if reset:
            self.eval_metrics.reset()

        return _eval_metric_log


class QueryBertModelForMCRC(QueryBertModel):
    model_prefix = 'query_bert_mcrc'

    def __init__(self, config: BertQueryPreTrainedConfig):
        super().__init__(config)

        self.sent_sum = nn.Linear(config.hidden_size, config.hidden_size)

        if config.cls_type == 1:
            self.pooler = nn.Sequential(
                nn.Linear(config.hidden_size * 2, config.hidden_size),
                nn.Tanh()
            )
        else:
            self.pooler = nn.Linear(config.hidden_size * 2, config.hidden_size)
        
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-1)

        self.init_weights()

    @staticmethod
    def fold_tensor(x):
        if x is None:
            return None
        return x.reshape(x.size(0) * x.size(1), *x.size()[2:])
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                sentence_index=None, sentence_mask=None, sent_word_mask=None,
                labels=None, **kwargs):
        
        batch, num_choice, _ = input_ids.size()
        input_ids = self.fold_tensor(input_ids)
        token_type_ids = self.fold_tensor(token_type_ids)
        attention_mask = self.fold_tensor(attention_mask)
        sentence_index = self.fold_tensor(sentence_index)
        sent_word_mask = self.fold_tensor(sent_word_mask)
        sentence_mask = self.fold_tensor(sentence_mask)

        seq_output = self.bert(input_ids=input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids)[0]

        fb, sent_num, seq_len = sent_word_mask.size()
        sentence_index = sentence_index.unsqueeze(-1).expand(
            -1, -1, -1, self.config.hidden_size
        ).reshape(fb, sent_num * seq_len, self.config.hidden_size)

        # cls_h = seq_output[:, :1]

        sent_word_hidden = seq_output.gather(dim=1, index=sentence_index).reshape(
            fb, sent_num, seq_len, -1)
        sent_word_hidden = sent_word_hidden * (1 - sent_word_mask.unsqueeze(-1))

        q_op_word_hidden = sent_word_hidden[:, :2].reshape(fb, 1, 2 * seq_len, -1)
        q_op_word_mask = sent_word_mask[:, :2].reshape(fb, 1, 2 * seq_len)
        q_op_hidden_sent = self.query(q_op_word_hidden, q_op_word_mask).view(fb, seq_output.size(-1))

        # =====================================

        q_op_query = self.sent_sum(q_op_hidden_sent)
        p_hidden_sent, _ = layers.sentence_sum(q_op_query, sent_word_hidden[:, 2:], sent_word_mask[:, 2:])
        p_hidden_sent = p_hidden_sent * (1 - sentence_mask[:, 2:].unsqueeze(-1))

        attended_h, _ = layers.weighted_sum(q_op_query, p_hidden_sent, sentence_mask[:, 2:])

        cls_input = torch.cat([q_op_hidden_sent, attended_h], dim=-1)
        logits = self.classifier(self.dropout(self.pooler(cls_input))).view(batch, num_choice)

        outputs = (logits,)
        if labels is not None:
            loss = self.loss_fct(logits, labels)
            outputs = (loss,) + outputs

            _, pred = logits.max(dim=-1)
            acc = torch.sum(pred == labels) / (1.0 * batch)

            outputs = outputs + (acc,)

        return outputs


query_bert_models_map = {
    QueryBertModelForSRAndMLM.model_prefix: QueryBertModelForSRAndMLM,
    QueryBertModelForSRAndPosMLM.model_prefix: QueryBertModelForSRAndPosMLM,
    QueryBertModelForMCRC.model_prefix: QueryBertModelForMCRC,
    QueryBertModelForBiSRAndMLM.model_prefix: QueryBertModelForBiSRAndMLM,
    
    QueryBertModelForPosQMLM.model_prefix: QueryBertModelForPosQMLM,
    QueryBertModelForSRAndPosQMLM.model_prefix: QueryBertModelForSRAndPosQMLM
}
