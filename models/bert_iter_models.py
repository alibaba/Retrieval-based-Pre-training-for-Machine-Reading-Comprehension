import torch
from torch import nn
from transformers.modeling_bert import BertConfig, BertPreTrainedModel, BertModel, BertForMaskedLM, \
    BertForQuestionAnswering, QuestionAnsweringModelOutput

from general_util.logger import get_child_logger
from general_util.mixin import LogMixin, PredictionMixin
from modules import layers

logger = get_child_logger(__name__)


class BertForMaskedLMBaseline(BertForMaskedLM, LogMixin):
    model_prefix = 'bert_mlm_baseline'

    def __init__(self, config: BertConfig):
        super().__init__(config)

        self.config = config

        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-1)

        self.init_metric("mlm_acc", "mlm_loss")

        logger.info(self.config.to_dict())

    def forward(self, input_ids: torch.Tensor = None,
                attention_mask: torch.Tensor = None,
                token_type_ids: torch.Tensor = None,
                labels: torch.Tensor = None,
                **kwargs):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        output_dict = {}
        if labels is not None:
            masked_lm_loss = self.loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

            print(masked_lm_loss)

            output_dict["loss"] = masked_lm_loss

            if not self.training:
                valid_num = (labels != -1).sum().item()
                _, mlm_pred = prediction_scores.max(dim=-1)
                mlm_acc = (mlm_pred == labels).sum().to(masked_lm_loss.dtype) / valid_num

                self.eval_metrics.update("mlm_loss", masked_lm_loss.item(), valid_num)
                self.eval_metrics.update("mlm_acc", mlm_acc.item(), valid_num)

                output_dict["acc"] = mlm_acc
                output_dict["valid_num"] = valid_num

        return output_dict


class IterBertPreTrainedConfig(BertConfig):
    added_configs = [
        'query_dropout', 'cls_type', 'sr_query_dropout', 'lm_query_dropout', 'pos_emb_size',
        'z_step', 'num_labels', 'share_mlm_sum', 'share_ssp_sum', 'word_dropout'
    ]

    def __init__(self, query_dropout=0.1, cls_type=0,
                 sr_query_dropout=0.1, lm_query_dropout=0.1,
                 pos_emb_size=200, z_step=0, num_labels=2,
                 share_mlm_sum=False, share_ssp_sum=False,
                 word_dropout=0.0, **kwargs):
        super().__init__(**kwargs)

        self.query_dropout = query_dropout
        self.cls_type = cls_type
        self.sr_query_dropout = sr_query_dropout
        self.lm_query_dropout = lm_query_dropout
        self.pos_emb_size = pos_emb_size
        self.z_step = z_step
        self.num_labels = num_labels
        self.share_mlm_sum = share_mlm_sum
        self.share_ssp_sum = share_ssp_sum
        self.word_dropout = word_dropout

    def expand_configs(self, *args):
        self.added_configs.extend(list(args))


class IterBertModel(BertPreTrainedModel):
    config_class = IterBertPreTrainedConfig
    model_prefix = 'iter_bert'

    def __init__(self, config: IterBertPreTrainedConfig):
        super().__init__(config)

        self.config = config
        self.bert = BertModel(config)

        config.layer_norm_eps = 1e-5
        self.query = layers.MultiHeadAlignedTokenAttention(
            config,
            attn_dropout_p=config.query_dropout,
            dropout_p=config.query_dropout
        )
        self.z_step = config.z_step

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

        q_vec = seq_output[:, :1]  # [CLS]
        for _step in range(self.z_step):
            if _step == 0:
                _aligned = False
            else:
                _aligned = True
            q_vec = self.query(q_vec, sent_word_hidden, sent_word_mask, aligned=_aligned, residual=False)
            if _step == 0:
                q_vec = q_vec.squeeze(1)

        hidden_sent = q_vec
        assert hidden_sent.size() == (batch, sent_num, seq_output.size(-1))

        return hidden_sent, seq_output, sent_word_hidden


class IterBertModelForBiSR(IterBertModel, LogMixin):
    model_prefix = 'iter_bert_bi_sr'

    def __init__(self, config: IterBertPreTrainedConfig):
        super().__init__(config)

        self.sr_sent_sum = nn.Linear(config.hidden_size, config.hidden_size)

        self.pre_sr_pooler = layers.Pooler(config.hidden_size)
        self.pre_sr_prediction_head = nn.Linear(config.hidden_size, 1)

        self.fol_sr_pooler = layers.Pooler(config.hidden_size)
        self.fol_sr_prediction_head = nn.Linear(config.hidden_size, 1)

        self.sr_dropout = nn.Dropout(config.sr_query_dropout)

        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-1)

        self.init_weights()

        # metric
        self.init_metric("sr_acc", "sr_loss")

        logger.info(self.config.to_dict())

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                sentence_index=None, sentence_mask=None, sent_word_mask=None,
                mlm_ids=None, true_sent_ids=None, reverse_sentence_index=None,
                answers: torch.Tensor = None, pre_answers: torch.Tensor = None, **kwargs):

        hidden_sent, seq_output, sent_word_hidden = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            sentence_index=sentence_index,
            sentence_mask=sentence_mask,
            sent_word_mask=sent_word_mask
        )
        batch, sent_num, word_num = sent_word_mask.size()

        query_num = answers.size(1)
        query_h = hidden_sent[:, :query_num]

        # SR
        sr_query_h = self.sr_sent_sum(query_h)

        q_rel_d_sent_h, _ = layers.mul_sentence_sum(
            sr_query_h, sent_word_hidden, sent_word_mask
        )

        pre_sr_scores = self.pre_sr_prediction_head(
            self.sr_dropout(self.pre_sr_pooler(q_rel_d_sent_h))).squeeze(-1)

        fol_sr_scores = self.fol_sr_prediction_head(
            self.sr_dropout(self.fol_sr_pooler(q_rel_d_sent_h))).squeeze(-1)

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

            print(sr_loss1, sr_loss2)

            loss = sr_loss1 + sr_loss2

            output_dict["loss"] = loss

            if not self.training:
                valid_num1 = (answers != -1).sum().item()
                valid_num2 = (pre_answers != -1).sum().item()
                valid_num = valid_num1 + valid_num2

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


class IterBertModelForBiSRAndMLM(IterBertModel, LogMixin):
    model_prefix = 'iter_bert_bi_sr_mlm'

    def __init__(self, config: IterBertPreTrainedConfig):
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

        self.sr_dropout = nn.Dropout(config.sr_query_dropout)
        self.lm_dropout = nn.Dropout(config.lm_query_dropout)

        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-1)

        self.init_weights()

        # metric
        self.init_metric("sr_acc", "sr_loss", "mlm_loss", "mlm_acc")

        logger.info(self.config.to_dict())

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                sentence_index=None, sentence_mask=None, sent_word_mask=None,
                mlm_ids=None, true_sent_ids=None, reverse_sentence_index=None,
                answers: torch.Tensor = None, pre_answers: torch.Tensor = None, **kwargs):

        hidden_sent, seq_output, sent_word_hidden = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            sentence_index=sentence_index,
            sentence_mask=sentence_mask,
            sent_word_mask=sent_word_mask
        )
        batch, sent_num, word_num = sent_word_mask.size()

        query_num = answers.size(1)
        query_h = hidden_sent[:, :query_num]

        attention_mask = attention_mask.to(query_h.dtype)

        # SR
        sr_query_h = self.sr_sent_sum(query_h)

        q_rel_d_sent_h, _ = layers.mul_sentence_sum(
            sr_query_h, sent_word_hidden, sent_word_mask
        )

        pre_sr_scores = self.pre_sr_prediction_head(
            self.sr_dropout(self.pre_sr_pooler(q_rel_d_sent_h))).squeeze(-1)

        fol_sr_scores = self.fol_sr_prediction_head(
            self.sr_dropout(self.fol_sr_pooler(q_rel_d_sent_h))).squeeze(-1)

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


class IterBertModelForSRAndMLM(IterBertModel, LogMixin):
    model_prefix = 'iter_bert_sr_mlm'

    def __init__(self, config: IterBertPreTrainedConfig):
        super().__init__(config)

        self.sr_sent_sum = nn.Linear(config.hidden_size, config.hidden_size)
        self.lm_sent_sum = nn.Linear(config.hidden_size, config.hidden_size)

        word_embedding_weight = self.bert.get_input_embeddings().weight
        self.vocab_size = word_embedding_weight.size(0)

        config.layer_norm_eps = 1e-5  # avoid fp16 underflow
        self.lm_prediction_head = layers.MaskedLMPredictionHead(config, word_embedding_weight)

        self.sr_pooler = layers.Pooler(config.hidden_size)
        self.sr_prediction_head = nn.Linear(config.hidden_size, 1)

        self.sr_dropout = nn.Dropout(config.sr_query_dropout)
        self.lm_dropout = nn.Dropout(config.lm_query_dropout)

        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-1)

        self.init_weights()

        # metric
        self.init_metric("sr_acc", "sr_loss", "mlm_loss", "mlm_acc")

        logger.info(self.config.to_dict())

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                sentence_index=None, sentence_mask=None, sent_word_mask=None,
                mlm_ids=None, true_sent_ids=None, reverse_sentence_index=None,
                answers: torch.Tensor = None, pre_answers: torch.Tensor = None, **kwargs):

        hidden_sent, seq_output, sent_word_hidden = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            sentence_index=sentence_index,
            sentence_mask=sentence_mask,
            sent_word_mask=sent_word_mask
        )
        batch, sent_num, word_num = sent_word_mask.size()

        query_num = answers.size(1)
        query_h = hidden_sent[:, :query_num]

        attention_mask = attention_mask.to(query_h.dtype)

        # SR
        sr_query_h = self.sr_sent_sum(query_h)

        q_rel_d_sent_h, _ = layers.mul_sentence_sum(
            sr_query_h, sent_word_hidden, sent_word_mask
        )

        sr_scores = self.sr_prediction_head(
            self.sr_dropout(self.sr_pooler(q_rel_d_sent_h))
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


class IterBertModelForSR(IterBertModel, LogMixin):
    model_prefix = 'iter_bert_sr'

    def __init__(self, config: IterBertPreTrainedConfig):
        super().__init__(config)

        self.sr_sent_sum = nn.Linear(config.hidden_size, config.hidden_size)

        self.sr_pooler = layers.Pooler(config.hidden_size)
        self.sr_prediction_head = nn.Linear(config.hidden_size, 1)

        self.sr_dropout = nn.Dropout(config.sr_query_dropout)

        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-1)

        self.init_weights()

        # metric
        self.init_metric("sr_acc", "sr_loss")

        logger.info(self.config.to_dict())

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                sentence_index=None, sentence_mask=None, sent_word_mask=None,
                mlm_ids=None, true_sent_ids=None, reverse_sentence_index=None,
                answers: torch.Tensor = None, pre_answers: torch.Tensor = None, **kwargs):

        hidden_sent, seq_output, sent_word_hidden = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            sentence_index=sentence_index,
            sentence_mask=sentence_mask,
            sent_word_mask=sent_word_mask
        )
        batch, sent_num, word_num = sent_word_mask.size()

        query_num = answers.size(1)
        query_h = hidden_sent[:, :query_num]

        attention_mask = attention_mask.to(query_h.dtype)

        # SR
        sr_query_h = self.sr_sent_sum(query_h)

        q_rel_d_sent_h, _ = layers.mul_sentence_sum(
            sr_query_h, sent_word_hidden, sent_word_mask
        )

        sr_scores = self.sr_prediction_head(
            self.sr_dropout(self.sr_pooler(q_rel_d_sent_h))
        ).squeeze(-1)

        output_dict = {}

        if answers is not None and pre_answers is not None:

            sent_mask = sentence_mask
            sent_mask = sent_mask.unsqueeze(1).expand(-1, query_num, -1)

            sr_scores = sr_scores + sent_mask * -10000.0

            fol_masked_scores = layers.mask_scores_with_labels(sr_scores, answers).contiguous()
            sr_loss1 = self.loss_fct(fol_masked_scores.view(batch * query_num, -1),
                                     pre_answers.view(-1))

            pre_masked_scores = layers.mask_scores_with_labels(sr_scores, pre_answers).contiguous()
            sr_loss2 = self.loss_fct(pre_masked_scores.view(batch * query_num, -1),
                                     answers.view(-1))

            print(sr_loss1, sr_loss2)

            loss = sr_loss1 + sr_loss2

            output_dict["loss"] = loss

            if not self.training:
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


class IterBertModelForMLM(IterBertModel, LogMixin):
    model_prefix = 'iter_bert_mlm'

    def __init__(self, config: IterBertPreTrainedConfig):
        super().__init__(config)

        self.lm_sent_sum = nn.Linear(config.hidden_size, config.hidden_size)

        word_embedding_weight = self.bert.get_input_embeddings().weight
        self.vocab_size = word_embedding_weight.size(0)

        config.layer_norm_eps = 1e-5  # avoid fp16 underflow
        self.lm_prediction_head = layers.MaskedLMPredictionHead(config, word_embedding_weight)

        self.lm_dropout = nn.Dropout(config.lm_query_dropout)

        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-1)

        self.init_weights()

        # metric
        self.init_metric("mlm_loss", "mlm_acc")

        logger.info(self.config.to_dict())

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                sentence_index=None, sentence_mask=None, sent_word_mask=None,
                mlm_ids: torch.Tensor = None, true_sent_ids=None, reverse_sentence_index=None,
                answers: torch.Tensor = None, pre_answers: torch.Tensor = None, **kwargs):

        hidden_sent, seq_output, sent_word_hidden = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            sentence_index=sentence_index,
            sentence_mask=sentence_mask,
            sent_word_mask=sent_word_mask
        )
        batch, sent_num, word_num = sent_word_mask.size()

        query_num = answers.size(1)
        query_h = hidden_sent[:, :query_num]

        attention_mask = attention_mask.to(query_h.dtype)

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

        if mlm_ids is not None:

            mlm_loss = self.loss_fct(mlm_scores.view(-1, self.config.vocab_size),
                                     mlm_ids.view(-1))

            print(mlm_loss)

            loss = mlm_loss

            output_dict["loss"] = loss

            if not self.training:
                _, mlm_pred = mlm_scores.max(dim=-1)
                mlm_valid_num = (mlm_ids != -1).sum().item()
                mlm_acc = (mlm_pred == mlm_ids).sum().to(loss.dtype) / mlm_valid_num

                self.eval_metrics.update("mlm_loss", mlm_loss.item(), mlm_valid_num)
                self.eval_metrics.update("mlm_acc", mlm_acc.item(), mlm_valid_num)

                output_dict["acc"] = mlm_acc
                output_dict["valid_num"] = mlm_valid_num

        return output_dict


class IterBertModelForMCRC(IterBertModel):
    model_prefix = 'iter_bert_mcrc'

    def __init__(self, config: IterBertPreTrainedConfig):
        super().__init__(config)

        self.sent_sum = nn.Linear(config.hidden_size, config.hidden_size)
        if config.share_ssp_sum:
            self.sr_sent_sum = nn.Linear(config.hidden_size, config.hidden_size)
            self.sent_sum = self.sr_sent_sum

        if config.word_dropout > 0:
            self.word_dropout = nn.Dropout(config.word_dropout)
        else:
            self.word_dropout = lambda x: x

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

        cls_h = seq_output[:, :1]

        sent_word_hidden = seq_output.gather(dim=1, index=sentence_index).reshape(
            fb, sent_num, seq_len, -1)
        sent_word_hidden = sent_word_hidden * (1 - sent_word_mask.unsqueeze(-1))

        q_op_word_hidden = sent_word_hidden[:, :2].reshape(fb, 1, 2 * seq_len, -1)
        q_op_word_mask = sent_word_mask[:, :2].reshape(fb, 1, 2 * seq_len)
        q_op_hidden_sent = self.query(cls_h, q_op_word_hidden, q_op_word_mask,
                                      aligned=False, residual=False).view(fb, seq_output.size(-1))

        # =====================================

        q_op_query = self.sent_sum(q_op_hidden_sent)
        p_hidden_sent, _ = layers.sentence_sum(
            q=q_op_query,
            kv=sent_word_hidden[:, 2:],
            mask=sent_word_mask[:, 2:],
            _dropout=self.word_dropout
        )
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


class IterBertModelForMCRCDropout(IterBertModel):
    model_prefix = 'iter_bert_mcrc_d'

    def __init__(self, config: IterBertPreTrainedConfig):
        super().__init__(config)

        self.sent_sum = nn.Linear(config.hidden_size, config.hidden_size)
        if config.share_ssp_sum:
            self.sr_sent_sum = nn.Linear(config.hidden_size, config.hidden_size)
            self.sent_sum = self.sr_sent_sum

        if config.word_dropout > 0:
            self.word_dropout = nn.Dropout(config.word_dropout)
        else:
            self.word_dropout = lambda x: x

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

        cls_h = seq_output[:, :1]

        sent_word_hidden = seq_output.gather(dim=1, index=sentence_index).reshape(
            fb, sent_num, seq_len, -1)
        sent_word_hidden = sent_word_hidden * (1 - sent_word_mask.unsqueeze(-1))

        q_op_word_hidden = sent_word_hidden[:, :2].reshape(fb, 1, 2 * seq_len, -1)
        q_op_word_mask = sent_word_mask[:, :2].reshape(fb, 1, 2 * seq_len)
        q_op_hidden_sent = self.query(cls_h, q_op_word_hidden, q_op_word_mask,
                                      aligned=False, residual=False).view(fb, seq_output.size(-1))

        # =====================================

        q_op_query = self.sent_sum(q_op_hidden_sent)
        p_hidden_sent, _ = layers.sentence_sum(
            q=q_op_query,
            kv=self.word_dropout(sent_word_hidden[:, 2:]),
            mask=sent_word_mask[:, 2:],
            v=sent_word_hidden[:, 2:]
        )
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


class IterBertModelForMCRC2(IterBertModel):
    model_prefix = 'iter_bert_mcrc2'

    def __init__(self, config: IterBertPreTrainedConfig):
        super().__init__(config)

        self.sent_sum = nn.Linear(config.hidden_size, config.hidden_size)
        if config.share_ssp_sum:
            self.sr_sent_sum = nn.Linear(config.hidden_size, config.hidden_size)
            self.sent_sum = self.sr_sent_sum

        self.doc_sum = nn.Linear(config.hidden_size, config.hidden_size)

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

        cls_h = seq_output[:, :1]

        sent_word_hidden = seq_output.gather(dim=1, index=sentence_index).reshape(
            fb, sent_num, seq_len, -1)
        sent_word_hidden = sent_word_hidden * (1 - sent_word_mask.unsqueeze(-1))

        q_op_word_hidden = sent_word_hidden[:, :2].reshape(fb, 1, 2 * seq_len, -1)
        q_op_word_mask = sent_word_mask[:, :2].reshape(fb, 1, 2 * seq_len)
        q_op_hidden_sent = self.query(cls_h, q_op_word_hidden, q_op_word_mask,
                                      aligned=False, residual=False).view(fb, seq_output.size(-1))

        # =====================================

        q_op_query1 = self.sent_sum(q_op_hidden_sent)
        p_hidden_sent, _ = layers.sentence_sum(q_op_query1, sent_word_hidden[:, 2:], sent_word_mask[:, 2:])
        p_hidden_sent = p_hidden_sent * (1 - sentence_mask[:, 2:].unsqueeze(-1))

        q_op_query2 = self.doc_sum(q_op_hidden_sent)
        attended_h, _ = layers.weighted_sum(q_op_query2, p_hidden_sent, sentence_mask[:, 2:])

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


class IterBertModelForMCRC3(IterBertModel):
    model_prefix = 'iter_bert_mcrc3'

    def __init__(self, config: IterBertPreTrainedConfig):
        super().__init__(config)

        self.sen_sum_q = nn.Linear(config.hidden_size, config.hidden_size)
        self.sen_sum_k = nn.Linear(config.hidden_size, config.hidden_size)
        self.doc_sum_q = nn.Linear(config.hidden_size, config.hidden_size)
        self.doc_sum_k = nn.Linear(config.hidden_size, config.hidden_size)

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

        cls_h = seq_output[:, :1]

        sent_word_hidden = seq_output.gather(dim=1, index=sentence_index).reshape(
            fb, sent_num, seq_len, -1)
        sent_word_hidden = sent_word_hidden * (1 - sent_word_mask.unsqueeze(-1))

        q_op_word_hidden = sent_word_hidden[:, :2].reshape(fb, 1, 2 * seq_len, -1)
        q_op_word_mask = sent_word_mask[:, :2].reshape(fb, 1, 2 * seq_len)
        q_op_hidden_sent = self.query(cls_h, q_op_word_hidden, q_op_word_mask,
                                      aligned=False, residual=False).view(fb, seq_output.size(-1))

        # =====================================

        p_hidden_sent, _ = layers.sentence_sum(
            q=self.sen_sum_q(q_op_hidden_sent),
            kv=self.sen_sum_k(sent_word_hidden[:, 2:]),
            mask=sent_word_mask[:, 2:]
        )
        p_hidden_sent = p_hidden_sent * (1 - sentence_mask[:, 2:].unsqueeze(-1))

        attended_h, _scores = layers.weighted_sum(
            q=self.doc_sum_q(q_op_hidden_sent),
            kv=self.doc_sum_k(p_hidden_sent),
            mask=sentence_mask[:, 2:]
        )

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


class IterBertModelForMCRC4(IterBertModel):
    model_prefix = 'iter_bert_mcrc4'

    def __init__(self, config: IterBertPreTrainedConfig):
        super().__init__(config)

        self.sen_sum_q = nn.Linear(config.hidden_size, config.hidden_size)
        self.sen_sum_k = nn.Linear(config.hidden_size, config.hidden_size)
        self.doc_sum_q = nn.Linear(config.hidden_size, config.hidden_size)
        self.doc_sum_k = nn.Linear(config.hidden_size, config.hidden_size)

        if config.word_dropout > 0:
            self.word_dropout = nn.Dropout(config.word_dropout)
        else:
            self.word_dropout = lambda x: x

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

        cls_h = seq_output[:, :1]

        sent_word_hidden = seq_output.gather(dim=1, index=sentence_index).reshape(
            fb, sent_num, seq_len, -1)
        sent_word_hidden = sent_word_hidden * (1 - sent_word_mask.unsqueeze(-1))

        q_op_word_hidden = sent_word_hidden[:, :2].reshape(fb, 1, 2 * seq_len, -1)
        q_op_word_mask = sent_word_mask[:, :2].reshape(fb, 1, 2 * seq_len)
        q_op_hidden_sent = self.query(cls_h, q_op_word_hidden, q_op_word_mask,
                                      aligned=False, residual=False).view(fb, seq_output.size(-1))

        # =====================================

        p_hidden_sent, _ = layers.sentence_sum(
            q=self.sen_sum_q(q_op_hidden_sent),
            kv=self.sen_sum_k(sent_word_hidden[:, 2:]),
            mask=sent_word_mask[:, 2:],
            v=sent_word_hidden[:, 2:],
            _dropout=self.word_dropout
        )
        p_hidden_sent = p_hidden_sent * (1 - sentence_mask[:, 2:].unsqueeze(-1))

        attended_h, _scores = layers.weighted_sum(
            q=self.doc_sum_q(q_op_hidden_sent),
            kv=self.doc_sum_k(p_hidden_sent),
            mask=sentence_mask[:, 2:],
            v=p_hidden_sent
        )

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


class IterBertModelForSequenceClassification(IterBertModel, PredictionMixin):
    model_prefix = 'iter_bert_sc'

    def __init__(self, config: IterBertPreTrainedConfig):
        super().__init__(config)

        self.sent_sum = nn.Linear(config.hidden_size, config.hidden_size)

        if config.cls_type == 1:
            self.pooler = nn.Sequential(
                nn.Linear(config.hidden_size * 2, config.hidden_size),
                nn.Tanh()
            )
        else:
            self.pooler = nn.Linear(config.hidden_size * 2, config.hidden_size)

        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-1)

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                sentence_index=None, sentence_mask=None, sent_word_mask=None,
                labels=None, **kwargs):

        seq_output = self.bert(input_ids=input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids)[0]

        batch, sent_num, seq_len = sent_word_mask.size()
        sentence_index = sentence_index.unsqueeze(-1).expand(
            -1, -1, -1, self.config.hidden_size
        ).reshape(batch, sent_num * seq_len, self.config.hidden_size)

        cls_h = seq_output[:, :1]

        sent_word_hidden = seq_output.gather(dim=1, index=sentence_index).reshape(
            batch, sent_num, seq_len, -1)
        sent_word_hidden = sent_word_hidden * (1 - sent_word_mask.unsqueeze(-1))

        q_op_word_hidden = sent_word_hidden[:, :2].reshape(batch, 1, 2 * seq_len, -1)
        q_op_word_mask = sent_word_mask[:, :2].reshape(batch, 1, 2 * seq_len)
        q_op_hidden_sent = self.query(cls_h, q_op_word_hidden, q_op_word_mask,
                                      aligned=False, residual=False).view(batch, seq_output.size(-1))

        # =====================================

        q_op_query = self.sent_sum(q_op_hidden_sent)
        p_hidden_sent, _ = layers.sentence_sum(q_op_query, sent_word_hidden[:, 2:], sent_word_mask[:, 2:])
        p_hidden_sent = p_hidden_sent * (1 - sentence_mask[:, 2:].unsqueeze(-1))

        attended_h, _scores = layers.weighted_sum(q_op_query, p_hidden_sent, sentence_mask[:, 2:])

        cls_input = torch.cat([q_op_hidden_sent, attended_h], dim=-1)
        logits = self.classifier(self.dropout(self.pooler(cls_input)))

        outputs = (logits,)
        if labels is not None:
            loss = self.loss_fct(logits, labels)
            outputs = (loss,) + outputs

            _, pred = logits.max(dim=-1)
            acc = torch.sum(pred == labels) / (1.0 * batch)

            outputs = outputs + (acc,)

        # prediction utils
        if not self.training:
            self.concat_predict_tensors(sentence_logits=_scores,
                                        sent_word_ids=input_ids.gather(dim=1, index=sentence_index[:, :, 0]).reshape(
                                            batch, sent_num, seq_len))

        return outputs


class IterBertModelForSequenceClassificationV2(IterBertModel, PredictionMixin):
    model_prefix = 'iter_bert_sc_v2'

    def __init__(self, config: IterBertPreTrainedConfig):
        super().__init__(config)

        self.sent_sum = nn.Linear(config.hidden_size, config.hidden_size)
        self.doc_sum = nn.Linear(config.hidden_size, config.hidden_size)

        if config.cls_type == 1:
            self.pooler = nn.Sequential(
                nn.Linear(config.hidden_size * 2, config.hidden_size),
                nn.Tanh()
            )
        else:
            self.pooler = nn.Linear(config.hidden_size * 2, config.hidden_size)

        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-1)

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                sentence_index=None, sentence_mask=None, sent_word_mask=None,
                labels=None, **kwargs):

        seq_output = self.bert(input_ids=input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids)[0]

        batch, sent_num, seq_len = sent_word_mask.size()
        sentence_index = sentence_index.unsqueeze(-1).expand(
            -1, -1, -1, self.config.hidden_size
        ).reshape(batch, sent_num * seq_len, self.config.hidden_size)

        cls_h = seq_output[:, :1]

        sent_word_hidden = seq_output.gather(dim=1, index=sentence_index).reshape(
            batch, sent_num, seq_len, -1)
        sent_word_hidden = sent_word_hidden * (1 - sent_word_mask.unsqueeze(-1))

        q_op_word_hidden = sent_word_hidden[:, :2].reshape(batch, 1, 2 * seq_len, -1)
        q_op_word_mask = sent_word_mask[:, :2].reshape(batch, 1, 2 * seq_len)
        q_op_hidden_sent = self.query(cls_h, q_op_word_hidden, q_op_word_mask,
                                      aligned=False, residual=False).view(batch, seq_output.size(-1))

        # =====================================

        q_op_query1 = self.sent_sum(q_op_hidden_sent)
        p_hidden_sent, _ = layers.sentence_sum(q_op_query1, sent_word_hidden[:, 2:], sent_word_mask[:, 2:])
        p_hidden_sent = p_hidden_sent * (1 - sentence_mask[:, 2:].unsqueeze(-1))

        q_op_query2 = self.doc_sum(q_op_hidden_sent)
        attended_h, _scores = layers.weighted_sum(q_op_query2, p_hidden_sent, sentence_mask[:, 2:])

        cls_input = torch.cat([q_op_hidden_sent, attended_h], dim=-1)
        logits = self.classifier(self.dropout(self.pooler(cls_input)))

        outputs = (logits,)
        if labels is not None:
            loss = self.loss_fct(logits, labels)
            outputs = (loss,) + outputs

            _, pred = logits.max(dim=-1)
            acc = torch.sum(pred == labels) / (1.0 * batch)

            outputs = outputs + (acc,)

        # prediction utils
        if not self.training:
            self.concat_predict_tensors(sentence_logits=_scores,
                                        sent_word_ids=input_ids.gather(dim=1, index=sentence_index[:, :, 0]).reshape(
                                            batch, sent_num, seq_len))

        return outputs


class IterBertModelForSequenceClassificationV3(IterBertModel, PredictionMixin):
    model_prefix = 'iter_bert_sc_v3'

    def __init__(self, config: IterBertPreTrainedConfig):
        super().__init__(config)

        self.sen_sum_q = nn.Linear(config.hidden_size, config.hidden_size)
        self.sen_sum_k = nn.Linear(config.hidden_size, config.hidden_size)
        self.doc_sum_q = nn.Linear(config.hidden_size, config.hidden_size)
        self.doc_sum_k = nn.Linear(config.hidden_size, config.hidden_size)

        if config.cls_type == 1:
            self.pooler = nn.Sequential(
                nn.Linear(config.hidden_size * 2, config.hidden_size),
                nn.Tanh()
            )
        else:
            self.pooler = nn.Linear(config.hidden_size * 2, config.hidden_size)

        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-1)

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                sentence_index=None, sentence_mask=None, sent_word_mask=None,
                labels=None, **kwargs):

        seq_output = self.bert(input_ids=input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids)[0]

        batch, sent_num, seq_len = sent_word_mask.size()
        sentence_index = sentence_index.unsqueeze(-1).expand(
            -1, -1, -1, self.config.hidden_size
        ).reshape(batch, sent_num * seq_len, self.config.hidden_size)

        cls_h = seq_output[:, :1]

        sent_word_hidden = seq_output.gather(dim=1, index=sentence_index).reshape(
            batch, sent_num, seq_len, -1)
        sent_word_hidden = sent_word_hidden * (1 - sent_word_mask.unsqueeze(-1))

        q_op_word_hidden = sent_word_hidden[:, :2].reshape(batch, 1, 2 * seq_len, -1)
        q_op_word_mask = sent_word_mask[:, :2].reshape(batch, 1, 2 * seq_len)
        q_op_hidden_sent = self.query(cls_h, q_op_word_hidden, q_op_word_mask,
                                      aligned=False, residual=False).view(batch, seq_output.size(-1))

        # =====================================

        p_hidden_sent, _ = layers.sentence_sum(
            q=self.sen_sum_q(q_op_hidden_sent),
            kv=self.sen_sum_k(sent_word_hidden[:, 2:]),
            mask=sent_word_mask[:, 2:]
        )
        p_hidden_sent = p_hidden_sent * (1 - sentence_mask[:, 2:].unsqueeze(-1))

        attended_h, _scores = layers.weighted_sum(
            q=self.doc_sum_q(q_op_hidden_sent),
            kv=self.doc_sum_k(p_hidden_sent),
            mask=sentence_mask[:, 2:]
        )

        cls_input = torch.cat([q_op_hidden_sent, attended_h], dim=-1)
        logits = self.classifier(self.dropout(self.pooler(cls_input)))

        outputs = (logits,)
        if labels is not None:
            loss = self.loss_fct(logits, labels)
            outputs = (loss,) + outputs

            _, pred = logits.max(dim=-1)
            acc = torch.sum(pred == labels) / (1.0 * batch)

            outputs = outputs + (acc,)

        # prediction utils
        if not self.training:
            self.concat_predict_tensors(sentence_logits=_scores.float(),
                                        sent_word_ids=input_ids.gather(dim=1, index=sentence_index[:, :, 0]).reshape(
                                            batch, sent_num, seq_len).int())

        return outputs


class IterBertModelForSequenceClassificationV4(IterBertModel, PredictionMixin):
    model_prefix = 'iter_bert_sc_v4'

    def __init__(self, config: IterBertPreTrainedConfig):
        super().__init__(config)

        self.sen_sum_q = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.sen_sum_k = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.doc_sum_q = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.doc_sum_k = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

        if config.cls_type == 1:
            self.pooler = nn.Sequential(
                nn.Linear(config.hidden_size * 2, config.hidden_size),
                nn.Tanh()
            )
        else:
            self.pooler = nn.Linear(config.hidden_size * 2, config.hidden_size)

        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-1)

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                sentence_index=None, sentence_mask=None, sent_word_mask=None,
                labels=None, **kwargs):

        seq_output = self.bert(input_ids=input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids)[0]

        batch, sent_num, seq_len = sent_word_mask.size()
        sentence_index = sentence_index.unsqueeze(-1).expand(
            -1, -1, -1, self.config.hidden_size
        ).reshape(batch, sent_num * seq_len, self.config.hidden_size)

        cls_h = seq_output[:, :1]

        sent_word_hidden = seq_output.gather(dim=1, index=sentence_index).reshape(
            batch, sent_num, seq_len, -1)
        sent_word_hidden = sent_word_hidden * (1 - sent_word_mask.unsqueeze(-1))

        q_op_word_hidden = sent_word_hidden[:, :2].reshape(batch, 1, 2 * seq_len, -1)
        q_op_word_mask = sent_word_mask[:, :2].reshape(batch, 1, 2 * seq_len)
        q_op_hidden_sent = self.query(cls_h, q_op_word_hidden, q_op_word_mask,
                                      aligned=False, residual=False).view(batch, seq_output.size(-1))

        # =====================================

        p_hidden_sent, _ = layers.sentence_sum(
            q=self.sen_sum_q(q_op_hidden_sent),
            kv=self.sen_sum_k(sent_word_hidden[:, 2:]),
            v=sent_word_hidden[:, 2:],
            mask=sent_word_mask[:, 2:]
        )
        p_hidden_sent = p_hidden_sent * (1 - sentence_mask[:, 2:].unsqueeze(-1))

        attended_h, _scores = layers.weighted_sum(
            q=self.doc_sum_q(q_op_hidden_sent),
            kv=self.doc_sum_k(p_hidden_sent),
            v=p_hidden_sent,
            mask=sentence_mask[:, 2:]
        )

        cls_input = torch.cat([q_op_hidden_sent, attended_h], dim=-1)
        logits = self.classifier(self.dropout(self.pooler(cls_input)))

        outputs = (logits,)
        if labels is not None:
            loss = self.loss_fct(logits, labels)
            outputs = (loss,) + outputs

            _, pred = logits.max(dim=-1)
            acc = torch.sum(pred == labels) / (1.0 * batch)

            outputs = outputs + (acc,)

        # prediction utils
        if not self.training:
            self.concat_predict_tensors(sentence_logits=_scores,
                                        sent_word_ids=input_ids.gather(dim=1, index=sentence_index[:, :, 0]).reshape(
                                            batch, sent_num, seq_len))

        return outputs


class BertForMultipleChoice(BertPreTrainedModel):
    model_prefix = 'bert_mcrc'
    config_class = IterBertPreTrainedConfig

    def __init__(self, config: IterBertPreTrainedConfig):
        super().__init__(config)

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        if config.cls_type == 1:
            self.pooler = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.Tanh()
            )
        else:
            self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
        self.classifier = nn.Linear(config.hidden_size, 1)

        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None, **kwargs):

        batch, num_choices = input_ids.size()[:2]

        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None

        seq_output = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )[0]

        logits = self.classifier(self.dropout(self.pooler(seq_output[:, 0])))
        logits = logits.view(-1, num_choices)

        outputs = (logits,)
        if labels is not None:
            loss = self.loss_fct(logits, labels)
            outputs = (loss,) + outputs

            _, pred = logits.max(dim=-1)
            acc = torch.sum(pred == labels) / (1.0 * batch)

            outputs = outputs + (acc,)

        return outputs


class BertForSequenceClassification(BertPreTrainedModel):
    model_prefix = 'bert_sc'
    config_class = IterBertPreTrainedConfig

    def __init__(self, config: IterBertPreTrainedConfig):
        super().__init__(config)

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        if config.cls_type == 1:
            self.pooler = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.Tanh()
            )
        else:
            self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None, **kwargs):

        batch, num_choices = input_ids.size()[:2]

        # input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        # attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        # token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None

        seq_output = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )[0]

        logits = self.classifier(self.dropout(self.pooler(seq_output[:, 0])))
        # logits = logits.view(-1, num_choices)

        outputs = (logits,)
        if labels is not None:
            loss = self.loss_fct(logits, labels)
            outputs = (loss,) + outputs

            _, pred = logits.max(dim=-1)
            acc = torch.sum(pred == labels) / (1.0 * batch)

            outputs = outputs + (acc,)

        return outputs


class IterBertForQuestionAnswering(IterBertModel):
    model_prefix = 'iter_bert_span'

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels * config.hidden_size)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            start_positions=None,
            end_positions=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch, seq_len = input_ids.size()
        # `token_type_ids`: [0,0,0,1,1,1,1,0,0,0]
        # `attention_mask`: [1,1,1,1,1,1,1,0,0,0]
        # `1` for true token and `0` for mask
        question_mask = (1 - token_type_ids) * attention_mask
        passage_mask = token_type_ids * attention_mask

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        d = sequence_output.size(-1)

        cls_h = sequence_output[:, :1]
        question_mask = question_mask.to(sequence_output.dtype)
        passage_mask = passage_mask.to(sequence_output.dtype)
        attention_mask = attention_mask.to(sequence_output.dtype)

        q_hidden = self.query(cls_h, sequence_output.unsqueeze(1), 1 - question_mask.unsqueeze(1),
                              aligned=True, residual=False).view(batch, d)

        bilinear_q = self.qa_outputs(q_hidden).view(batch, self.num_labels, d)
        # [batch, 2, d], [batch, seq_len, d] -> [batch, seq_len, 2]
        logits = torch.einsum("bih,bjh->bji", bilinear_q, sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


iter_bert_models_map = {
    BertForMaskedLMBaseline.model_prefix: BertForMaskedLMBaseline,

    IterBertModelForBiSR.model_prefix: IterBertModelForBiSR,
    IterBertModelForBiSRAndMLM.model_prefix: IterBertModelForBiSRAndMLM,
    IterBertModelForSRAndMLM.model_prefix: IterBertModelForSRAndMLM,
    IterBertModelForSR.model_prefix: IterBertModelForSR,
    IterBertModelForMLM.model_prefix: IterBertModelForMLM,

    IterBertModelForMCRC.model_prefix: IterBertModelForMCRC,
    IterBertModelForMCRCDropout.model_prefix: IterBertModelForMCRCDropout,
    IterBertModelForMCRC2.model_prefix: IterBertModelForMCRC2,
    IterBertModelForMCRC3.model_prefix: IterBertModelForMCRC3,
    IterBertModelForMCRC4.model_prefix: IterBertModelForMCRC4,

    IterBertModelForSequenceClassification.model_prefix: IterBertModelForSequenceClassification,
    IterBertModelForSequenceClassificationV2.model_prefix: IterBertModelForSequenceClassificationV2,
    IterBertModelForSequenceClassificationV3.model_prefix: IterBertModelForSequenceClassificationV3,
    IterBertModelForSequenceClassificationV4.model_prefix: IterBertModelForSequenceClassificationV4,

    BertForMultipleChoice.model_prefix: BertForMultipleChoice,
    BertForSequenceClassification.model_prefix: BertForSequenceClassification
}
