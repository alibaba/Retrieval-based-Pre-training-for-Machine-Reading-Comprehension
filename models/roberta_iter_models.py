import torch
from torch import nn
from copy import deepcopy
from transformers.modeling_roberta import RobertaPreTrainedModel, RobertaConfig, RobertaModel

from general_util.logger import get_child_logger
from general_util.mixin import LogMixin, PredictionMixin
from modules import layers

logger = get_child_logger(__name__)


class IterRobertaPreTrainedConfig(RobertaConfig):
    added_configs = [
        'query_dropout', 'cls_type', 'sr_query_dropout', 'lm_query_dropout',
        'z_step', 'pos_emb_size', 'weight_typing', 'share_ssp_sum'
    ]

    def __init__(self, query_dropout=0.1, cls_type=0,
                 sr_query_dropout=0.1, lm_query_dropout=0.1,
                 pos_emb_size=200, z_step=0, weight_typing=True, 
                 share_ssp_sum=False, **kwargs):
        super().__init__(**kwargs)

        self.query_dropout = query_dropout
        self.cls_type = cls_type
        self.sr_query_dropout = sr_query_dropout
        self.lm_query_dropout = lm_query_dropout
        self.pos_emb_size = pos_emb_size
        self.z_step = z_step
        self.weight_typing = weight_typing
        self.share_ssp_sum = share_ssp_sum

    def expand_configs(self, *args):
        self.added_configs.extend(list(args))


class IterRobertaModel(RobertaPreTrainedModel):
    config_class = IterRobertaPreTrainedConfig
    model_prefix = 'iter_roberta'

    def __init__(self, config: IterRobertaPreTrainedConfig):
        super().__init__(config)

        self.config = config
        self.roberta = RobertaModel(config)

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
        seq_output = self.roberta(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  token_type_ids=token_type_ids)[0]

        batch, sent_num, seq_len = sent_word_mask.size()
        sentence_index = sentence_index.unsqueeze(-1).expand(
            -1, -1, -1, self.config.hidden_size
        ).reshape(batch, sent_num * seq_len, self.config.hidden_size)

        sent_word_hidden = seq_output.gather(dim=1, index=sentence_index).reshape(
            batch, sent_num, seq_len, -1)

        q_vec = seq_output[:, :1]  # <s>
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


class IterRobertaModelForSRAndMLM(IterRobertaModel, LogMixin):
    model_prefix = 'iter_roberta_sr_mlm'

    def __init__(self, config: IterRobertaPreTrainedConfig):
        super().__init__(config)

        self.sr_sent_sum_q = nn.Linear(config.hidden_size, config.hidden_size)
        self.sr_sent_sum_k = nn.Linear(config.hidden_size, config.hidden_size)
        self.sr_sent_sum_v = nn.Linear(config.hidden_size, config.hidden_size)
        self.lm_sent_sum_q = nn.Linear(config.hidden_size, config.hidden_size)
        self.lm_sent_sum_k = nn.Linear(config.hidden_size, config.hidden_size)
        self.lm_sent_sum_v = nn.Linear(config.hidden_size, config.hidden_size)

        self.lm_sum_dropout = nn.Dropout(config.lm_query_dropout)
        self.sr_sum_dropout = nn.Dropout(config.sr_query_dropout)

        self.sr_pooler = layers.Pooler(config.hidden_size)
        self.sr_prediction_head = nn.Linear(config.hidden_size, 1)

        if not self.config.weight_typing:
            word_embedding_weight = deepcopy(self.roberta.get_input_embeddings().weight)
        else:
            word_embedding_weight = self.roberta.get_input_embeddings().weight

        self.vocab_size = word_embedding_weight.size(0)

        self.lm_prediction_head = layers.MaskedLMPredictionHead(config, word_embedding_weight)

        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-1)

        self.init_weights()

        # metric
        self.init_metric("sr_acc", "sr_loss", "mlm_loss", "mlm_acc")

        logger.info(self.config.to_dict())

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                sentence_index=None, sentence_mask=None, sent_word_mask=None,
                mlm_ids: torch.Tensor = None, reverse_sentence_index: torch.Tensor = None,
                answers: torch.Tensor = None, pre_answers: torch.Tensor = None, **kwargs):

        hidden_sent, seq_output, sent_word_hidden = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            sentence_index=sentence_index,
            sentence_mask=sentence_mask,
            sent_word_mask=sent_word_mask
        )
        batch, sent_num, word_num = sent_word_mask.size()

        query_num = answers.size(1)
        query_h = hidden_sent[:, :query_num]

        attention_mask = attention_mask.to(query_h.dtype)

        # SR
        q_rel_d_sent_h = layers.multi_head_sent_sum(
            q=self.sr_sent_sum_q(query_h),
            k=self.sr_sent_sum_k(sent_word_hidden),
            v=self.sr_sent_sum_v(sent_word_hidden),
            mask=sent_word_mask,
            head_num=self.config.num_attention_heads,
            attn_dropout=self.sr_sum_dropout
        )
        sr_scores = self.sr_prediction_head(self.sr_sum_dropout(self.sr_pooler(q_rel_d_sent_h))).squeeze(-1)

        # MLM
        q_rel_d_h = layers.multi_head_sum(
            q=self.lm_sent_sum_q(query_h),
            k=self.lm_sent_sum_k(seq_output),
            v=self.lm_sent_sum_v(seq_output),
            mask=(1 - attention_mask),
            head_num=self.config.num_attention_heads,
            attn_dropout=self.lm_sum_dropout
        )
        query_token_num = mlm_ids.size(1)

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


class IterRobertaModelForSRAndMLMSimple(IterRobertaModel, LogMixin):
    model_prefix = 'iter_roberta_sr_mlm_s'

    def __init__(self, config: IterRobertaPreTrainedConfig):
        super().__init__(config)

        self.sr_sent_sum = nn.Linear(config.hidden_size, config.hidden_size)
        self.lm_sent_sum = nn.Linear(config.hidden_size, config.hidden_size)

        self.lm_dropout = nn.Dropout(config.lm_query_dropout)
        self.sr_dropout = nn.Dropout(config.sr_query_dropout)

        self.sr_pooler = layers.Pooler(config.hidden_size)
        self.sr_prediction_head = nn.Linear(config.hidden_size, 1)

        if not self.config.weight_typing:
            word_embedding_weight = deepcopy(self.roberta.get_input_embeddings().weight)
        else:
            word_embedding_weight = self.roberta.get_input_embeddings().weight

        self.vocab_size = word_embedding_weight.size(0)

        self.lm_prediction_head = layers.MaskedLMPredictionHead(config, word_embedding_weight)

        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-1)

        self.init_weights()

        # metric
        self.init_metric("sr_acc", "sr_loss", "mlm_loss", "mlm_acc")

        logger.info(self.config.to_dict())

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                sentence_index=None, sentence_mask=None, sent_word_mask=None,
                mlm_ids: torch.Tensor = None, reverse_sentence_index: torch.Tensor = None,
                answers: torch.Tensor = None, pre_answers: torch.Tensor = None, **kwargs):

        hidden_sent, seq_output, sent_word_hidden = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
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


class IterRobertaModelForSRAndMLMWithPosBias(IterRobertaModel, LogMixin):
    model_prefix = 'iter_roberta_sr_mlm_pb'

    def __init__(self, config: IterRobertaPreTrainedConfig):
        super().__init__(config)

        self.sr_sent_sum_q = nn.Linear(config.hidden_size, config.hidden_size)
        self.sr_sent_sum_k = nn.Linear(config.hidden_size, config.hidden_size)
        self.sr_sent_sum_v = nn.Linear(config.hidden_size, config.hidden_size)
        self.lm_sent_sum_q = nn.Linear(config.hidden_size, config.hidden_size)
        self.lm_sent_sum_k = nn.Linear(config.hidden_size, config.hidden_size)
        self.lm_sent_sum_v = nn.Linear(config.hidden_size, config.hidden_size)

        self.lm_sum_dropout = nn.Dropout(config.lm_query_dropout)
        self.sr_sum_dropout = nn.Dropout(config.sr_query_dropout)

        self.sr_pooler = layers.Pooler(config.hidden_size)
        self.sr_prediction_head = nn.Linear(config.hidden_size, 1)

        if not self.config.weight_typing:
            word_embedding_weight = deepcopy(self.roberta.get_input_embeddings().weight)
        else:
            word_embedding_weight = self.roberta.get_input_embeddings().weight

        self.vocab_size = word_embedding_weight.size(0)

        self.lm_prediction_head = layers.MultiHeadPositionBiasBasedForMLM(config,
                                                                          word_embedding_weight,
                                                                          config.pos_emb_size)

        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-1)

        self.init_weights()

        # metric
        self.init_metric("sr_acc", "sr_loss", "mlm_loss", "mlm_acc")

        logger.info(self.config.to_dict())

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                sentence_index=None, sentence_mask=None, sent_word_mask=None,
                mlm_ids: torch.Tensor = None, true_sent_ids: torch.Tensor = None, reverse_sentence_index=None,
                answers: torch.Tensor = None, pre_answers: torch.Tensor = None, **kwargs):

        hidden_sent, seq_output, sent_word_hidden = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            sentence_index=sentence_index,
            sentence_mask=sentence_mask,
            sent_word_mask=sent_word_mask
        )
        batch, sent_num, word_num = sent_word_mask.size()

        query_num = answers.size(1)
        query_h = hidden_sent[:, :query_num]

        attention_mask = attention_mask.to(query_h.dtype)

        # SR
        q_rel_d_sent_h = layers.multi_head_sent_sum(
            q=self.sr_sent_sum_q(query_h),
            k=self.sr_sent_sum_k(sent_word_hidden),
            v=self.sr_sent_sum_v(sent_word_hidden),
            mask=sent_word_mask,
            head_num=self.config.num_attention_heads,
            attn_dropout=self.sr_sum_dropout
        )
        sr_scores = self.sr_prediction_head(self.sr_sum_dropout(self.sr_pooler(q_rel_d_sent_h))).squeeze(-1)

        # MLM
        lm_q = self.lm_sent_sum_q(query_h)
        lm_k = self.lm_sent_sum_k(seq_output)
        lm_v = self.lm_sent_sum_v(seq_output)
        q_rel_mh_scores = layers.multi_head_sum(
            q=lm_q,
            k=lm_k,
            v=lm_v,
            mask=(1 - attention_mask),
            head_num=self.config.num_attention_heads,
            attn_dropout=self.lm_sum_dropout,
            return_scores=True
        )  # [batch, head_num, query_num, seq_len]
        query_token_num = mlm_ids.size(1)

        aligned_q_rel_scores = q_rel_mh_scores.gather(
            dim=2,
            index=reverse_sentence_index.view(batch, 1, query_token_num, 1).expand(
                -1, self.config.num_attention_heads, -1, q_rel_mh_scores.size(-1))
        )  # [batch, head_num, query_token_num, seq_len]

        mlm_scores = self.lm_prediction_head(aligned_q_rel_scores,
                                             seq_hidden_k=lm_k,
                                             seq_hidden_v=lm_v,
                                             seq_mask=(1 - attention_mask),
                                             dropout=self.lm_sum_dropout)

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


class IterRobertaModelForMCRC(IterRobertaModel):
    model_prefix = 'iter_roberta_mcrc'

    def __init__(self, config: IterRobertaPreTrainedConfig):
        super().__init__(config)

        self.sent_sum = nn.Linear(config.hidden_size, config.hidden_size)
        if config.share_ssp_sum:
            self.sr_sent_sum = nn.Linear(config.hidden_size, config.hidden_size)
            self.sent_sum = self.sr_sent_sum

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

        seq_output = self.roberta(input_ids=input_ids,
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


class IterRobertaModelForMCRC3(IterRobertaModel):
    model_prefix = 'iter_roberta_mcrc3'

    def __init__(self, config: IterRobertaPreTrainedConfig):
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

        seq_output = self.roberta(input_ids=input_ids,
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


class IterRobertaModelForSequenceClassificationV3(IterRobertaModel, PredictionMixin):
    model_prefix = 'iter_roberta_sc_v3'

    def __init__(self, config: IterRobertaPreTrainedConfig):
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

        seq_output = self.roberta(input_ids=input_ids,
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
            self.concat_predict_tensors(sentence_logits=_scores, sent_word_ids=input_ids.gather(dim=1, index=sentence_index[:, :, 0]).reshape(batch, sent_num, seq_len))

        return outputs



from transformers.modeling_roberta import RobertaForMultipleChoice


class RobertaForMultipleChoice(RobertaPreTrainedModel):
    model_prefix = 'roberta_mcrc'
    config_class = IterRobertaPreTrainedConfig

    def __init__(self, config: IterRobertaPreTrainedConfig):
        super().__init__(config)

        self.roberta = RobertaModel(config)
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

        seq_output = self.roberta(
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



class RobertaForSequenceClassification(RobertaPreTrainedModel):
    model_prefix = 'roberta_sc'
    config_class = IterRobertaPreTrainedConfig

    def __init__(self, config: IterRobertaPreTrainedConfig):
        super().__init__(config)

        self.roberta = RobertaModel(config)
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

        seq_output = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )[0]

        logits = self.classifier(self.dropout(self.pooler(seq_output[:, 0])))

        outputs = (logits,)
        if labels is not None:
            loss = self.loss_fct(logits, labels)
            outputs = (loss,) + outputs

            _, pred = logits.max(dim=-1)
            acc = torch.sum(pred == labels) / (1.0 * batch)

            outputs = outputs + (acc,)
        
        return outputs


iter_roberta_models_map = {
    IterRobertaModelForSRAndMLM.model_prefix: IterRobertaModelForSRAndMLM,
    IterRobertaModelForSRAndMLMWithPosBias.model_prefix: IterRobertaModelForSRAndMLMWithPosBias,
    IterRobertaModelForSRAndMLMSimple.model_prefix: IterRobertaModelForSRAndMLMSimple,

    IterRobertaModelForMCRC.model_prefix: IterRobertaModelForMCRC,
    IterRobertaModelForMCRC3.model_prefix: IterRobertaModelForMCRC3,

    IterRobertaModelForSequenceClassificationV3.model_prefix: IterRobertaModelForSequenceClassificationV3,

    RobertaForMultipleChoice.model_prefix: RobertaForMultipleChoice,
    RobertaForSequenceClassification.model_prefix: RobertaForSequenceClassification
}
