import torch
from torch import nn
from transformers.modeling_roberta import RobertaPreTrainedModel, RobertaConfig, RobertaModel, ACT2FN

from general_util.utils import LogMetric
from modules import layers


class CQueryPreTrainedConfig(RobertaConfig):
    added_configs = [
        'query_dropout', 'cls_type', 'sr_query_dropout', 'lm_query_dropout'
    ]

    def __init__(self, query_dropout=0.4, cls_type=0, 
                 sr_query_dropout=0.2, lm_query_dropout=0.1, **kwargs):
        super().__init__(**kwargs)

        self.query_dropout = query_dropout
        self.sr_query_dropout = sr_query_dropout
        self.lm_query_dropout = lm_query_dropout
        self.cls_type = cls_type

    def expand_configs(self, *args):
        self.added_configs.extend(list(args))


class ClsQueryDRobertaModel(RobertaPreTrainedModel):
    config_class = CQueryPreTrainedConfig
    model_prefix = 'cquery_d_roberta'

    def __init__(self, config: CQueryPreTrainedConfig):
        super().__init__(config)

        self.roberta = RobertaModel(config)

        self.c_query = layers.MultiHeadTokenAttention(
            config,
            attn_dropout_p=config.query_dropout,
            dropout_p=config.query_dropout
        )

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

        cls_h = seq_output[:, :1]
        sent_word_hidden = seq_output.gather(dim=1, index=sentence_index).reshape(
            batch, sent_num, seq_len, -1)

        hidden_sent = self.c_query(cls_h, sent_word_hidden, sent_word_mask).squeeze(1)

        return hidden_sent, seq_output, sent_word_hidden


class ClsQueryDRobertaModelForSentenceReorderingAndMLMDual(ClsQueryDRobertaModel):
    model_prefix = 'cquery_d_roberta_sr_mlm'

    def __init__(self, config: CQueryPreTrainedConfig):
        super().__init__(config)

        self.sr_sent_sum = nn.Linear(config.hidden_size, config.hidden_size)
        self.lm_sent_sum = nn.Linear(config.hidden_size, config.hidden_size)

        self.lm_sum_dropout = nn.Dropout(config.lm_query_dropout)
        self.sr_sum_dropout = nn.Dropout(config.sr_query_dropout)

        word_embedding_weight = self.roberta.get_input_embeddings().weight
        self.vocab_size = word_embedding_weight.size(0)

        self.sr_prediction_head = layers.SentenceReorderPredictionHeadReverse(config)
        self.lm_prediction_head = layers.MaskedLMPredictionHead(config, word_embedding_weight)

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
            sr_query_h, sent_word_hidden, sent_word_mask,
            _dropout=self.sr_sum_dropout
        )

        sr_scores = self.sr_prediction_head(sr_query_h, q_rel_d_sent_h)

        # MLM
        lm_query_h = self.lm_sent_sum(query_h)
        query_token_num = mlm_ids.size(1)
        
        q_rel_d_h, _ = layers.mul_weighted_sum(
            lm_query_h, seq_output, 1 - attention_mask,
            _dropout=self.lm_sum_dropout
        )
        
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


class ClsQueryDRobertaModelForMCRC(ClsQueryDRobertaModel):
    model_prefix = 'cquery_d_roberta_mcrc'
    
    def __init__(self, config: CQueryPreTrainedConfig):
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
        q_op_hidden_sent = self.c_query(cls_h, q_op_word_hidden, q_op_word_mask).view(fb, cls_h.size(-1))

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


cuqery_d_roberta_models_map = {
    ClsQueryDRobertaModelForSentenceReorderingAndMLMDual.model_prefix: ClsQueryDRobertaModelForSentenceReorderingAndMLMDual,

    ClsQueryDRobertaModelForMCRC.model_prefix: ClsQueryDRobertaModelForMCRC
}
