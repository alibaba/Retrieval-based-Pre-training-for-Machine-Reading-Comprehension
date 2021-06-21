import torch
from torch import nn
from transformers.modeling_roberta import RobertaPreTrainedModel, RobertaConfig, RobertaModel, ACT2FN

from general_util.utils import LogMetric
from modules import layers


class CQueryPreTrainedConfig(RobertaConfig):
    added_configs = [
        'query_dropout', 'query_ff_size', 'cls_type'
    ]

    def __init__(self, query_dropout=0.2, query_ff_size=1536, cls_type=0, **kwargs):
        super().__init__(**kwargs)

        self.query_dropout = query_dropout
        self.query_ff_size = query_ff_size
        self.cls_type = cls_type

    def expand_configs(self, *args):
        self.added_configs.extend(list(args))


class ClsQueryFFRobertaModel(RobertaPreTrainedModel):
    config_class = CQueryPreTrainedConfig
    model_prefix = 'cquery_ff_roberta'

    def __init__(self, config: CQueryPreTrainedConfig):
        super().__init__(config)

        self.roberta = RobertaModel(config)

        self.cls_w = nn.Linear(config.hidden_size, config.hidden_size)

        self.ffn = nn.Linear(config.hidden_size, config.intermediate_size // 2)
        self.ffn_out = nn.Linear(config.intermediate_size // 2, config.hidden_size)
        self.ffn_act_fn = ACT2FN[config.hidden_act]
        self.ffn_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.query_dropout = nn.Dropout(p=config.query_dropout)

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

        cls_h = seq_output[:, 0]
        sent_word_hidden = seq_output.gather(dim=1, index=sentence_index).reshape(
            batch, sent_num, seq_len, -1)
        sent_word_hidden = sent_word_hidden * (1 - sent_word_mask.unsqueeze(-1))

        cls_query = self.cls_w(cls_h)
        hidden_sent, _ = layers.sentence_sum(cls_query, sent_word_hidden, sent_word_mask)
        hidden_sent = hidden_sent.squeeze(1) * (1 - sentence_mask.unsqueeze(-1))

        hidden_sent = self.query_dropout(hidden_sent)

        ffn_output = self.ffn_out(self.ffn_act_fn(self.ffn(hidden_sent)))
        ffn_output = self.ffn_layer_norm(self.query_dropout(ffn_output))

        return hidden_sent, seq_output, sent_word_hidden


class ClsQueryFFRobertaModelForSentenceReorderingAndMLMDual(ClsQueryFFRobertaModel):
    model_prefix = 'cquery_ff_roberta_sr_mlm_d'

    def __init__(self, config: CQueryPreTrainedConfig):
        super().__init__(config)

        self.sr_sent_sum = nn.Linear(config.hidden_size, config.hidden_size)
        self.lm_sent_sum = nn.Linear(config.hidden_size, config.hidden_size)
        self.sum_dropout = nn.Dropout(config.attention_probs_dropout_prob)

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
            _dropout=self.sum_dropout
        )
        
        q_rel_d_sent_h = q_rel_d_sent_h * (1 - sentence_mask[:, None, :, None])

        sr_scores = self.sr_prediction_head(sr_query_h, q_rel_d_sent_h)

        # MLM
        lm_query_h = self.lm_sent_sum(query_h)
        query_token_num = mlm_ids.size(1)
        
        q_rel_d_h, _ = layers.mul_weighted_sum(
            lm_query_h, seq_output, 1 - attention_mask,
            _dropout=self.sum_dropout
        )
        
        aligned_sent_hidden = q_rel_d_h.gather(
            dim=1,
            index=reverse_sentence_index.unsqueeze(-1).expand(-1, -1, seq_output.size(-1))
        )
        
        concat_word_hidden = torch.cat([seq_output[:, :query_token_num], aligned_sent_hidden], dim=-1)
        concat_word_hidden = concat_word_hidden * attention_mask[:, :query_token_num].unsqueeze(-1)

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


cuqery_ff_roberta_models_map = {
    ClsQueryFFRobertaModelForSentenceReorderingAndMLMDual.model_prefix: ClsQueryFFRobertaModelForSentenceReorderingAndMLMDual
}
