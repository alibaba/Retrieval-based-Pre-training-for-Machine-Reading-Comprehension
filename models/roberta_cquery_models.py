import copy

import torch
from torch import nn
import transformers
from transformers.modeling_roberta import RobertaPreTrainedModel, RobertaConfig, RobertaModel, RobertaLayer, \
    RobertaLMHead

from modules import layers
from modules.modeling_reusable_graph_attention import TransformerDecoder
from general_util.utils import LogMetric


class ClsQueryRobertaModel(RobertaPreTrainedModel):
    model_prefix = 'cquery_roberta'

    def __init__(self, config: RobertaConfig):
        super().__init__(config)

        self.roberta = RobertaModel(config)

        self.cls_w = nn.Linear(config.hidden_size, config.hidden_size)

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

        return hidden_sent, seq_output, sent_word_hidden


class ClsQueryRobertaModelForSentenceReordering(ClsQueryRobertaModel):
    model_prefix = 'cquery_roberta_sr'

    def __init__(self, config: RobertaConfig):
        super().__init__(config)

        self.sr_sent_sum = nn.Linear(config.hidden_size, config.hidden_size)

        self.sr_prediction_head = layers.SentenceReorderPredictionHead(config)

        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-1)

        self.init_weights()

        # metric
        self.eval_metrics = LogMetric("sr_acc", "sr_loss")

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

        # SR
        query_num = answers.size(1)
        query_h = hidden_sent[:, :query_num]
        query_h = self.sr_sent_sum(query_h)

        q_rel_d_sent_h, _ = layers.mul_sentence_sum(query_h, sent_word_hidden, sent_word_mask)
        q_rel_d_sent_h = q_rel_d_sent_h * (1 - sentence_mask[:, None, :, None])

        sr_scores = self.sr_prediction_head(query_h, q_rel_d_sent_h)

        output_dict = {}

        if answers is not None and pre_answers is not None:

            sent_mask = sentence_mask
            diag_mask = torch.eye(query_num, m=sent_num,
                                  device=sr_scores.device,
                                  dtype=sent_mask.dtype).view(1, query_num, sent_num)
            sent_mask = sent_mask.unsqueeze(1).expand(-1, query_num, -1)

            sr_scores = sr_scores + sent_mask * -10000.0

            fol_masked_scores = layers.mask_scores_with_labels(sr_scores, answers).contiguous()
            sr_loss1 = self.loss_fct(fol_masked_scores.view(batch * query_num, -1),
                                     pre_answers.view(-1))

            pre_masked_scores = layers.mask_scores_with_labels(sr_scores, pre_answers).contiguous()
            sr_loss2 = self.loss_fct(pre_masked_scores.view(batch * query_num, -1),
                                     answers.view(-1))

            loss = sr_loss1 + sr_loss2

            output_dict["loss"] = loss

            if not self.training:
                valid_num1 = (answers != -1).sum().item()
                valid_num2 = (pre_answers != -1).sum().item()
                valid_num = valid_num1 + valid_num2

                # pred = torch.argsort(sr_scores.detach(), dim=-1, descending=True)[:, :, :2]
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


class ClsQueryRobertaModelForSentenceReorderingAndCopyMLM(ClsQueryRobertaModel):
    model_prefix = 'cquery_roberta_sr_cmlm'

    def __init__(self, config: RobertaConfig):
        super().__init__(config)

        self.sr_sent_sum = nn.Linear(config.hidden_size, config.hidden_size)

        word_embedding_weight = self.roberta.get_input_embeddings().weight
        self.vocab_size = word_embedding_weight.size(0)

        self.copy_head = layers.MaskedLMCopyHead(config)

        self.sr_prediction_head = layers.SentenceReorderPredictionHead(config)

        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
        self.nll_loss_fct = nn.NLLLoss(ignore_index=-1)

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

        # Copy MLM
        aligned_sent_hidden = hidden_sent.gather(
            dim=1,
            index=reverse_sentence_index.unsqueeze(-1).expand(-1, -1, seq_output.size(-1))
        )
        concat_word_hidden = torch.cat([seq_output, aligned_sent_hidden], dim=-1)
        concat_word_hidden = concat_word_hidden * attention_mask[:, :, None]

        seq_scores = self.copy_head(concat_word_hidden, seq_output, 1 - attention_mask)
        # seq_prob = torch.softmax(seq_scores, dim=-1)

        # mlm_prob = seq_prob.new_zeros((batch, seq_length, self.vocab_size))
        # mlm_prob.scatter_add_(dim=2,
        #                       index=input_ids.unsqueeze(1).expand(-1, seq_length, -1),
        #                       src=seq_prob)
        # mlm_log_prob = mlm_prob.log()

        mlm_scores = seq_scores.new_zeros((batch, seq_length, self.vocab_size))
        mlm_scores.scatter_add_(dim=2,
                                index=input_ids.unsqueeze(1).expand(-1, seq_length, -1),
                                src=seq_scores)
        mlm_scores = mlm_scores.clamp_min(min=-10000.0)

        # SR
        query_num = answers.size(1)
        query_h = hidden_sent[:, :query_num]
        query_h = self.sr_sent_sum(query_h)

        q_rel_d_sent_h, _ = layers.mul_sentence_sum(query_h, sent_word_hidden, sent_word_mask)
        q_rel_d_sent_h = q_rel_d_sent_h * (1 - sentence_mask[:, None, :, None])

        sr_scores = self.sr_prediction_head(query_h, q_rel_d_sent_h)

        output_dict = {}

        if mlm_ids is not None and answers is not None and pre_answers is not None:

            sent_mask = sentence_mask
            diag_mask = torch.eye(query_num, m=sent_num,
                                  device=sr_scores.device,
                                  dtype=sent_mask.dtype).view(1, query_num, sent_num)
            sent_mask = sent_mask.unsqueeze(1).expand(-1, query_num, -1)

            sr_scores = sr_scores + sent_mask * -10000.0

            fol_masked_scores = layers.mask_scores_with_labels(sr_scores, answers).contiguous()
            sr_loss1 = self.loss_fct(fol_masked_scores.view(batch * query_num, -1),
                                     pre_answers.view(-1))

            pre_masked_scores = layers.mask_scores_with_labels(sr_scores, pre_answers).contiguous()
            sr_loss2 = self.loss_fct(pre_masked_scores.view(batch * query_num, -1),
                                     answers.view(-1))

            # mlm_loss = self.nll_loss_fct(mlm_log_prob.view(-1, self.config.vocab_size),
            #                              mlm_ids.view(-1))
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

                # pred = torch.argsort(sr_scores.detach(), dim=-1, descending=True)[:, :, :2]
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


class ClsQueryRobertaModelForSentenceReorderingAndMLM(ClsQueryRobertaModel):
    model_prefix = 'cquery_roberta_sr_mlm'

    def __init__(self, config: RobertaConfig):
        super().__init__(config)

        self.sr_sent_sum = nn.Linear(config.hidden_size, config.hidden_size)

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
        query_h = self.sr_sent_sum(query_h)

        attention_mask = attention_mask.to(query_h.dtype)

        # SR
        q_rel_d_sent_h, _ = layers.mul_sentence_sum(query_h, sent_word_hidden, sent_word_mask)
        q_rel_d_sent_h = q_rel_d_sent_h * (1 - sentence_mask[:, None, :, None])

        sr_scores = self.sr_prediction_head(query_h, q_rel_d_sent_h)

        # MLM
        query_token_num = mlm_ids.size(1)
        q_rel_d_h, _ = layers.mul_weighted_sum(query_h, seq_output, 1 - attention_mask)
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
            diag_mask = torch.eye(query_num, m=sent_num,
                                  device=sr_scores.device,
                                  dtype=sent_mask.dtype).view(1, query_num, sent_num)
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


class ClsQueryRobertaModelForSentenceReorderingAndMLMDual(ClsQueryRobertaModel):
    model_prefix = 'cquery_roberta_sr_mlm_d'

    def __init__(self, config: RobertaConfig):
        super().__init__(config)

        self.sr_sent_sum = nn.Linear(config.hidden_size, config.hidden_size)
        self.lm_sent_sum = nn.Linear(config.hidden_size, config.hidden_size)

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
        sr_query_h = self.sr_sent_sum(query_h)

        attention_mask = attention_mask.to(query_h.dtype)

        # SR
        q_rel_d_sent_h, _ = layers.mul_sentence_sum(sr_query_h, sent_word_hidden, sent_word_mask)
        q_rel_d_sent_h = q_rel_d_sent_h * (1 - sentence_mask[:, None, :, None])

        sr_scores = self.sr_prediction_head(sr_query_h, q_rel_d_sent_h)

        # MLM
        lm_query_h = self.lm_sent_sum(query_h)
        query_token_num = mlm_ids.size(1)
        q_rel_d_h, _ = layers.mul_weighted_sum(lm_query_h, seq_output, 1 - attention_mask)
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
            diag_mask = torch.eye(query_num, m=sent_num,
                                  device=sr_scores.device,
                                  dtype=sent_mask.dtype).view(1, query_num, sent_num)
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


class ClsQueryRobertaModelForSentenceReorderingSimple(ClsQueryRobertaModel):
    model_prefix = 'cquery_roberta_sr_s'

    def __init__(self, config: RobertaConfig):
        super().__init__(config)

        self.sr_sent_sum = nn.Linear(config.hidden_size, config.hidden_size)

        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-1)

        self.init_weights()

        # metric
        self.eval_metrics = LogMetric("sr_acc", "sr_loss")

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

        # SR
        query_num = answers.size(1)
        query_h = hidden_sent[:, :query_num]
        query_h = self.sr_sent_sum(query_h)

        q_rel_d_sent_h, _ = layers.mul_sentence_sum(query_h, sent_word_hidden, sent_word_mask)
        q_rel_d_sent_h = q_rel_d_sent_h * (1 - sentence_mask[:, None, :, None])

        # sr_scores = self.sr_prediction_head(query_h, q_rel_d_sent_h)
        sr_scores = torch.einsum("bqh,bqsh->bqs", query_h, q_rel_d_sent_h)    

        output_dict = {}

        if answers is not None and pre_answers is not None:

            sent_mask = sentence_mask
            # diag_mask = torch.eye(query_num, m=sent_num,
            #                       device=sr_scores.device,
            #                       dtype=sent_mask.dtype).view(1, query_num, sent_num)
            sent_mask = sent_mask.unsqueeze(1).expand(-1, query_num, -1)

            sr_scores = sr_scores + sent_mask * -10000.0

            fol_masked_scores = layers.mask_scores_with_labels(sr_scores, answers).contiguous()
            sr_loss1 = self.loss_fct(fol_masked_scores.view(batch * query_num, -1),
                                     pre_answers.view(-1))

            pre_masked_scores = layers.mask_scores_with_labels(sr_scores, pre_answers).contiguous()
            sr_loss2 = self.loss_fct(pre_masked_scores.view(batch * query_num, -1),
                                     answers.view(-1))

            loss = sr_loss1 + sr_loss2

            output_dict["loss"] = loss

            if not self.training:
                valid_num1 = (answers != -1).sum().item()
                valid_num2 = (pre_answers != -1).sum().item()
                valid_num = valid_num1 + valid_num2

                # pred = torch.argsort(sr_scores.detach(), dim=-1, descending=True)[:, :, :2]
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


class ClsQueryRobertaModelForSentenceReorderingProject(ClsQueryRobertaModel):
    model_prefix = 'cquery_roberta_sr_pro'

    def __init__(self, config: RobertaConfig):
        super().__init__(config)

        self.sr_sent_sum = nn.Linear(config.hidden_size, config.hidden_size)

        self.sr_pro_query = nn.Linear(config.hidden_size, config.hidden_size * 2)
        self.sr_pro_key = nn.Linear(config.hidden_size, config.hidden_size * 2)

        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-1)

        self.init_weights()

        # metric
        self.eval_metrics = LogMetric("sr_acc", "sr_loss")

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

        # SR
        query_num = answers.size(1)
        query_h = hidden_sent[:, :query_num]
        query_h = self.sr_sent_sum(query_h)

        q_rel_d_sent_h, _ = layers.mul_sentence_sum(query_h, sent_word_hidden, sent_word_mask)
        q_rel_d_sent_h = q_rel_d_sent_h * (1 - sentence_mask[:, None, :, None])

        h = query_h.size(-1)
        # sr_scores = self.sr_prediction_head(query_h, q_rel_d_sent_h)
        # sr_scores = torch.einsum("bqh,bqsh->bqs", query_h, q_rel_d_sent_h)    
        q = self.sr_pro_query(query_h).reshape(batch, query_num, 2, h)
        k = self.sr_pro_query(q_rel_d_sent_h).reshape(batch, query_num, sent_num, 2, h)
        sr_scores = torch.einsum("bqxh,bqsxh->bqxs", q, k)

        output_dict = {}

        if answers is not None and pre_answers is not None:

            sent_mask = sentence_mask
            # diag_mask = torch.eye(query_num, m=sent_num,
            #                       device=sr_scores.device,
            #                       dtype=sent_mask.dtype).view(1, query_num, sent_num)
            sent_mask = sent_mask.view(batch, 1, 1, sent_num).expand(-1, query_num, 2, -1)

            sr_scores = sr_scores + sent_mask * -10000.0

            sr_loss1 = self.loss_fct(sr_scores[:, :, 0].reshape(batch * query_num, -1),
                                     pre_answers.view(-1))
            sr_loss2 = self.loss_fct(sr_scores[:, :, 1].reshape(batch * query_num, -1),
                                     answers.view(-1))

            loss = sr_loss1 + sr_loss2

            output_dict["loss"] = loss

            if not self.training:
                valid_num1 = (answers != -1).sum().item()
                valid_num2 = (pre_answers != -1).sum().item()
                valid_num = valid_num1 + valid_num2

                # pred = torch.argsort(sr_scores.detach(), dim=-1, descending=True)[:, :, :2]
                # _, pred = torch.topk(sr_scores, k=2, dim=-1, largest=True)
                _, pred = sr_scores.max(dim=-1)

                acc1 = (pred[:, :, 0] == pre_answers).sum()
                acc2 = (pred[:, :, 1] == answers).sum()

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


class ClsQueryRobertaModelForSentenceReorderingUniProject(ClsQueryRobertaModel):
    model_prefix = 'cquery_roberta_sr_uni_pro'

    def __init__(self, config: RobertaConfig):
        super().__init__(config)

        self.sr_sent_sum = nn.Linear(config.hidden_size, config.hidden_size)

        self.sr_pro_query = nn.Linear(config.hidden_size, config.hidden_size)
        self.sr_pro_key = nn.Linear(config.hidden_size, config.hidden_size)

        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-1)

        self.init_weights()

        # metric
        self.eval_metrics = LogMetric("sr_acc", "sr_loss")

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

        # SR
        query_num = answers.size(1)
        query_h = hidden_sent[:, :query_num]
        query_h = self.sr_sent_sum(query_h)

        q_rel_d_sent_h, _ = layers.mul_sentence_sum(query_h, sent_word_hidden, sent_word_mask)
        q_rel_d_sent_h = q_rel_d_sent_h * (1 - sentence_mask[:, None, :, None])

        h = query_h.size(-1) 
        q = self.sr_pro_query(query_h)
        k = self.sr_pro_query(q_rel_d_sent_h)
        sr_scores = torch.einsum("bqh,bqsh->bqs", q, k)

        output_dict = {}

        if answers is not None:

            sent_mask = sentence_mask
            # diag_mask = torch.eye(query_num, m=sent_num,
            #                       device=sr_scores.device,
            #                       dtype=sent_mask.dtype).view(1, query_num, sent_num)
            sent_mask = sent_mask.view(batch, 1, sent_num).expand(-1, query_num, -1)

            sr_scores = sr_scores + sent_mask * -10000.0

            sr_loss = self.loss_fct(sr_scores.reshape(batch * query_num, -1),
                                    answers.view(-1))

            loss = sr_loss

            output_dict["loss"] = loss

            if not self.training:
                valid_num = (answers != -1).sum().item()

                _, pred = sr_scores.max(dim=-1)

                acc = (pred == answers).sum().to(sr_scores.dtype) / (valid_num * 1.0)

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


cquery_roberta_models_map = {
    ClsQueryRobertaModelForSentenceReordering.model_prefix: ClsQueryRobertaModelForSentenceReordering,
    ClsQueryRobertaModelForSentenceReorderingSimple.model_prefix: ClsQueryRobertaModelForSentenceReorderingSimple,
    ClsQueryRobertaModelForSentenceReorderingAndMLM.model_prefix: ClsQueryRobertaModelForSentenceReorderingAndMLM,
    ClsQueryRobertaModelForSentenceReorderingAndMLMDual.model_prefix: ClsQueryRobertaModelForSentenceReorderingAndMLMDual,

    ClsQueryRobertaModelForSentenceReorderingAndCopyMLM.model_prefix: ClsQueryRobertaModelForSentenceReorderingAndCopyMLM,
    ClsQueryRobertaModelForSentenceReorderingProject.model_prefix: ClsQueryRobertaModelForSentenceReorderingProject,
    ClsQueryRobertaModelForSentenceReorderingUniProject.model_prefix: ClsQueryRobertaModelForSentenceReorderingUniProject
}
