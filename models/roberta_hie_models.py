import copy

import torch
from torch import nn
import transformers
from transformers.modeling_roberta import RobertaPreTrainedModel, RobertaConfig, RobertaModel, RobertaLayer, \
    RobertaLMHead

from modules import layers
from modules.modeling_reusable_graph_attention import TransformerDecoder
from general_util.utils import LogMetric


class HierarchicalRobertaPreTrainedModel(RobertaPreTrainedModel):
    base_model_prefix = 'hie_roberta'


class HierarchicalRobertaModel(HierarchicalRobertaPreTrainedModel):
    model_prefix = 'hie_roberta'

    def __init__(self, config: RobertaConfig):
        super().__init__(config)

        self.roberta = RobertaModel(config)

        self.sent_sum = layers.MHAToken(config, bias=True)
        
        self.sent_transformer = RobertaLayer(config)

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                sentence_index=None, sentence_mask=None, sent_word_mask=None,
                roberta_only=False, **kwargs):
        seq_output = self.roberta(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  token_type_ids=token_type_ids)[0]

        sentence_index, mask, sent_mask = sentence_index, sent_word_mask, sentence_mask
        batch, sent_num, seq_len = mask.size()
        sentence_index = sentence_index.unsqueeze(-1).expand(
            -1, -1, -1, self.config.hidden_size
        ).reshape(batch, sent_num * seq_len, self.config.hidden_size)

        cls_h = seq_output[:, 0]
        sent_word_hidden = seq_output.gather(dim=1, index=sentence_index).reshape(
            batch, sent_num, seq_len, -1)
        sent_word_hidden = sent_word_hidden * (1 - mask.unsqueeze(-1))

        if roberta_only:
            return cls_h, sent_word_hidden

        hidden_sent = self.sent_sum(cls_h.unsqueeze(1), sent_word_hidden, mask, residual=False)
        hidden_sent = hidden_sent.squeeze(1) * (1 - sent_mask.unsqueeze(-1))

        sent_mask = sent_mask.view(batch, 1, 1, sent_num) * -10000.0
        
        sent_outputs = self.sent_transformer(
            hidden_states=hidden_sent,
            attention_mask=sent_mask,
        )

        hidden_sent = sent_outputs[0]

        return hidden_sent, seq_output, sent_word_hidden


class HierarchicalRobertaModelForPreTraining(HierarchicalRobertaPreTrainedModel):
    model_prefix = 'hie_roberta_sr_mlm'

    authorized_missing_keys = [r"position_ids", r"predictions.decoder.bias"]
    authorized_unexpected_keys = [r"pooler"]

    def __init__(self, config: RobertaConfig):
        super().__init__(config)

        self.hie_roberta = HierarchicalRobertaModel(config)

        self.sr_prediction_head = layers.SentenceReorderPredictionBidirectionalHead(config)
        self.lm_head = RobertaLMHead(config)

        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-1)

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                sentence_index=None, sentence_mask=None, sent_word_mask=None,
                answers=None, pre_answers=None, mlm_ids=None, **kwargs):
        
        hidden_sent, seq_output = self.hie_roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            sentence_index=sentence_index,
            sentence_mask=sentence_mask,
            sent_word_mask=sent_word_mask
        )

        batch, query_num = answers.size()

        pre_scores, fol_scores = self.sr_prediction_head(hidden_sent[:, :query_num], hidden_sent)
        
        sent_mask = sentence_mask
        sent_num = sent_mask.size(1)

        mlm_scores = self.lm_head(seq_output)

        output_dict = {
            "logits": torch.stack([pre_scores, fol_scores], dim=-1),
            "mlm_scores": mlm_scores
        }
        if answers is not None and pre_answers is not None and mlm_ids is not None:
            diag_mask = torch.eye(query_num, m=sent_num,
                                  device=pre_scores.device,
                                  dtype=sent_mask.dtype).view(1, query_num, sent_num)
            sent_mask = sent_mask.unsqueeze(1).expand(-1, query_num, -1)
            sent_mask = torch.clamp_max(diag_mask + sent_mask, max=1.0) * -10000.0

            pre_scores = pre_scores + sent_mask
            fol_scores = fol_scores + sent_mask

            loss1 = self.loss_fct(pre_scores.view(batch * query_num, -1),
                                  pre_answers.view(-1))
            loss2 = self.loss_fct(fol_scores.view(batch * query_num, -1),
                                  answers.view(-1))

            mlm_loss = self.loss_fct(mlm_scores.view(-1, self.config.vocab_size), mlm_ids.view(-1))

            loss = (loss1 + loss2 + mlm_loss)

            output_dict["loss"] = loss

            _, pred1 = fol_scores.max(dim=-1)
            acc1 = (pred1 == answers).sum().to(loss.dtype)
            valid_num1 = (answers != -1).sum().item()

            _, pred2 = pre_scores.max(dim=-1)
            acc2 = (pred2 == pre_answers).sum().to(loss.dtype)
            valid_num2 = (pre_answers != -1).sum().item()

            valid_num = valid_num1 + valid_num2
            acc = (acc1 + acc2) / (valid_num * 1.0)

            output_dict["acc"] = acc
            output_dict["valid_num"] = valid_num

            mlm_acc = (mlm_scores == mlm_ids).sum().to(loss.dtype).item()
            mlm_valid_num = (mlm_ids != -1).sum().item()
            output_dict["mlm_acc"] = mlm_acc / (mlm_valid_num * 1.0)
            output_dict["mlm_valid_num"] = mlm_valid_num
            output_dict["mlm_loss"] = mlm_loss.item()

        return output_dict


class HierarchicalRobertaModelForSentenceReorder(HierarchicalRobertaPreTrainedModel):

    model_prefix = 'hie_roberta_sr'

    def __init__(self, config: RobertaConfig):
        super().__init__(config)

        self.hie_roberta = HierarchicalRobertaModel(config)

        self.sr_prediction_head = layers.SentenceReorderPredictionBidirectionalHead(config)

        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-1)

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                sentence_index=None, sentence_mask=None, sent_word_mask=None,
                answers=None, pre_answers=None, **kwargs):
        
        hidden_sent = self.hie_roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            sentence_index=sentence_index,
            sentence_mask=sentence_mask,
            sent_word_mask=sent_word_mask
        )

        batch, query_num = answers.size()

        pre_scores, fol_scores = self.sr_prediction_head(hidden_sent[:, :query_num], hidden_sent)
        
        sent_mask = sentence_mask
        sent_num = sent_mask.size(1)

        output_dict = {
            "logits": torch.stack([pre_scores, fol_scores], dim=-1)
        }
        if answers is not None and pre_answers is not None:
            diag_mask = torch.eye(query_num, m=sent_num,
                                  device=pre_scores.device,
                                  dtype=sent_mask.dtype).view(1, query_num, sent_num)
            sent_mask = sent_mask.unsqueeze(1).expand(-1, query_num, -1)
            sent_mask = torch.clamp_max(diag_mask + sent_mask, max=1.0) * -10000.0

            pre_scores = pre_scores + sent_mask
            fol_scores = fol_scores + sent_mask

            loss1 = self.loss_fct(pre_scores.view(batch * query_num, -1),
                                  pre_answers.view(-1))
            loss2 = self.loss_fct(fol_scores.view(batch * query_num, -1),
                                  answers.view(-1))

            loss = (loss1 + loss2)

            output_dict["loss"] = loss

            _, pred1 = fol_scores.max(dim=-1)
            acc1 = (pred1 == answers).sum().to(loss.dtype)
            valid_num1 = (answers != -1).sum().item()

            _, pred2 = pre_scores.max(dim=-1)
            acc2 = (pred2 == pre_answers).sum().to(loss.dtype)
            valid_num2 = (pre_answers != -1).sum().item()

            valid_num = valid_num1 + valid_num2
            acc = (acc1 + acc2) / (valid_num * 1.0)

            output_dict["acc"] = acc
            output_dict["valid_num"] = valid_num

        return output_dict


class RobertaSentDecoderConfig(RobertaConfig):
    added_configs = [
        'max_targets', 'pos_emb_size', 'decoder_inter_size'
    ]

    def __init__(self, max_targets=80, pos_emb_size=200,
                 decoder_inter_size=1536, **kwargs):
        super().__init__(**kwargs)

        self.max_targets = max_targets
        self.pos_emb_size = pos_emb_size
        self.decoder_inter_size = decoder_inter_size

    def expand_configs(self, *args):
        self.added_configs.extend(list(args))


class SentenceReorderDecoder(nn.Module):

    def __init__(self, config: RobertaSentDecoderConfig):
        super().__init__()

        self.position_embeddings = nn.Embedding(
            config.max_targets, config.pos_emb_size)
        self.pos_emb_proj = nn.Linear(config.pos_emb_size, config.hidden_size)

        decoder_config = copy.deepcopy(config)
        decoder_config.intermediate_size = decoder_config.decoder_inter_size

        self.sent_decoder = TransformerDecoder(decoder_config)

        self.project = nn.Linear(config.hidden_size, config.hidden_size)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def get_decoder_mask(self, mask):
        batch_size, seq_length = mask.size()

        seq_ids = torch.arange(seq_length, device=mask.device)
        causal_mask = seq_ids[None, None, :].repeat(
            batch_size, seq_length, 1) <= seq_ids[None, :, None]
        # causal and attention masks must have same type with pytorch version < 1.3
        causal_mask = causal_mask.to(mask.dtype)

        extended_mask = mask[:, None, None, :]  # `1` for masked value
        decoder_mask = (1 - extended_mask) * causal_mask[:, None, :, :]
        decoder_mask = (1 - decoder_mask) * -10000.0
        return decoder_mask

    def forward(self, st_hidden, hidden, mask, true_index):
        batch_size, seq_len = mask.size()

        index_mask = (true_index == -1)
        masked_index = true_index.masked_fill(index_mask, 0)
        decoder_input_hidden = hidden.gather(
            index=masked_index.unsqueeze(-1).expand(-1, -1, hidden.size(-1)),
            dim=1
        )
        # Input shift
        decoder_input_hidden = torch.cat([
            st_hidden.unsqueeze(1), decoder_input_hidden
        ], dim=1)
        # We currently drop the last input to avoid the <eos> mark
        # since the decoder will not be used for downstream generation
        decoder_input_hidden = decoder_input_hidden[:, :-1]
        # So as the self-attention mask
        shift_mask = torch.cat([
            mask.new_zeros(batch_size, 1), mask
        ], dim=-1)
        shift_mask = shift_mask[:, :-1]

        decoder_mask = self.get_decoder_mask(shift_mask)
        encoder_mask = mask[:, None, None, :] * -10000.0

        seq_ids = torch.arange(
            seq_len, device=st_hidden.device, dtype=torch.long)
        pos_emb = self.pos_emb_proj(self.position_embeddings(seq_ids)).unsqueeze(0).expand(
            batch_size, -1, -1)
        decoder_input_hidden = self.dropout(decoder_input_hidden + pos_emb)

        outputs = self.sent_decoder(decoder_input_hidden,
                                    attention_mask=decoder_mask,
                                    encoder_hidden_states=hidden,
                                    encoder_attention_mask=encoder_mask)

        decoder_output = outputs[0]
        decoder_output = self.project(decoder_output)
        return torch.einsum("bih,bjh->bij", decoder_output, hidden)


class HierarchicalRobertaModelForRLM(HierarchicalRobertaPreTrainedModel):
    model_prefix = 'hie_roberta_rlm'
    config_class = RobertaSentDecoderConfig

    authorized_missing_keys = [r"position_ids", r"mlm_prediction_head.bias"]

    def __init__(self, config: RobertaConfig):
        super().__init__(config)

        self.hie_roberta = HierarchicalRobertaModel(config)

        word_embedding_weight = self.hie_roberta.roberta.get_input_embeddings().weight

        self.sr_prediction_head = SentenceReorderDecoder(config)
        self.mlm_prediction_head = layers.MaskedLMPredictionHead(config, word_embedding_weight)

        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-1)

        self.init_weights()

        # metric
        self.eval_metrics = LogMetric("mlm_loss", "mlm_acc", "sr_loss", "sr_acc")

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                sentence_index=None, sentence_mask=None, sent_word_mask=None,
                mlm_ids=None, true_sent_ids=None, reverse_sentence_index=None, **kwargs):
        
        hidden_sent, seq_output, sent_word_hidden = self.hie_roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            sentence_index=sentence_index,
            sentence_mask=sentence_mask,
            sent_word_mask=sent_word_mask
        )
        batch, sent_num, word_num = sent_word_mask.size()

        cls_h = seq_output[:, 0]
        sr_decoder_scores = self.sr_prediction_head(cls_h, hidden_sent,
                                                    sentence_mask, true_sent_ids)

        aligned_sent_hidden = hidden_sent.gather(
            dim=1,
            index=reverse_sentence_index.unsqueeze(-1).expand(-1, -1, cls_h.size(-1))
        )
        aligned_sent_hidden = aligned_sent_hidden * attention_mask[:, :, None]
        concat_word_hidden = torch.cat([
            seq_output,
            aligned_sent_hidden
        ], dim=-1)

        mlm_scores = self.mlm_prediction_head(concat_word_hidden)

        output_dict = {
            "logits": mlm_scores
        }
        if true_sent_ids is not None and mlm_ids is not None:

            nsr_loss = self.loss_fct(sr_decoder_scores.reshape(batch * sent_num, sent_num),
                                     true_sent_ids.reshape(-1))

            mlm_loss = self.loss_fct(mlm_scores.view(-1, self.config.vocab_size),
                                     mlm_ids.view(-1))

            loss = nsr_loss + mlm_loss

            output_dict["loss"] = loss

            _, mlm_pred = mlm_scores.max(dim=-1)
            mlm_acc = (mlm_pred == mlm_ids).sum().to(loss.dtype)
            mlm_valid_num = (mlm_ids != -1).sum().item()

            _, sr_pred = sr_decoder_scores.max(dim=-1)
            sr_acc = (sr_pred == true_sent_ids).sum().to(loss.dtype)
            sr_valid_num = (true_sent_ids != -1).sum().item()

            output_dict["acc"] = mlm_acc / mlm_valid_num
            output_dict["valid_num"] = mlm_valid_num

            self.eval_metrics.update("mlm_loss", mlm_loss.item(), mlm_valid_num)
            self.eval_metrics.update("mlm_acc", mlm_acc.item() / mlm_valid_num, mlm_valid_num)
            self.eval_metrics.update("sr_loss", nsr_loss.item(), sr_valid_num)
            self.eval_metrics.update("sr_acc", sr_acc.item() / sr_valid_num, sr_valid_num)

        return output_dict

    def get_eval_log(self, reset=False):
        _eval_metric_log = self.eval_metrics.get_log()
        _eval_metric_log = '\t'.join([f"{k}: {v}" for k, v in _eval_metric_log.items()])
        
        if reset:
            self.eval_metrics.reset()

        return _eval_metric_log



hierarchical_roberta_models_map = {
    HierarchicalRobertaModel.model_prefix: HierarchicalRobertaModel,

    HierarchicalRobertaModelForSentenceReorder.model_prefix: HierarchicalRobertaModelForSentenceReorder,

    HierarchicalRobertaModelForRLM.model_prefix: HierarchicalRobertaModelForRLM
}
