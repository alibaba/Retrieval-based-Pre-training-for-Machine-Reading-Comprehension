import torch
from torch import nn
from torch.nn import functional as F
from transformers import RobertaConfig, RobertaModel
from transformers.modeling_roberta import RobertaPreTrainedModel, RobertaLMHead

from modules.layers import simple_circle_loss, sentence_sum, mul_sentence_sum, MHAToken, fix_circle_loss, \
    cross_entropy_loss_and_accuracy, mask_scores_with_labels, fixed_circle_loss_clean, \
        SentenceReorderPredictionHead, SentenceReorderPredictionHeadDouble
from modules.modeling_reusable_graph_attention import SentenceReorderDecoder, SentenceSum, SentenceReorderHead, \
    SentenceReorderWithPosition


class RobertaForMaskedLM(RobertaPreTrainedModel):
    model_prefix = 'roberta_mlm'

    def __init__(self, config):
        super().__init__(config)

        self.roberta = RobertaModel(config)
        self.lm_head = RobertaLMHead(config)

        self.init_weights()

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        **kwargs
    ):
        if "masked_lm_labels" in kwargs:
            warnings.warn(
                "The `masked_lm_labels` argument is deprecated and will be removed in a future version, use `labels` instead.",
                DeprecationWarning,
            )
            labels = kwargs.pop("masked_lm_labels")
        # assert kwargs == {}, f"Unexpected keyword arguments: {list(kwargs.keys())}."

        if 'mlm_ids' in kwargs:
            labels = kwargs.pop("mlm_ids")


        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)

        # outputs = (prediction_scores,) + outputs[2:]  # Add hidden states and attention if they are here

        outputs = {
            "logits": prediction_scores,
            "others": outputs[2:]
        }

        if labels is not None:
            batch_size = labels.size(0)
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            # outputs = (masked_lm_loss,) + outputs
            outputs["loss"] = masked_lm_loss

            _, pred = prediction_scores.max(dim=-1)
            valid_num = torch.sum(labels != -1)
            acc = torch.sum(pred == labels).to(dtype=prediction_scores.dtype) / (valid_num * 1.0)
            outputs["acc"] = acc
            outputs["valid_num"] = valid_num.item()
            outputs["knowledge_loss"] = torch.Tensor([0])

        return outputs  # (masked_lm_loss), prediction_scores, (hidden_states), (attentions)


class RobertaModelForSentReorder(RobertaPreTrainedModel):
    model_prefix = 'roberta_sr'
    config_class = RobertaConfig
    base_model_prefix = 'roberta'

    def __init__(self, config: RobertaConfig):
        super().__init__(config)

        self.roberta = RobertaModel(config)

        self.sentecne_reorder_head = SentenceReorderHead(config.hidden_size)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                sentence_spans=None, sentence_index=None, sentence_mask=None,
                sent_word_mask=None, answers=None, edge_index=None, edges=None,
                **kwargs):
        seq_output = self.roberta(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  token_type_ids=token_type_ids)[0]

        sentence_index, mask, sent_mask = sentence_index, sent_word_mask, sentence_mask
        batch, sent_num, seq_len = mask.size()
        sentence_index = sentence_index.unsqueeze(-1).expand(
            -1, -1, -1, self.config.hidden_size
        ).reshape(batch, sent_num * seq_len, self.config.hidden_size)

        cls_h = seq_output[:, 0]
        hidden = seq_output.gather(dim=1, index=sentence_index).reshape(
            batch, sent_num, seq_len, -1)
        hidden = self.dropout(hidden)

        query_num = answers.size(1)
        scores, _ = self.sentecne_reorder_head(
            cls_h, hidden, mask, query_num=query_num)

        output_dict = {"logits": scores}
        if answers is not None:
            assert answers.size(1) == scores.size(1)
            scores = scores + sent_mask[:, None, :] * -10000.0
            loss = F.cross_entropy(scores.reshape(-1, sent_num), answers.reshape(-1),
                                   ignore_index=-1, reduction='sum') / (batch * 1.0)
            output_dict["loss"] = loss

            _, pred = scores.max(dim=-1)
            valid_num = torch.sum(answers != -1)
            acc = torch.sum(pred == answers).to(
                dtype=scores.dtype) / (valid_num * 1.0)
            output_dict["acc"] = acc
            output_dict["valid_num"] = valid_num.item()
            output_dict["knowledge_loss"] = torch.Tensor([0])

        return output_dict


class RobertaModelForSentReorderQ(RobertaPreTrainedModel):
    model_prefix = 'roberta_sr_q'
    config_class = RobertaConfig
    base_model_prefix = 'roberta'

    def __init__(self, config: RobertaConfig):
        super().__init__(config)

        self.roberta = RobertaModel(config)

        self.cls_w = nn.Linear(config.hidden_size, config.hidden_size)
        self.que_w = nn.Linear(config.hidden_size, config.hidden_size)

        self.project1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.project2 = nn.Linear(config.hidden_size, config.hidden_size)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                sentence_index=None, sentence_mask=None,
                sent_word_mask=None, answers=None, **kwargs):

        seq_output = self.roberta(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  token_type_ids=token_type_ids)[0]

        sentence_index, mask, sent_mask = sentence_index, sent_word_mask, sentence_mask
        batch, sent_num, seq_len = mask.size()
        sentence_index = sentence_index.unsqueeze(-1).expand(
            -1, -1, -1, self.config.hidden_size
        ).reshape(batch, sent_num * seq_len, self.config.hidden_size)

        cls_h = seq_output[:, 0]
        hidden = seq_output.gather(dim=1, index=sentence_index).reshape(
            batch, sent_num, seq_len, -1)
        # hidden = self.dropout(hidden)

        query_num = answers.size(1)

        query_q = self.cls_w(cls_h)
        query_h, _ = sentence_sum(query_q, hidden[:, :query_num], mask[:, :query_num])

        sent_q = self.que_w(query_h)
        sent_h, _ = mul_sentence_sum(sent_q, hidden, mask)

        sent_q = self.project1(sent_q)
        sent_h = self.project2(sent_h)

        scores = torch.einsum("bqh,bqsh->bqs", sent_q, sent_h)

        output_dict = {"logits": scores}
        if answers is not None:
            assert answers.size(1) == scores.size(1)
            scores = scores + sent_mask[:, None, :] * -10000.0
            loss = F.cross_entropy(scores.reshape(-1, sent_num), answers.reshape(-1),
                                   ignore_index=-1, reduction='sum') / (batch * 1.0)
            output_dict["loss"] = loss

            _, pred = scores.max(dim=-1)
            valid_num = torch.sum(answers != -1)
            acc = torch.sum(pred == answers).to(
                dtype=scores.dtype) / (valid_num * 1.0)
            output_dict["acc"] = acc
            output_dict["valid_num"] = valid_num.item()
            output_dict["knowledge_loss"] = torch.Tensor([0])

        return output_dict


class RobertaModelForSentReorderMHQ(RobertaPreTrainedModel):
    model_prefix = 'roberta_sr_mhq'
    config_class = RobertaConfig
    base_model_prefix = 'roberta'

    def __init__(self, config: RobertaConfig):
        super().__init__(config)

        self.roberta = RobertaModel(config)

        self.q_sum = MHAToken(config)
        self.s_sum = MHAToken(config)

        self.project1 = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.project2 = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                sentence_index=None, sentence_mask=None,
                sent_word_mask=None, answers=None, **kwargs):

        seq_output = self.roberta(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  token_type_ids=token_type_ids)[0]

        sentence_index, mask, sent_mask = sentence_index, sent_word_mask, sentence_mask
        batch, sent_num, seq_len = mask.size()
        sentence_index = sentence_index.unsqueeze(-1).expand(
            -1, -1, -1, self.config.hidden_size
        ).reshape(batch, sent_num * seq_len, self.config.hidden_size)

        cls_h = seq_output[:, 0]
        hidden = seq_output.gather(dim=1, index=sentence_index).reshape(
            batch, sent_num, seq_len, -1)
        hidden = hidden * (1 - mask.unsqueeze(-1))

        query_num = answers.size(1)

        query_h = self.q_sum(cls_h.unsqueeze(1), hidden[:, :query_num], mask[:, :query_num])
        query_h = query_h.squeeze(1)
        assert query_h.size(1) == query_num

        sent_h = self.s_sum(query_h, hidden, mask)

        query_h = self.project1(query_h)
        sent_h = self.project2(sent_h)

        scores = torch.einsum("bqh,bqsh->bqs", query_h, sent_h)

        output_dict = {"logits": scores}
        if answers is not None:
            assert answers.size(1) == scores.size(1)
            scores = scores + sent_mask[:, None, :] * -10000.0
            loss = F.cross_entropy(scores.reshape(-1, sent_num), answers.reshape(-1),
                                   ignore_index=-1, reduction='sum') / (batch * 1.0)
            output_dict["loss"] = loss

            _, pred = scores.max(dim=-1)
            valid_num = torch.sum(answers != -1)
            acc = torch.sum(pred == answers).to(
                dtype=scores.dtype) / (valid_num * 1.0)
            output_dict["acc"] = acc
            output_dict["valid_num"] = valid_num.item()
            output_dict["knowledge_loss"] = torch.Tensor([0])

        return output_dict


class RobertaModelForBiSentReorder(RobertaPreTrainedModel):
    model_prefix = 'roberta_bi_sr'
    config_class = RobertaConfig
    base_model_prefix = 'roberta'

    def __init__(self, config: RobertaConfig):
        super().__init__(config)

        self.roberta = RobertaModel(config)

        self.sentence_reorder_head = SentenceReorderHead(config.hidden_size)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.init_weights()

    @staticmethod
    def select_prob(_prob, _index):
        _index_mask = (_index == -1)
        _index = _index.masked_fill(_index_mask, 0)
        _selected_scores = _prob.gather(dim=-1, index=_index)
        _selected_scores[_index_mask] = 0.
        return _selected_scores, _index_mask

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                sentence_index=None, sentence_mask=None, sent_word_mask=None,
                answers=None, pre_answers=None, **kwargs):
        seq_output = self.roberta(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  token_type_ids=token_type_ids)[0]

        sentence_index, mask, sent_mask = sentence_index, sent_word_mask, sentence_mask
        batch, sent_num, seq_len = mask.size()
        sentence_index = sentence_index.unsqueeze(-1).expand(
            -1, -1, -1, self.config.hidden_size
        ).reshape(batch, sent_num * seq_len, self.config.hidden_size)

        cls_h = seq_output[:, 0]
        hidden = seq_output.gather(dim=1, index=sentence_index).reshape(
            batch, sent_num, seq_len, -1)
        hidden = self.dropout(hidden)

        query_num = answers.size(1)
        scores, _ = self.sentence_reorder_head(
            cls_h, hidden, mask, query_num=query_num)

        output_dict = {"logits": scores}

        if answers is not None and pre_answers is not None:
            prob = torch.softmax(scores + sent_mask[:, None, :] * -65500.0, dim=-1)
            answer_prob, answer_ignore_mask = self.select_prob(prob, answers.unsqueeze(-1))
            pre_answer_prob, pre_answer_ignore_mask = self.select_prob(prob, pre_answers.unsqueeze(-1))

            loss = -((answer_prob + pre_answer_prob + 1e-6).log())
            ignore_mask = (answer_ignore_mask * pre_answer_ignore_mask).bool()
            assert loss.size() == (batch, query_num, 1)
            loss[ignore_mask] = 0.
            # print(loss[0])
            # print(ignore_mask[0])
            loss = loss.sum() / (batch * 1.0)
            output_dict["loss"] = loss

            all_answers = torch.stack([answers, pre_answers], dim=-1)
            pred = torch.argsort(prob, dim=-1, descending=True)[:, :, :2]
            acc = (pred.unsqueeze(-1) == all_answers.unsqueeze(-2)).any(dim=-1).sum()
            valid_num = torch.sum(all_answers != -1).item()
            acc = acc.to(dtype=prob.dtype) / (valid_num * 1.0)
            output_dict["acc"] = acc
            output_dict["valid_num"] = valid_num

        return output_dict


class RobertaModelForBiSentReorderCL(RobertaPreTrainedModel):
    model_prefix = 'roberta_bi_sr'
    config_class = RobertaConfig
    base_model_prefix = 'roberta'

    def __init__(self, config: RobertaConfig):
        super().__init__(config)

        self.roberta = RobertaModel(config)

        self.sentence_reorder_head = SentenceReorderHead(config.hidden_size)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.softplus = nn.Softplus()

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                sentence_index=None, sentence_mask=None, sent_word_mask=None,
                answers=None, pre_answers=None, **kwargs):
        seq_output = self.roberta(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  token_type_ids=token_type_ids)[0]

        sentence_index, mask, sent_mask = sentence_index, sent_word_mask, sentence_mask
        batch, sent_num, seq_len = mask.size()
        sentence_index = sentence_index.unsqueeze(-1).expand(
            -1, -1, -1, self.config.hidden_size
        ).reshape(batch, sent_num * seq_len, self.config.hidden_size)

        cls_h = seq_output[:, 0]
        hidden = seq_output.gather(dim=1, index=sentence_index).reshape(
            batch, sent_num, seq_len, -1)
        hidden = self.dropout(hidden)

        query_num = answers.size(1)
        scores, _ = self.sentence_reorder_head(
            cls_h, hidden, mask, query_num=query_num)

        output_dict = {"logits": scores}

        if answers is not None and pre_answers is not None:
            sent_mask = sent_mask.unsqueeze(1).expand(-1, query_num, -1)

            all_answers = torch.stack([answers, pre_answers], dim=-1)
            
            loss = simple_circle_loss(output=scores.view(batch * query_num, sent_num),
                                      target=all_answers.view(batch * query_num, 2),
                                      mask=sent_mask.reshape(batch * query_num, sent_num),
                                      num_classes=sent_num,
                                      softplus=self.softplus) / (batch * 1.0)

            output_dict["loss"] = loss

            prob = scores + sent_mask * -10000.0
            pred = torch.argsort(prob, dim=-1, descending=True)[:, :, :2]  # (batch, query_num, 2)
            # acc = (pred.unsqueeze(-1) == all_answers.unsqueeze(-2)).any(dim=-1).sum(dim=-1)  # (batch, query_num, 2, 2)
            # assert acc.size() == (batch, query_num), acc.size()
            # acc = acc.sum()
            acc = (pred.unsqueeze(-1) == all_answers.unsqueeze(-2)).any(dim=-1).sum()
            valid_num = torch.sum(all_answers != -1).item()
            acc = acc.to(dtype=prob.dtype) / (valid_num * 1.0)
            output_dict["acc"] = acc
            output_dict["valid_num"] = valid_num

        return output_dict


class RobertaModelForBiSentReorderCLQ(RobertaPreTrainedModel):
    model_prefix = 'roberta_bi_sr_q'
    config_class = RobertaConfig
    base_model_prefix = 'roberta'

    def __init__(self, config: RobertaConfig):
        super().__init__(config)

        self.roberta = RobertaModel(config)

        self.cls_w = nn.Linear(config.hidden_size, config.hidden_size)
        self.que_w = nn.Linear(config.hidden_size, config.hidden_size)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.softplus = nn.Softplus()

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                sentence_index=None, sentence_mask=None, sent_word_mask=None,
                answers=None, pre_answers=None, **kwargs):
        seq_output = self.roberta(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  token_type_ids=token_type_ids)[0]

        sentence_index, mask, sent_mask = sentence_index, sent_word_mask, sentence_mask
        batch, sent_num, seq_len = mask.size()
        sentence_index = sentence_index.unsqueeze(-1).expand(
            -1, -1, -1, self.config.hidden_size
        ).reshape(batch, sent_num * seq_len, self.config.hidden_size)

        cls_h = seq_output[:, 0]
        hidden = seq_output.gather(dim=1, index=sentence_index).reshape(
            batch, sent_num, seq_len, -1)
        # hidden = self.dropout(hidden)

        query_num = answers.size(1)

        query_q = self.cls_w(cls_h)
        query_h, _ = sentence_sum(query_q, hidden[:, :query_num], mask[:, :query_num])

        sent_q = self.que_w(query_h)
        sent_h, _ = mul_sentence_sum(sent_q, hidden, mask)

        scores = torch.einsum("bqh,bqsh->bqs", sent_q, sent_h)

        output_dict = {"logits": scores}
        if answers is not None and pre_answers is not None:
            sent_mask = sent_mask.unsqueeze(1).expand(-1, query_num, -1)

            all_answers = torch.stack([answers, pre_answers], dim=-1)
            
            loss = simple_circle_loss(output=scores.view(batch * query_num, sent_num),
                                      target=all_answers.view(batch * query_num, 2),
                                      mask=sent_mask.reshape(batch * query_num, sent_num),
                                      num_classes=sent_num,
                                      softplus=self.softplus) / (batch * 1.0)

            output_dict["loss"] = loss

            prob = scores + sent_mask * -10000.0
            pred = torch.argsort(prob, dim=-1, descending=True)[:, :, :2]  # (batch, query_num, 2)
            # acc = (pred.unsqueeze(-1) == all_answers.unsqueeze(-2)).any(dim=-1).sum(dim=-1)  # (batch, query_num, 2, 2)
            # assert acc.size() == (batch, query_num), acc.size()
            # acc = acc.sum()
            acc = (pred.unsqueeze(-1) == all_answers.unsqueeze(-2)).any(dim=-1).sum()
            valid_num = torch.sum(all_answers != -1).item()
            acc = acc.to(dtype=prob.dtype) / (valid_num * 1.0)
            output_dict["acc"] = acc
            output_dict["valid_num"] = valid_num

        return output_dict


class RobertaModelForBiSentReorderCLMHQ(RobertaPreTrainedModel):
    model_prefix = 'roberta_bi_sr_mhq'
    config_class = RobertaConfig
    base_model_prefix = 'roberta'

    def __init__(self, config: RobertaConfig):
        super().__init__(config)

        self.roberta = RobertaModel(config)

        self.q_sum = MHAToken(config)
        self.s_sum = MHAToken(config)

        self.project1 = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.project2 = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.softplus = nn.Softplus()

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                sentence_index=None, sentence_mask=None, sent_word_mask=None,
                answers=None, pre_answers=None, **kwargs):
        seq_output = self.roberta(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  token_type_ids=token_type_ids)[0]

        sentence_index, mask, sent_mask = sentence_index, sent_word_mask, sentence_mask
        batch, sent_num, seq_len = mask.size()
        sentence_index = sentence_index.unsqueeze(-1).expand(
            -1, -1, -1, self.config.hidden_size
        ).reshape(batch, sent_num * seq_len, self.config.hidden_size)

        cls_h = seq_output[:, 0]
        hidden = seq_output.gather(dim=1, index=sentence_index).reshape(
            batch, sent_num, seq_len, -1)
        hidden = hidden * (1 - mask.unsqueeze(-1))

        query_num = answers.size(1)

        query_h = self.q_sum(cls_h.unsqueeze(1), hidden[:, :query_num], mask[:, :query_num])
        query_h = query_h.squeeze(1)

        sent_h = self.s_sum(query_h, hidden, mask)

        query_h = self.project1(query_h)
        sent_h = self.project2(sent_h)

        scores = torch.einsum("bqh,bqsh->bqs", query_h, sent_h)

        output_dict = {"logits": scores}
        if answers is not None and pre_answers is not None:
            diag_mask = torch.eye(query_num, m=sent_num,
                                  device=scores.device,
                                  dtype=sent_mask.dtype).view(1, query_num, sent_num)
            sent_mask = sent_mask.unsqueeze(1).expand(-1, query_num, -1)
            sent_mask = torch.clamp_max(diag_mask + sent_mask, max=1.0)

            all_answers = torch.stack([answers, pre_answers], dim=-1)
            
            # loss = simple_circle_loss(output=scores.view(batch * query_num, sent_num),
            #                           target=all_answers.view(batch * query_num, 2),
            #                           mask=sent_mask.reshape(batch * query_num, sent_num),
            #                           num_classes=sent_num,
            #                           softplus=self.softplus) / (batch * 1.0)
            # loss = fix_circle_loss(output=scores.view(batch * query_num, sent_num),
            #                        target=all_answers.view(batch * query_num, 2),
            #                        mask=sent_mask.reshape(batch * query_num, sent_num),
            #                        num_classes=sent_num,
            #                        softplus=self.softplus)
            loss = fixed_circle_loss_clean(output=scores.view(batch * query_num, sent_num),
                                           target=all_answers.view(batch * query_num, 2),
                                           mask=sent_mask.reshape(batch * query_num, sent_num),
                                           num_classes=sent_num,
                                           softplus=self.softplus)

            valid_num = torch.sum(all_answers != -1).item()
            loss = loss.sum() / valid_num
            # print(loss)

            output_dict["loss"] = loss

            prob = scores + sent_mask * -10000.0
            # (batch, query_num, 2)
            pred = torch.argsort(prob, dim=-1, descending=True)[:, :, :2]

            acc = (pred.unsqueeze(-1) == all_answers.unsqueeze(-2)).any(dim=-1).sum()
            acc = acc.to(dtype=prob.dtype) / (valid_num * 1.0)
            output_dict["acc"] = acc
            output_dict["valid_num"] = valid_num

        return output_dict


class RobertaModelForBiSentReorderQ(RobertaPreTrainedModel):
    model_prefix = 'roberta_bi_sr_cs_q'
    config_class = RobertaConfig
    base_model_prefix = 'roberta'

    def __init__(self, config: RobertaConfig):
        super().__init__(config)

        self.roberta = RobertaModel(config)

        self.cls_w = nn.Linear(config.hidden_size, config.hidden_size)
        
        self.proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.que_w = nn.Linear(config.hidden_size, config.hidden_size)

        self.half_dim = config.hidden_size // 2

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                sentence_index=None, sentence_mask=None, sent_word_mask=None,
                answers=None, pre_answers=None, **kwargs):
        seq_output = self.roberta(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  token_type_ids=token_type_ids)[0]

        sentence_index, mask, sent_mask = sentence_index, sent_word_mask, sentence_mask
        batch, sent_num, seq_len = mask.size()
        sentence_index = sentence_index.unsqueeze(-1).expand(
            -1, -1, -1, self.config.hidden_size
        ).reshape(batch, sent_num * seq_len, self.config.hidden_size)

        cls_h = seq_output[:, 0]
        hidden = seq_output.gather(dim=1, index=sentence_index).reshape(
            batch, sent_num, seq_len, -1)
       # hidden = self.dropout(hidden)

        query_num = answers.size(1)

        query_q = self.cls_w(cls_h)
        # (batch, query_num, h)
        query_h, _ = sentence_sum(query_q, hidden[:, :query_num], mask[:, :query_num])

        hidden_kv = self.proj(hidden).reshape(batch, sent_num, seq_len, 2, -1)

        sent_q = self.que_w(query_h).reshape(batch, query_num, 2, -1)

        # x: head_num = 2
        mh_scores = torch.einsum("bqxh,bstxh->bqstx", sent_q, hidden_kv)
        mh_alpha = (mh_scores + mask[:, None, :, :, None] * -10000.0).softmax(dim=3)
        mh_sent_h = torch.einsum("bqstx,bstxh->bqsxh", mh_alpha, hidden_kv)

        scores = torch.einsum("bqxh,bqsxh->bqsx", sent_q, mh_sent_h)

        output_dict = {"logits": scores}
        if answers is not None and pre_answers is not None:
            sent_mask = sent_mask[:, None, :, None].expand(-1, query_num, -1, 2)
            scores = scores + sent_mask * -10000.0

            loss1 = F.cross_entropy(input=scores[:, :, :, 0].reshape(batch * query_num, -1),
                                    target=answers.reshape(-1),
                                    ignore_index=-1, reduction='sum')
            loss2 = F.cross_entropy(input=scores[:, :, :, 1].reshape(batch * query_num, -1),
                                    target=pre_answers.reshape(-1),
                                    ignore_index=-1, reduction='sum')

            loss = (loss1 + loss2) / (batch * 1.0)

            output_dict["loss"] = loss

            _, pred1 = scores[:, :, :, 0].max(dim=-1)
            acc1 = (pred1 == answers).sum().to(scores.dtype)
            valid_num1 = (answers != -1).sum().item()

            _, pred2 = scores[:, :, :, 1].max(dim=-1)
            acc2 = (pred2 == pre_answers).sum().to(scores.dtype)
            valid_num2 = (pre_answers != -1).sum().item()

            valid_num = valid_num1 + valid_num2
            acc = (acc1 + acc2) / (valid_num * 1.0)

            output_dict["acc"] = acc
            output_dict["valid_num"] = valid_num

        return output_dict


class RobertaModelForBiSentReorderMHQ(RobertaPreTrainedModel):
    model_prefix = 'roberta_bi_sr_cs_mhq'
    config_class = RobertaConfig
    base_model_prefix = 'roberta'

    def __init__(self, config: RobertaConfig):
        super().__init__(config)

        self.roberta = RobertaModel(config)

        self.q_sum = MHAToken(config)
        self.s_sum = MHAToken(config)
        
        self.project1 = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.project2 = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

        self.half_dim = config.hidden_size // 2

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                sentence_index=None, sentence_mask=None, sent_word_mask=None,
                answers=None, pre_answers=None, **kwargs):
        seq_output = self.roberta(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  token_type_ids=token_type_ids)[0]

        sentence_index, mask, sent_mask = sentence_index, sent_word_mask, sentence_mask
        batch, sent_num, seq_len = mask.size()
        sentence_index = sentence_index.unsqueeze(-1).expand(
            -1, -1, -1, self.config.hidden_size
        ).reshape(batch, sent_num * seq_len, self.config.hidden_size)

        cls_h = seq_output[:, 0]
        hidden = seq_output.gather(dim=1, index=sentence_index).reshape(
            batch, sent_num, seq_len, -1)
        hidden = hidden * (1 - mask.unsqueeze(-1))

        query_num = answers.size(1)

        query_h = self.q_sum(cls_h.unsqueeze(1), hidden[:, :query_num], mask[:, :query_num])
        query_h = query_h.squeeze(1)
        assert query_h.size(1) == query_num

        sent_h = self.s_sum(query_h, hidden, mask)

        query_h = self.project1(query_h).view(batch, query_num, 2, self.half_dim)
        sent_h = self.project2(sent_h).view(batch, query_num, sent_num, 2, self.half_dim)

        scores = torch.einsum("bqxh,bqsxh->bqsx", query_h, sent_h)

        output_dict = {"logits": scores}
        if answers is not None and pre_answers is not None:
            sent_mask = sent_mask[:, None, :, None].expand(-1, query_num, -1, 2)
            # diag_mask = torch.eye(query_num, m=sent_num,
            #                       device=scores.device,
            #                       dtype=sent_mask.dtype).view(1, query_num, sent_num)
            # sent_mask = sent_mask.unsqueeze(1).expand(-1, query_num, -1)
            # sent_mask = torch.clamp_max(diag_mask + sent_mask, max=1.0).unsqueeze(-1)

            scores = scores + sent_mask * -10000.0

            loss1 = F.cross_entropy(input=scores[:, :, :, 0].reshape(batch * query_num, -1),
                                    target=answers.reshape(-1),
                                    ignore_index=-1, reduction='sum')
            loss2 = F.cross_entropy(input=scores[:, :, :, 1].reshape(batch * query_num, -1),
                                    target=pre_answers.reshape(-1),
                                    ignore_index=-1, reduction='sum')
            # loss1 = F.cross_entropy(input=scores[:, :, :, 0].reshape(batch * query_num, -1),
            #                         target=answers.reshape(-1),
            #                         ignore_index=-1)
            # loss2 = F.cross_entropy(input=scores[:, :, :, 1].reshape(batch * query_num, -1),
            #                         target=pre_answers.reshape(-1),
            #                         ignore_index=-1)

            loss = (loss1 + loss2) / (batch * 1.0)
            # loss = (loss1 + loss2)

            output_dict["loss"] = loss

            _, pred1 = scores[:, :, :, 0].max(dim=-1)
            acc1 = (pred1 == answers).sum().to(scores.dtype)
            valid_num1 = (answers != -1).sum().item()

            _, pred2 = scores[:, :, :, 1].max(dim=-1)
            acc2 = (pred2 == pre_answers).sum().to(scores.dtype)
            valid_num2 = (pre_answers != -1).sum().item()

            valid_num = valid_num1 + valid_num2
            acc = (acc1 + acc2) / (valid_num * 1.0)

            output_dict["acc"] = acc
            output_dict["valid_num"] = valid_num

        return output_dict


class RobertaModelForBiSentReorderMHQDouble(RobertaPreTrainedModel):
    model_prefix = 'roberta_bi_sr_cs_mhq_double'
    config_class = RobertaConfig
    base_model_prefix = 'roberta'

    def __init__(self, config: RobertaConfig):
        super().__init__(config)

        self.roberta = RobertaModel(config)

        self.q_sum = MHAToken(config)
        self.s_sum = MHAToken(config)
        
        self.project1 = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.project2 = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

        self.half_dim = config.hidden_size // 2

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                sentence_index=None, sentence_mask=None, sent_word_mask=None,
                answers=None, pre_answers=None, nn_answers=None, pp_answers=None,
                **kwargs):
        seq_output = self.roberta(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  token_type_ids=token_type_ids)[0]

        sentence_index, mask, sent_mask = sentence_index, sent_word_mask, sentence_mask
        batch, sent_num, seq_len = mask.size()
        sentence_index = sentence_index.unsqueeze(-1).expand(
            -1, -1, -1, self.config.hidden_size
        ).reshape(batch, sent_num * seq_len, self.config.hidden_size)

        cls_h = seq_output[:, 0]
        hidden = seq_output.gather(dim=1, index=sentence_index).reshape(
            batch, sent_num, seq_len, -1)
        hidden = hidden * (1 - mask.unsqueeze(-1))

        query_num = answers.size(1)

        query_h = self.q_sum(cls_h.unsqueeze(1), hidden[:, :query_num], mask[:, :query_num])
        query_h = query_h.squeeze(1)
        assert query_h.size(1) == query_num

        sent_h = self.s_sum(query_h, hidden, mask)

        query_h = self.project1(query_h).view(batch, query_num, 2, self.half_dim)
        sent_h = self.project2(sent_h).view(batch, query_num, sent_num, 2, self.half_dim)

        scores = torch.einsum("bqxh,bqsxh->bqsx", query_h, sent_h)

        output_dict = {"logits": scores}
        if answers is not None and pre_answers is not None and nn_answers is not None and pp_answers is not None:
            # sent_mask = sent_mask[:, None, :, None].expand(-1, query_num, -1, 2)
            diag_mask = torch.eye(query_num, m=sent_num,
                                  device=scores.device,
                                  dtype=sent_mask.dtype).view(1, query_num, sent_num)
            sent_mask = sent_mask.unsqueeze(1).expand(-1, query_num, -1)
            sent_mask = torch.clamp_max(diag_mask + sent_mask, max=1.0).unsqueeze(-1)
            
            scores = scores + sent_mask * -10000.0

            next_scores = scores[:, :, :, 0]
            loss_1_1, acc_1_1, valid_num_1_1 = cross_entropy_loss_and_accuracy(next_scores, answers)
            nn_scores = mask_scores_with_labels(next_scores, answers)
            loss_1_2, acc_1_2, valid_num_1_2 = cross_entropy_loss_and_accuracy(nn_scores, nn_answers)

            pref_scores = scores[:, :, :, 1]
            loss_2_1, acc_2_1, valid_num_2_1 = cross_entropy_loss_and_accuracy(pref_scores, pre_answers)
            pp_scores = mask_scores_with_labels(pref_scores, pre_answers)
            loss_2_2, acc_2_2, valid_num_2_2 = cross_entropy_loss_and_accuracy(pp_scores, pp_answers)

            loss = loss_1_1 + loss_1_2 + loss_2_1 + loss_2_2

            output_dict["loss"] = loss
            
            valid_num = valid_num_1_1 + valid_num_1_2 + valid_num_2_1 + valid_num_2_2
            acc = (acc_1_1 + acc_1_2 + acc_2_1 + acc_2_2) / (valid_num * 1.0)

            output_dict["acc"] = acc
            output_dict["valid_num"] = valid_num

        return output_dict


class RobertaModelForBiSentReorderMHQShare(RobertaPreTrainedModel):
    model_prefix = 'roberta_bi_sr_cs_mhq_share'
    base_model_prefix = 'roberta'

    def __init__(self, config: RobertaConfig):
        super().__init__(config)

        self.roberta = RobertaModel(config)

        self.q_sum = MHAToken(config)
        self.s_sum = MHAToken(config)
        
        self.project1 = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.project2 = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-1)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                sentence_index=None, sentence_mask=None, sent_word_mask=None,
                answers=None, pre_answers=None, **kwargs):
        seq_output = self.roberta(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  token_type_ids=token_type_ids)[0]

        sentence_index, mask, sent_mask = sentence_index, sent_word_mask, sentence_mask
        batch, sent_num, seq_len = mask.size()
        sentence_index = sentence_index.unsqueeze(-1).expand(
            -1, -1, -1, self.config.hidden_size
        ).reshape(batch, sent_num * seq_len, self.config.hidden_size)

        cls_h = seq_output[:, 0]
        hidden = seq_output.gather(dim=1, index=sentence_index).reshape(
            batch, sent_num, seq_len, -1)
        hidden = hidden * (1 - mask.unsqueeze(-1))

        query_num = answers.size(1)

        query_h = self.q_sum(cls_h.unsqueeze(1), hidden[:, :query_num], mask[:, :query_num])
        query_h = query_h.squeeze(1)
        # assert query_h.size(1) == query_num

        sent_h = self.s_sum(query_h, hidden, mask)

        query_h = self.project1(query_h)
        sent_h = self.project2(sent_h)

        scores = torch.einsum("bqh,bqsh->bqs", query_h, sent_h)

        output_dict = {"logits": scores}
        if answers is not None and pre_answers is not None:
            diag_mask = torch.eye(query_num, m=sent_num,
                                  device=scores.device,
                                  dtype=sent_mask.dtype).view(1, query_num, sent_num)
            sent_mask = sent_mask.unsqueeze(1).expand(-1, query_num, -1) + diag_mask

            scores = scores + sent_mask * -10000.0

            pre_masked_scores = mask_scores_with_labels(scores, pre_answers).contiguous()
            loss1 = self.loss_fct(pre_masked_scores.view(-1, sent_num), answers.view(-1))

            fol_masked_scores = mask_scores_with_labels(scores, answers).contiguous()
            loss2 = self.loss_fct(fol_masked_scores.view(-1, sent_num), pre_answers.view(-1))

            loss = loss1 + loss2

            output_dict["loss"] = loss

            valid_num1 = (answers != -1).sum().item()
            valid_num2 = (pre_answers != -1).sum().item()
            valid_num = valid_num1 + valid_num2

            pred = torch.argsort(scores.detach(), dim=-1, descending=True)[:, :, :2]

            acc1 = (pred == answers.unsqueeze(-1)).sum()
            acc2 = (pred == pre_answers.unsqueeze(-1)).sum()

            acc = (acc1 + acc2).to(dtype=scores.dtype) / (valid_num * 1.0)

            output_dict["acc"] = acc
            output_dict["valid_num"] = valid_num

        return output_dict


class RobertaModelForSentReorderMHQShareAct(RobertaPreTrainedModel):
    model_prefix = 'roberta_sr_mhq_share_act'
    base_model_prefix = 'roberta'

    def __init__(self, config: RobertaConfig):
        super().__init__(config)

        self.roberta = RobertaModel(config)

        self.q_sum = MHAToken(config)
        self.s_sum = MHAToken(config)
        
        self.prediction_head = SentenceReorderPredictionHeadDouble(config)

        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-1)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                sentence_index=None, sentence_mask=None, sent_word_mask=None,
                answers=None, pre_answers=None, **kwargs):
        seq_output = self.roberta(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  token_type_ids=token_type_ids)[0]

        sentence_index, mask, sent_mask = sentence_index, sent_word_mask, sentence_mask
        batch, sent_num, seq_len = mask.size()
        sentence_index = sentence_index.unsqueeze(-1).expand(
            -1, -1, -1, self.config.hidden_size
        ).reshape(batch, sent_num * seq_len, self.config.hidden_size)

        cls_h = seq_output[:, 0]
        hidden = seq_output.gather(dim=1, index=sentence_index).reshape(
            batch, sent_num, seq_len, -1)
        hidden = hidden * (1 - mask.unsqueeze(-1))

        query_num = answers.size(1)

        query_h = self.q_sum(
            cls_h.unsqueeze(1),
            hidden[:, :query_num],
            mask[:, :query_num],
            residual=False
        )
        query_h = query_h.squeeze(1)

        sent_h = self.s_sum(query_h, hidden, mask, residual=False)

        scores = self.prediction_head(query_h, sent_h)

        output_dict = {"logits": scores}
        if answers is not None and pre_answers is not None:
            diag_mask = torch.eye(query_num, m=sent_num,
                                  device=scores.device,
                                  dtype=sent_mask.dtype).view(1, query_num, sent_num)
            sent_mask = sent_mask.unsqueeze(1).expand(-1, query_num, -1) + diag_mask

            scores = scores + sent_mask * -10000.0

            pre_masked_scores = mask_scores_with_labels(scores, pre_answers).contiguous()
            loss1 = self.loss_fct(pre_masked_scores.view(-1, sent_num), answers.view(-1))

            fol_masked_scores = mask_scores_with_labels(scores, answers).contiguous()
            loss2 = self.loss_fct(fol_masked_scores.view(-1, sent_num), pre_answers.view(-1))

            loss = loss1 + loss2

            output_dict["loss"] = loss

            valid_num1 = (answers != -1).sum().item()
            valid_num2 = (pre_answers != -1).sum().item()
            valid_num = valid_num1 + valid_num2

            pred = torch.argsort(scores.detach(), dim=-1, descending=True)[:, :, :2]

            acc1 = (pred == answers.unsqueeze(-1)).sum()
            acc2 = (pred == pre_answers.unsqueeze(-1)).sum()

            acc = (acc1 + acc2).to(dtype=scores.dtype) / (valid_num * 1.0)

            output_dict["acc"] = acc
            output_dict["valid_num"] = valid_num

        return output_dict


class RobertaModelForSentReorderMHQShareActD(RobertaPreTrainedModel):
    model_prefix = 'roberta_sr_mhq_share_actd'
    base_model_prefix = 'roberta'

    def __init__(self, config: RobertaConfig):
        super().__init__(config)

        self.roberta = RobertaModel(config)

        self.q_sum = MHAToken(config)
        self.s_sum = MHAToken(config)
        
        self.prediction_head = SentenceReorderPredictionHeadDouble(config)

        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-1)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                sentence_index=None, sentence_mask=None, sent_word_mask=None,
                answers=None, pre_answers=None, **kwargs):
        seq_output = self.roberta(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  token_type_ids=token_type_ids)[0]

        sentence_index, mask, sent_mask = sentence_index, sent_word_mask, sentence_mask
        batch, sent_num, seq_len = mask.size()
        sentence_index = sentence_index.unsqueeze(-1).expand(
            -1, -1, -1, self.config.hidden_size
        ).reshape(batch, sent_num * seq_len, self.config.hidden_size)

        cls_h = seq_output[:, 0]
        hidden = seq_output.gather(dim=1, index=sentence_index).reshape(
            batch, sent_num, seq_len, -1)
        hidden = hidden * (1 - mask.unsqueeze(-1))

        query_num = answers.size(1)

        query_h = self.q_sum(
            cls_h.unsqueeze(1),
            hidden[:, :query_num],
            mask[:, :query_num],
            residual=False
        )
        query_h = query_h.squeeze(1)

        sent_h = self.s_sum(query_h, hidden, mask, residual=False)

        scores = self.prediction_head(query_h, sent_h)

        output_dict = {"logits": scores}
        if answers is not None and pre_answers is not None:
            # Remove self mask here.
            # diag_mask = torch.eye(query_num, m=sent_num,
            #                       device=scores.device,
            #                       dtype=sent_mask.dtype).view(1, query_num, sent_num)
            # sent_mask = sent_mask.unsqueeze(1).expand(-1, query_num, -1) + diag_mask
            sent_mask = sent_mask.unsqueeze(1)

            scores = scores + sent_mask * -10000.0

            pre_masked_scores = mask_scores_with_labels(scores, pre_answers).contiguous()
            loss1 = self.loss_fct(pre_masked_scores.view(-1, sent_num), answers.view(-1))

            fol_masked_scores = mask_scores_with_labels(scores, answers).contiguous()
            loss2 = self.loss_fct(fol_masked_scores.view(-1, sent_num), pre_answers.view(-1))

            loss = loss1 + loss2

            output_dict["loss"] = loss

            valid_num1 = (answers != -1).sum().item()
            valid_num2 = (pre_answers != -1).sum().item()
            valid_num = valid_num1 + valid_num2

            pred = torch.argsort(scores.detach(), dim=-1, descending=True)[:, :, :2]

            acc1 = (pred == answers.unsqueeze(-1)).sum()
            acc2 = (pred == pre_answers.unsqueeze(-1)).sum()

            acc = (acc1 + acc2).to(dtype=scores.dtype) / (valid_num * 1.0)

            output_dict["acc"] = acc
            output_dict["valid_num"] = valid_num

        return output_dict


class RobertaModelForBiSentReorderMHQShareDouble(RobertaPreTrainedModel):
    model_prefix = 'roberta_bi_sr_cs_mhq_share_double'
    base_model_prefix = 'roberta'

    def __init__(self, config: RobertaConfig):
        super().__init__(config)

        self.roberta = RobertaModel(config)

        self.q_sum = MHAToken(config)
        self.s_sum = MHAToken(config)
        
        self.project1 = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.project2 = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-1)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                sentence_index=None, sentence_mask=None, sent_word_mask=None,
                answers=None, pre_answers=None, nn_answers=None, pp_answers=None,
                **kwargs):
        seq_output = self.roberta(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  token_type_ids=token_type_ids)[0]

        sentence_index, mask, sent_mask = sentence_index, sent_word_mask, sentence_mask
        batch, sent_num, seq_len = mask.size()
        sentence_index = sentence_index.unsqueeze(-1).expand(
            -1, -1, -1, self.config.hidden_size
        ).reshape(batch, sent_num * seq_len, self.config.hidden_size)

        cls_h = seq_output[:, 0]
        hidden = seq_output.gather(dim=1, index=sentence_index).reshape(
            batch, sent_num, seq_len, -1)
        hidden = hidden * (1 - mask.unsqueeze(-1))

        query_num = answers.size(1)

        query_h = self.q_sum(cls_h.unsqueeze(1), hidden[:, :query_num], mask[:, :query_num])
        query_h = query_h.squeeze(1)

        sent_h = self.s_sum(query_h, hidden, mask)

        query_h = self.project1(query_h)
        sent_h = self.project2(sent_h)

        scores = torch.einsum("bqh,bqsh->bqs", query_h, sent_h)

        output_dict = {"logits": scores}
        if answers is not None and pre_answers is not None:
            diag_mask = torch.eye(query_num, m=sent_num,
                                  device=scores.device,
                                  dtype=sent_mask.dtype).view(1, query_num, sent_num)
            sent_mask = sent_mask.unsqueeze(1).expand(-1, query_num, -1) + diag_mask

            scores = scores + sent_mask * -10000.0

            pre_masked_scores = mask_scores_with_labels(scores, pre_answers).contiguous()
            loss1 = self.loss_fct(pre_masked_scores.view(-1, sent_num), answers.view(-1))

            fol_masked_scores = mask_scores_with_labels(scores, answers).contiguous()
            loss2 = self.loss_fct(fol_masked_scores.view(-1, sent_num), pre_answers.view(-1))

            scores_next = mask_scores_with_labels(pre_masked_scores, answers).contiguous()
            
            pre_masked_scores_next = mask_scores_with_labels(scores_next, pp_answers).contiguous()
            loss3 = self.loss_fct(pre_masked_scores_next.view(-1, sent_num),
                                  nn_answers.view(-1))

            fol_masked_scores_next = mask_scores_with_labels(scores_next, nn_answers).contiguous()
            loss4 = self.loss_fct(fol_masked_scores_next.view(-1, sent_num),
                                  pp_answers.view(-1))

            loss = loss1 + loss2 + loss3 + loss4

            output_dict["loss"] = loss

            valid_num1 = (answers != -1).sum().item()
            valid_num2 = (pre_answers != -1).sum().item()
            valid_num3 = (pp_answers != -1).sum().item()
            valid_num4 = (nn_answers != -1).sum().item()
            valid_num = valid_num1 + valid_num2 + valid_num3 + valid_num4

            pred = torch.argsort(scores.detach(), dim=-1, descending=True)[:, :, :4]

            acc1 = (pred[:, :, :2] == answers.unsqueeze(-1)).sum()
            acc2 = (pred[:, :, :2] == pre_answers.unsqueeze(-1)).sum()
            acc3 = (pred[:, :, 2:] == nn_answers.unsqueeze(-1)).sum()
            acc4 = (pred[:, :, 2:] == pp_answers.unsqueeze(-1)).sum()

            acc = (acc1 + acc2 + acc3 + acc4).to(dtype=scores.dtype) / (valid_num * 1.0)

            output_dict["acc"] = acc
            output_dict["valid_num"] = valid_num

        return output_dict


class RobertaModelForSentReorderShare(RobertaPreTrainedModel):
    model_prefix = 'roberta_bi_sr_share'
    base_model_prefix = 'roberta'

    def __init__(self, config: RobertaConfig):
        super().__init__(config)

        self.roberta = RobertaModel(config)

        self.q_sum = nn.Linear(config.hidden_size, config.hidden_size)
        self.s_sum = nn.Linear(config.hidden_size, config.hidden_size)
        
        self.project1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.project2 = nn.Linear(config.hidden_size, config.hidden_size)

        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-1)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                sentence_index=None, sentence_mask=None, sent_word_mask=None,
                answers=None, pre_answers=None, **kwargs):
        seq_output = self.roberta(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  token_type_ids=token_type_ids)[0]

        sentence_index, mask, sent_mask = sentence_index, sent_word_mask, sentence_mask
        batch, sent_num, seq_len = mask.size()
        sentence_index = sentence_index.unsqueeze(-1).expand(
            -1, -1, -1, self.config.hidden_size
        ).reshape(batch, sent_num * seq_len, self.config.hidden_size)

        cls_h = seq_output[:, 0]
        hidden = seq_output.gather(dim=1, index=sentence_index).reshape(
            batch, sent_num, seq_len, -1)
        hidden = hidden * (1 - mask.unsqueeze(-1))

        query_num = answers.size(1)

        query_h, _ = sentence_sum(
            self.q_sum(cls_h), hidden[:, :query_num], mask[:, :query_num]
        )

        sent_h, _ = mul_sentence_sum(self.s_sum(query_h), hidden, mask)

        scores = torch.einsum(
            "bqh,bqsh->bqs",
            self.project1(query_h),
            self.project2(sent_h)
        )

        output_dict = {"logits": scores}
        if answers is not None and pre_answers is not None:
            diag_mask = torch.eye(query_num, m=sent_num,
                                  device=scores.device,
                                  dtype=sent_mask.dtype).view(1, query_num, sent_num)
            sent_mask = sent_mask.unsqueeze(1).expand(-1, query_num, -1) + diag_mask

            scores = scores + sent_mask * -10000.0

            pre_masked_scores = mask_scores_with_labels(scores, pre_answers).contiguous()
            loss1 = self.loss_fct(pre_masked_scores.view(-1, sent_num), answers.view(-1))

            fol_masked_scores = mask_scores_with_labels(scores, answers).contiguous()
            loss2 = self.loss_fct(fol_masked_scores.view(-1, sent_num), pre_answers.view(-1))

            loss = loss1 + loss2

            output_dict["loss"] = loss

            valid_num1 = (answers != -1).sum().item()
            valid_num2 = (pre_answers != -1).sum().item()
            valid_num = valid_num1 + valid_num2

            pred = torch.argsort(scores.detach(), dim=-1, descending=True)[:, :, :2]

            acc1 = (pred == answers.unsqueeze(-1)).sum()
            acc2 = (pred == pre_answers.unsqueeze(-1)).sum()

            acc = (acc1 + acc2).to(dtype=scores.dtype) / (valid_num * 1.0)

            output_dict["acc"] = acc
            output_dict["valid_num"] = valid_num

        return output_dict


class RobertaSRDecoderConfig(RobertaConfig):
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


class RobertaModelForSentReorderDecoder(RobertaPreTrainedModel):
    model_prefix = 'roberta_srd'
    config_class = RobertaSRDecoderConfig
    base_model_prefix = 'roberta'

    def __init__(self, config: RobertaConfig):
        super().__init__(config)

        self.roberta = RobertaModel(config)

        self.sentence_sum = SentenceSum(config.hidden_size)
        self.sentecne_reorder_head = SentenceReorderDecoder(config)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                sentence_spans=None, sentence_index=None, sentence_mask=None,
                sent_word_mask=None, query_mask=None, answers=None, edge_index=None, edges=None):

        seq_output = self.roberta(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  token_type_ids=token_type_ids)[0]

        sentence_index, mask, sent_mask = sentence_index, sent_word_mask, sentence_mask
        batch, sent_num, seq_len = mask.size()
        sentence_index = sentence_index.unsqueeze(-1).expand(
            -1, -1, -1, self.config.hidden_size
        ).reshape(batch, sent_num * seq_len, self.config.hidden_size)

        cls_h = seq_output[:, 0]
        hidden = seq_output.gather(dim=1, index=sentence_index).reshape(
            batch, sent_num, seq_len, -1)
        hidden = self.dropout(hidden)

        hidden = self.sentence_sum(cls_h, hidden, mask)

        decoder_output_logits = self.sentecne_reorder_head(
            st_hidden=cls_h,
            hidden=hidden,
            mask=sent_mask,
            true_index=answers
        )

        output_dict = {"logits": decoder_output_logits}
        if answers is not None:
            scores = decoder_output_logits

            if query_mask is not None:
                answers[~query_mask] = -1

            scores = scores + sent_mask[:, None, :] * -65500.0
            loss = F.cross_entropy(scores.reshape(-1, sent_num), answers.reshape(-1),
                                   ignore_index=-1, reduction='sum') / (batch * 1.0)
            output_dict["loss"] = loss

            _, pred = scores.max(dim=-1)
            valid_num = torch.sum(answers != -1)
            acc = torch.sum(pred == answers).to(
                dtype=scores.dtype) / (valid_num * 1.0)
            output_dict["acc"] = acc
            output_dict["valid_num"] = valid_num

        return output_dict


class RobertaModelForSentReorderWithPosition(RobertaPreTrainedModel):
    model_prefix = 'roberta_srp'
    config_class = RobertaSRDecoderConfig
    base_model_prefix = 'roberta'

    def __init__(self, config: RobertaConfig):
        super().__init__(config)

        self.roberta = RobertaModel(config)

        self.sentence_sum = SentenceSum(config.hidden_size)
        self.sentence_reorder_head = SentenceReorderWithPosition(config)

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                sentence_index=None, sentence_mask=None, sent_word_mask=None,
                query_mask=None, answers=None, **kwargs):

        seq_output = self.roberta(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  token_type_ids=token_type_ids)[0]

        sentence_index, mask, sent_mask = sentence_index, sent_word_mask, sentence_mask
        batch, sent_num, seq_len = mask.size()
        sentence_index = sentence_index.unsqueeze(-1).expand(
            -1, -1, -1, self.config.hidden_size
        ).reshape(batch, sent_num * seq_len, self.config.hidden_size)

        cls_h = seq_output[:, 0]
        hidden = seq_output.gather(dim=1, index=sentence_index).reshape(
            batch, sent_num, seq_len, -1)
        # hidden = self.dropout(hidden)

        hidden = self.sentence_sum(cls_h, hidden, mask)

        sent_mask = sent_mask.to(dtype=hidden.dtype)
        hidden = hidden * (1 - sent_mask[:, :, None])

        sentence_reorder_logits = self.sentence_reorder_head(
            cls_h=cls_h,
            hidden=hidden
        )

        output_dict = {"logits": sentence_reorder_logits}
        if answers is not None:
            scores = sentence_reorder_logits.float()
            scores = scores + sent_mask[:, None, :] * -10000.0

            loss = F.cross_entropy(scores.reshape(-1, sent_num), answers.reshape(-1),
                                   ignore_index=-1, reduction='sum') / (batch * 1.0)

            output_dict["loss"] = loss

            _, pred = scores.max(dim=-1)
            valid_num = torch.sum(answers != -1).item()
            acc = torch.sum(pred == answers).to(dtype=scores.dtype) / (valid_num * 1.0)
            output_dict["acc"] = acc
            output_dict["valid_num"] = valid_num

        return output_dict


class RobertaForMCRC(RobertaPreTrainedModel):
    model_prefix = 'roberta_mcrc'
    config_class = RobertaConfig
    base_model_prefix = 'roberta'

    def __init__(self, config: RobertaConfig):
        super().__init__(config)

        self.roberta = RobertaModel(config)

        self.classifier = nn.Linear(config.hidden_size, 1)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.init_weights()

    @staticmethod
    def fold_tensor(x):
        if x is None:
            return None
        return x.reshape(x.size(0) * x.size(1), *x.size()[2:])

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                labels=None, **kwargs):

        batch, num_choice, _ = input_ids.size()
        input_ids = self.fold_tensor(input_ids)
        token_type_ids = self.fold_tensor(token_type_ids)
        attention_mask = self.fold_tensor(attention_mask)

        sequence_output = self.roberta(input_ids=input_ids,
                                       token_type_ids=token_type_ids,
                                       attention_mask=attention_mask)[0]

        cls_h = sequence_output[:, 0].contiguous()
        logits = self.classifier(self.dropout(cls_h)).view(batch, num_choice)

        outputs = (logits,)
        if labels is not None:
            loss = F.cross_entropy(logits, labels, ignore_index=-1)
            outputs = (loss,) + outputs

            _, pred = logits.max(dim=-1)
            acc = torch.sum(pred == labels) / (1.0 * batch)

            outputs = outputs + (acc,)

        return outputs


class RobertaPoolForMCRC(RobertaPreTrainedModel):
    model_prefix = 'roberta_pool_mcrc'
    config_class = RobertaConfig
    base_model_prefix = 'roberta'

    def __init__(self, config: RobertaConfig):
        super().__init__(config)

        self.roberta = RobertaModel(config)

        self.classifier = nn.Linear(config.hidden_size, 1)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.init_weights()

    @staticmethod
    def fold_tensor(x):
        if x is None:
            return None
        return x.reshape(x.size(0) * x.size(1), *x.size()[2:])

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                labels=None, **kwargs):

        batch, num_choice, _ = input_ids.size()
        input_ids = self.fold_tensor(input_ids)
        token_type_ids = self.fold_tensor(token_type_ids)
        attention_mask = self.fold_tensor(attention_mask)

        pool_output = self.roberta(input_ids=input_ids,
                                   token_type_ids=token_type_ids,
                                   attention_mask=attention_mask)[1]

        logits = self.classifier(self.dropout(pool_output)).view(batch, num_choice)

        outputs = (logits,)
        if labels is not None:
            loss = F.cross_entropy(logits, labels, ignore_index=-1)
            outputs = (loss,) + outputs

            _, pred = logits.max(dim=-1)
            acc = torch.sum(pred == labels) / (1.0 * batch)

            outputs = outputs + (acc,)

        return outputs


class RobertaHAMCRCConfig(RobertaConfig):
    added_configs = [
        'cls_type', 'no_residual', 'use_new_pro'
    ]

    def __init__(self, cls_type=0, no_residual=False, use_new_pro=False, **kwargs):
        super().__init__(**kwargs)

        self.cls_type = cls_type
        self.no_residual = no_residual
        self.use_new_pro = use_new_pro

    def expand_configs(self, *args):
        self.added_configs.extend(list(args))


class RobertaModelHAForMCRC(RobertaPreTrainedModel):
    model_prefix = 'roberta_ha_mcrc'
    config_class = RobertaHAMCRCConfig
    base_model_prefix = 'roberta'

    def __init__(self, config: RobertaHAMCRCConfig):
        super().__init__(config)

        self.roberta = RobertaModel(config)

        self.sentence_reorder_head = SentenceReorderHead(config.hidden_size)

        if config.cls_type == 0:
            self.classifier = nn.Linear(config.hidden_size * 4, 1)
        elif config.cls_type == 1:
            self.classifier = nn.Sequential(
                nn.Linear(config.hidden_size * 4, config.hidden_size),
                nn.Tanh(),
                nn.Linear(config.hidden_size, 1)
            )
        else:
            raise RuntimeError()

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

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

        sentence_index, mask, sent_mask = sentence_index, sent_word_mask, sentence_mask

        fb, sent_num, seq_len = mask.size()
        sentence_index = sentence_index.unsqueeze(-1).expand(
            -1, -1, -1, self.config.hidden_size
        ).reshape(fb, sent_num * seq_len, self.config.hidden_size)

        cls_h = seq_output[:, 0]
        hidden = seq_output.gather(dim=1, index=sentence_index).reshape(
            fb, sent_num, seq_len, -1)
        hidden = self.dropout(hidden)

        query_num = 2  # query[0]: question, query[1]: option
        scores, (hidden, q_op_q, hidden_kv) = self.sentence_reorder_head(
            cls_h, hidden, mask, query_num=2)

        assert scores.size(1) == query_num
        scores = scores[:, :, query_num:] + sent_mask[:, None, query_num:] * -65500.0
        alpha = torch.softmax(scores, dim=-1)
        att_sent_h = alpha.bmm(hidden_kv[:, query_num:]).view(batch, num_choice,
                                                              query_num * hidden.size(-1))
        q_op_init = hidden[:, :query_num].view(batch, num_choice, -1)

        logits = self.classifier(self.dropout(torch.cat([q_op_init, att_sent_h], dim=-1)))
        logits = logits.squeeze(-1)

        outputs = (logits,)
        if labels is not None:
            loss = F.cross_entropy(logits, labels, ignore_index=-1)
            outputs = (loss,) + outputs

            _, pred = logits.max(dim=-1)
            acc = torch.sum(pred == labels) / (1.0 * batch)

            outputs = outputs + (acc,)

        return outputs


class RobertaModelHACBForMCRC(RobertaPreTrainedModel):
    model_prefix = 'roberta_ha_cb_mcrc'
    config_class = RobertaHAMCRCConfig
    base_model_prefix = 'roberta'
    """
    question and option are combined as single sentence.
    """

    def __init__(self, config: RobertaHAMCRCConfig):
        super().__init__(config)

        self.roberta = RobertaModel(config)

        self.sentence_reorder_head = SentenceReorderHead(config.hidden_size)

        self.cls_w = self.sentence_reorder_head.cls_w
        self.project1 = self.sentence_reorder_head.project1
        self.project2 = self.sentence_reorder_head.project2

        if config.cls_type == 0:
            self.classifier = nn.Linear(config.hidden_size * 2, 1)
        elif config.cls_type == 1:
            self.classifier = nn.Sequential(
                nn.Linear(config.hidden_size * 2, config.hidden_size),
                nn.Tanh(),
                nn.Linear(config.hidden_size, 1)
            )
        else:
            raise RuntimeError()

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

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

        sentence_index, mask, sent_mask = sentence_index, sent_word_mask, sentence_mask

        fb, sent_num, seq_len = mask.size()
        sentence_index = sentence_index.unsqueeze(-1).expand(
            -1, -1, -1, self.config.hidden_size
        ).reshape(fb, sent_num * seq_len, self.config.hidden_size)

        cls_h = seq_output[:, 0]
        hidden = seq_output.gather(dim=1, index=sentence_index).reshape(
            fb, sent_num, seq_len, -1)
        hidden = self.dropout(hidden)

        scores = torch.einsum("bh,bsth->bst", self.cls_w(cls_h), hidden)
        scores = scores + mask * -65500.0

        q_op_alpha = torch.softmax(scores[:, :2].reshape(fb, 1, -1), dim=-1)
        q_op_h = q_op_alpha.bmm(hidden[:, :2].reshape(fb, -1, hidden.size(-1)))

        d_alpha = torch.softmax(scores[:, 2:], dim=-1)
        d_h = torch.einsum("bst,bsth->bsh", d_alpha, hidden[:, 2:])

        query = self.project1(q_op_h)
        key = self.project2(d_h)
        scores = query.bmm(key.transpose(1, 2)) + sent_mask[:, None, 2:] * -65500.0
        alpha = torch.softmax(scores, dim=-1)

        att_sent_h = alpha.bmm(key).squeeze(1)

        logits = self.classifier(self.dropout(torch.cat([q_op_h.squeeze(1), att_sent_h], dim=-1)))
        logits = logits.view(batch, num_choice)

        outputs = (logits,)
        if labels is not None:
            loss = F.cross_entropy(logits, labels, ignore_index=-1)
            outputs = (loss,) + outputs

            _, pred = logits.max(dim=-1)
            acc = torch.sum(pred == labels) / (1.0 * batch)

            outputs = outputs + (acc,)

        return outputs


class RobertaModelQueryForMCRC(RobertaPreTrainedModel):
    model_prefix = 'roberta_query_mcrc'
    config_class = RobertaHAMCRCConfig
    base_model_prefix = 'roberta'

    def __init__(self, config: RobertaHAMCRCConfig):
        super().__init__(config)

        self.roberta = RobertaModel(config)

        self.sentecne_reorder_head = SentenceReorderHead(config.hidden_size)
        self.cls_w = self.sentecne_reorder_head.cls_w

        if config.cls_type == 0:
            self.classifier = nn.Linear(config.hidden_size, 1)
        elif config.cls_type == 1:
            self.classifier = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.Tanh(),
                nn.Linear(config.hidden_size, 1)
            )
        else:
            raise RuntimeError()

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.init_weights()

    @staticmethod
    def fold_tensor(x):
        if x is None:
            return None
        return x.reshape(x.size(0) * x.size(1), *x.size()[2:])

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                labels=None, **kwargs):

        batch, num_choice, _ = input_ids.size()
        input_ids = self.fold_tensor(input_ids)
        token_type_ids = self.fold_tensor(token_type_ids)
        attention_mask = self.fold_tensor(attention_mask)

        seq_output = self.roberta(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  token_type_ids=token_type_ids)[0]

        cls_h = self.cls_w(seq_output[:, 0])
        alpha = torch.einsum("bh,bsh->bs", cls_h, seq_output[:, 1:])
        alpha = torch.softmax(alpha + (1 - attention_mask[:, 1:]) * -65500.0, dim=-1)
        seq_h = torch.einsum("bs,bsh->bh", alpha, seq_output[:, 1:])

        logits = self.classifier(self.dropout(seq_h)).view(batch, num_choice)

        outputs = (logits,)
        if labels is not None:
            loss = F.cross_entropy(logits, labels, ignore_index=-1)
            outputs = (loss,) + outputs

            _, pred = logits.max(dim=-1)
            acc = torch.sum(pred == labels) / (1.0 * batch)

            outputs = outputs + (acc,)

        return outputs


roberta_model_map = {
    RobertaForMaskedLM.model_prefix: RobertaForMaskedLM,

    RobertaModelForSentReorder.model_prefix: RobertaModelForSentReorder,
    RobertaModelForSentReorderQ.model_prefix: RobertaModelForSentReorderQ,
    RobertaModelForSentReorderMHQ.model_prefix: RobertaModelForSentReorderMHQ,
    # RobertaModelForBiSentReorder.model_prefix: RobertaModelForBiSentReorder,
    RobertaModelForBiSentReorderQ.model_prefix: RobertaModelForBiSentReorderQ,
    RobertaModelForBiSentReorderMHQ.model_prefix: RobertaModelForBiSentReorderMHQ,
    RobertaModelForBiSentReorderMHQDouble.model_prefix: RobertaModelForBiSentReorderMHQDouble,

    RobertaModelForBiSentReorderMHQShare.model_prefix: RobertaModelForBiSentReorderMHQShare,
    RobertaModelForBiSentReorderMHQShareDouble.model_prefix: RobertaModelForBiSentReorderMHQShareDouble,
    
    RobertaModelForSentReorderMHQShareAct.model_prefix: RobertaModelForSentReorderMHQShareAct,
    RobertaModelForSentReorderMHQShareActD.model_prefix: RobertaModelForSentReorderMHQShareActD,

    RobertaModelForSentReorderShare.model_prefix: RobertaModelForSentReorderShare,

    RobertaModelForBiSentReorderCL.model_prefix: RobertaModelForBiSentReorderCL,
    RobertaModelForBiSentReorderCLQ.model_prefix: RobertaModelForBiSentReorderCLQ,
    RobertaModelForBiSentReorderCLMHQ.model_prefix: RobertaModelForBiSentReorderCLMHQ,
    RobertaModelForSentReorderDecoder.model_prefix: RobertaModelForSentReorderDecoder,
    RobertaModelForSentReorderWithPosition.model_prefix: RobertaModelForSentReorderWithPosition,

    RobertaForMCRC.model_prefix: RobertaForMCRC,
    RobertaPoolForMCRC.model_prefix: RobertaPoolForMCRC,
    RobertaModelHAForMCRC.model_prefix: RobertaModelHAForMCRC,
    RobertaModelHACBForMCRC.model_prefix: RobertaModelHACBForMCRC,
    RobertaModelQueryForMCRC.model_prefix: RobertaModelQueryForMCRC
}
