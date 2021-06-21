import torch
from torch import nn
from torch.nn import functional as F
from transformers import BertPreTrainedModel, BertModel

from modules import layers


class BertSelfSupPretain(BertPreTrainedModel):
    """
    Pre-training BERT backbone or together with LinearSelfAttn
    """

    def __init__(self, config):
        super().__init__(config)
        print(f'The model {self.__class__.__name__} is loading...')

        layers.set_seq_dropout(True)
        layers.set_my_dropout_prob(config.hidden_dropout_prob)

        self.bert = BertModel(config)

        self.sent_self_attn = layers.LinearSelfAttnAllennlp(config.hidden_size)

        self.project1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.project2 = nn.Linear(config.hidden_size, config.hidden_size)

        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, answers=None,
                p_sentence_spans=None, q_sentence_spans=None):
        sequence_output = self.bert(input_ids=input_ids,
                                    token_type_ids=token_type_ids,
                                    attention_mask=attention_mask)[0]

        # mask: 1 for masked value and 0 for true value
        hidden, mask, sent_mask = split_doc_sen_que(sequence_output, q_sentence_spans, p_sentence_spans)

        batch, sent_num, seq_len = mask.size()
        hidden = hidden.view(batch * sent_num, seq_len, -1)
        mask = mask.view(batch * sent_num, seq_len)

        alpha = self.sent_self_attn(hidden, mask)
        hidden = alpha.unsqueeze(1).bmm(hidden).squeeze().reshape(batch, sent_num, -1)
        assert hidden.size(-1) == sequence_output.size(-1)

        query = self.project1(hidden)
        key = self.project2(hidden)
        scores = query.bmm(key.transpose(1, 2))

        output_dict = {"logits": scores}
        if answers is not None:
            if sent_num > answers.size(1):
                scores = scores[:, :answers.size(1)]
            elif answers.size(1) > sent_num:
                answers = answers[:, :sent_num]
            assert answers.size(1) == scores.size(1)
            scores = scores + sent_mask[:, None, :] * -10000.0
            loss = F.cross_entropy(scores.reshape(-1, sent_num), answers.reshape(-1), ignore_index=-1,
                                   reduction='sum') / (batch * 1.0)
            output_dict["loss"] = loss

            _, pred = scores.max(dim=-1)
            valid_num = torch.sum(answers != -1)
            acc = torch.sum(pred == answers).to(dtype=scores.dtype) / (valid_num * 1.0)
            output_dict["acc"] = acc
            output_dict["valid_num"] = valid_num

        return output_dict


class BertSelfSupPretainClsQuery(BertPreTrainedModel):
    """
    Pre-training BERT backbone or together with LinearSelfAttn.
    Use representation of [CLS] as query to make it trained for downstream task.
    """
    model_prefix = 'self_sup_pretrain_cls_query'

    def __init__(self, config):
        super().__init__(config)
        print(f'The model {self.__class__.__name__} is loading...')

        layers.set_seq_dropout(True)
        layers.set_my_dropout_prob(config.hidden_dropout_prob)

        self.bert = BertModel(config)

        self.cls_w = nn.Linear(config.hidden_size, config.hidden_size)

        self.project1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.project2 = nn.Linear(config.hidden_size, config.hidden_size)

        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, answers=None,
                p_sentence_spans=None, q_sentence_spans=None):
        sequence_output = self.bert(input_ids=input_ids,
                                    token_type_ids=token_type_ids,
                                    attention_mask=attention_mask)[0]

        # mask: 1 for masked value and 0 for true value
        hidden, mask, sent_mask, cls_h = split_doc_sen_que(sequence_output, q_sentence_spans, p_sentence_spans,
                                                           sep_cls=True)

        batch, sent_num, seq_len = mask.size()
        # hidden = hidden.view(batch * sent_num, seq_len, -1)
        # hidden = hidden.view(batch, sent_num * seq_len, -1)
        # mask = mask.view(batch * sent_num, seq_len)

        hidden = layers.dropout(hidden, p=layers.my_dropout_p, training=self.training)

        cls_h = self.cls_w(cls_h)  # [batch, h]
        alpha = torch.einsum('bh,bsth->bst', cls_h, hidden)
        alpha = (alpha + mask * -10000.0).softmax(dim=-1)
        hidden = torch.einsum('bst,bsth->bsh', alpha, hidden)

        query = self.project1(hidden)
        key = self.project2(hidden)
        scores = query.bmm(key.transpose(1, 2))

        output_dict = {"logits": scores}
        if answers is not None:
            if sent_num > answers.size(1):
                scores = scores[:, :answers.size(1)]
            elif answers.size(1) > sent_num:
                answers = answers[:, :sent_num]
            assert answers.size(1) == scores.size(1)
            scores = scores + sent_mask[:, None, :] * -10000.0
            loss = F.cross_entropy(scores.reshape(-1, sent_num), answers.reshape(-1), ignore_index=-1,
                                   reduction='sum') / (batch * 1.0)
            output_dict["loss"] = loss

            _, pred = scores.max(dim=-1)
            valid_num = torch.sum(answers != -1)
            acc = torch.sum(pred == answers).to(dtype=scores.dtype) / (valid_num * 1.0)
            output_dict["acc"] = acc
            output_dict["valid_num"] = valid_num

        return output_dict


def split_sentence(hidden_state, sentence_spans):
    batch, seq_len, h = hidden_state.size()

    max_sent_len = 0
    for b in range(batch):
        max_sent_len = max(max_sent_len, max(map(lambda x: x[1] - x[0] + 1, sentence_spans[b])))

    max_sent_num = max(map(lambda x: len(x), sentence_spans))

    output = hidden_state.new_zeros((batch, max_sent_num, max_sent_len, h))
    mask = hidden_state.new_ones(batch, max_sent_num, max_sent_len)

    for b in range(batch):
        for sent_id, (start, end) in enumerate(sentence_spans[b]):
            lens = end - start + 1
            output[b][sent_id][:lens] = hidden_state[b][start:(end + 1)]
            mask[b][sent_id][:lens] = hidden_state.new_zeros(lens)
    return output, mask


def split_doc_sen_que(hidden_state, q_sentence_spans, p_sentence_spans, sep_cls=False):
    # q_hidden, q_mask = split_sentence(hidden_state, q_sentence_spans)
    # p_hidden, p_mask = split_sentence(hidden_state, p_sentence_spans)
    # return q_hidden, q_mask, p_hidden, p_mask

    cls_h = hidden_state[:, 0]

    batch = hidden_state.size(0)
    h = hidden_state.size(-1)

    # print(hidden_state.size())
    # print(len(q_sentence_spans))

    max_sent_len = 0
    for b in range(batch):
        max_sent_len = max(max_sent_len, max(map(lambda x: x[1] - x[0] + 1, q_sentence_spans[b] + p_sentence_spans[b])))
    max_sent_num = max(map(lambda x: len(x[0]) + len(x[1]), zip(q_sentence_spans, p_sentence_spans)))

    # print(max_sent_len)
    # print(max_sent_num)

    output = hidden_state.new_zeros((batch, max_sent_num, max_sent_len, h))
    mask = hidden_state.new_ones(batch, max_sent_num, max_sent_len)
    sent_mask = hidden_state.new_ones(batch, max_sent_num)

    for b in range(batch):
        for sent_id, (start, end) in enumerate(q_sentence_spans[b] + p_sentence_spans[b]):
            if sep_cls and start == 0:
                assert end >= 1
                start += 1
            lens = end - start + 1
            output[b][sent_id][:lens] = hidden_state[b][start:(end + 1)]
            mask[b][sent_id][:lens] = hidden_state.new_zeros(lens)
            sent_mask[b][sent_id] = 0
            # print(b, sent_id, lens, start, end)

    if sep_cls:
        return output, mask, sent_mask, cls_h
    return output, mask, sent_mask
