import torch
from torch import nn
from torch.nn import functional as F
from transformers import BertPreTrainedModel, BertModel

from data import MetricType
from modules import layers


class BertSelfSupPretainClsQuery(BertPreTrainedModel):
    """
    Pre-training BERT backbone or together with LinearSelfAttn.
    Use representation of [CLS] as query to make it trained for downstream task.

    This model is used to test the processor with concepts relevant extra inputs.
    """
    model_prefix = 'self_sup_pretrain_cls_query_test'

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

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                sentence_spans=None, answers=None, edge_index=None, edges=None):
        sequence_output = self.bert(input_ids=input_ids,
                                    token_type_ids=token_type_ids,
                                    attention_mask=attention_mask)[0]

        # mask: 1 for masked value and 0 for true value
        hidden, mask, sent_mask, cls_h = split_doc_sen_que(sequence_output, sentence_spans,
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

        query_num = answers.size(1)

        query = self.project1(hidden[:, :query_num])
        key = self.project2(hidden)
        scores = query.bmm(key.transpose(1, 2))  # [batch, query_num, sent_num]

        output_dict = {"logits": scores}
        if answers is not None:
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
            output_dict["knowledge_loss"] = torch.Tensor([0])

        return output_dict


def split_doc_sen_que(hidden_state, sentence_spans, sep_cls=False):
    cls_h = hidden_state[:, 0]

    batch = hidden_state.size(0)
    h = hidden_state.size(-1)

    max_sent_len = 0
    for b in range(batch):
        max_sent_len = (sentence_spans[:, :, 1] - sentence_spans[:, :, 0] + 1).max().item()

    max_sent_num = sentence_spans.size(1)
    # max_sent_num = (sentence_spans[:, :, 0] > -1).sum(dim=1).max().item()

    output = hidden_state.new_zeros((batch, max_sent_num, max_sent_len, h))
    mask = hidden_state.new_ones(batch, max_sent_num, max_sent_len)
    sent_mask = hidden_state.new_ones(batch, max_sent_num)

    """
    This process can be implemented via torch.gather, and the `sentence_spans` should indicates the token index, not the section.
    As a result, the memory of sentence_spans can be enlarged `max_sent_len` times more. But the process can be accelerated with GPU.
    """

    for b in range(batch):
        for sent_id, sec in enumerate(sentence_spans[b]):
            start = sec[0].item()
            end = sec[1].item()
            if start == -1 and end == -1:
                break
            if sep_cls and start == 0:
                if end == 0:
                    continue
                start += 1
            lens = end - start + 1
            output[b][sent_id][:lens] = hidden_state[b][start:(end + 1)]
            mask[b][sent_id][:lens] = hidden_state.new_zeros(lens)
            sent_mask[b][sent_id] = 0

    if sep_cls:
        return output, mask, sent_mask, cls_h
    return output, mask, sent_mask
