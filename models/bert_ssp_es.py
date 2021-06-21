import torch
from torch import nn
from torch.nn import functional as F
from transformers import BertPreTrainedModel, BertModel, BertConfig

from data import MetricType
from modules import layers


class SSPESConfig(BertConfig):
    added_configs = [
        'graph_loss', 'es_layer_id'
    ]

    def __init__(self, graph_loss=0.0, es_layer_id=9, **kwargs):
        super().__init__(**kwargs)

        self.graph_loss = graph_loss
        self.es_layer_id = es_layer_id


class BertSelfSupPretainExtraSup(BertPreTrainedModel):
    """
    Pre-training BERT backbone or together with LinearSelfAttn.
    Use representation of [CLS] as query to make it trained for downstream task.

    This model incorporates extra supervision in the media layers of BERT to incorporate
    edges information.
    """
    model_prefix = 'ssp_cls_query_es'
    config_class = SSPESConfig

    def __init__(self, config: SSPESConfig):
        super().__init__(config)
        print(f'The model {self.__class__.__name__} is loading...')

        self.bert = BertModel(config)

        self.es_layer_id = config.es_layer_id
        self.graph_loss = config.graph_loss

        self.es_cls_q = nn.Linear(config.hidden_size, config.hidden_size)
        self.es_kv = nn.Linear(config.hidden_size, config.hidden_size)

        self.cls_w = nn.Linear(config.hidden_size, config.hidden_size)

        self.sen_sum = layers.sentence_sum

        self.project1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.project2 = nn.Linear(config.hidden_size, config.hidden_size)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                sentence_spans=None, answers=None, edge_index=None, edges=None):

        outputs = self.bert(input_ids=input_ids,
                            token_type_ids=token_type_ids,
                            attention_mask=attention_mask,
                            output_hidden_states=True)
        sequence_output = outputs[0]
        mid_hidden = outputs[2][self.es_layer_id]

        # mask: 1 for masked value and 0 for true value
        hidden, index_output, mask, sent_mask, cls_h = split_doc_sen_que(sequence_output, sentence_spans,
                                                                         sep_cls=True)
        batch, sent_num, seq_len = mask.size()

        # extra supervision
        index = index_output.unsqueeze(-1).expand(-1, -1,
                                                  self.config.hidden_size)
        mid_q = mid_hidden[:, 0]
        mid_h = torch.gather(mid_hidden, dim=1, index=index).reshape(
            batch, sent_num, seq_len, -1)
        mid_h, _ = self.sen_sum(self.es_cls_q(mid_q), self.es_kv(mid_h), mask)
        edge_scores_norm = torch.bmm(
            mid_h, mid_h.transpose(1, 2)).softmax(dim=-1).contiguous()

        # sentence re-ordering
        hidden = self.dropout(hidden)
        hidden, _ = self.sen_sum(self.cls_w(cls_h), hidden, mask)

        query_num = answers.size(1)

        query = self.project1(hidden[:, :query_num])
        key = self.project2(hidden)
        # [batch, query_num, sent_num]
        scores = query.bmm(key.transpose(1, 2)).contiguous()

        output_dict = {"logits": scores}
        if answers is not None:
            assert answers.size(1) == scores.size(1)
            scores = scores + sent_mask[:, None, :] * -10000.0
            loss = F.cross_entropy(scores.reshape(-1, sent_num), answers.reshape(-1), ignore_index=-1,
                                   reduction='sum') / (batch * 1.0)

            if edge_index is not None:
                knowledge_loss = self.get_knowledge_loss(edge_scores_norm,
                                                         edge_index, edges)
                knowledge_loss = knowledge_loss * self.graph_loss
                output_dict["knowledge_loss"] = knowledge_loss
                loss += knowledge_loss
            else:
                output_dict["knowledge_loss"] = torch.Tensor([0])

            output_dict["loss"] = loss

            _, pred = scores.max(dim=-1)
            valid_num = torch.sum(answers != -1)
            acc = torch.sum(pred == answers).to(
                dtype=scores.dtype) / (valid_num * 1.0)
            output_dict["acc"] = acc
            output_dict["valid_num"] = valid_num

        return output_dict

    @staticmethod
    def get_knowledge_loss(edge_scores, edge_index, edges):
        batch, sent_num, _ = edge_scores.size()
        edge_index_mask = (edge_index == -1)
        edge_index[edge_index_mask] = 0
        edge_index = edge_index.unsqueeze(-1).expand(-1, -1, sent_num)
        selected_sents = edge_scores.gather(dim=1, index=edge_index)

        edge_mask = (edges == -1)
        edges[edge_mask] = 0
        selected_scores = selected_sents.gather(dim=2, index=edges)
        selected_scores[edge_mask] = 0.
        selected_scores = selected_scores.sum(dim=2)
        selected_scores[edge_index_mask] = 1.

        selected_scores = -((selected_scores + 1e-8).log())
        knowledge_loss = selected_scores.sum() / (batch * 1.0)
        return knowledge_loss


def split_doc_sen_que(hidden_state, sentence_spans, sep_cls=False):
    cls_h = hidden_state[:, 0]

    batch = hidden_state.size(0)
    h = hidden_state.size(-1)

    max_sent_len = (sentence_spans[:, :, 1] -
                    sentence_spans[:, :, 0] + 1).max().item()
    max_sent_num = sentence_spans.size(1)

    output = hidden_state.new_zeros((batch, max_sent_num, max_sent_len, h))
    index_output = hidden_state.new_zeros(
        (batch, max_sent_num, max_sent_len), dtype=torch.long)
    mask = hidden_state.new_ones(batch, max_sent_num, max_sent_len)
    sent_mask = hidden_state.new_ones(batch, max_sent_num)

    r"""
    This process can be implemented via torch.gather, and the `sentence_spans` should indicates the token index, not the section.
    As a result, the memory of sentence_spans can be enlarged `max_sent_len` times more. But the process can be accelerated with GPU.
    """

    for b in range(batch):
        for sent_id, sec in enumerate(sentence_spans[b]):
            start = sec[0].item()
            end = sec[1].item()
            if start == -1 and end == -1:
                break
            if start == 0 and sep_cls:
                if end == 0:
                    continue
                start += 1
            lens = end - start + 1
            output[b][sent_id][:lens] = hidden_state[b][start:(end + 1)]
            index_output[b][sent_id][:lens] = torch.arange(start, end + 1)
            mask[b][sent_id][:lens] = hidden_state.new_zeros(lens)
            sent_mask[b][sent_id] = 0

    index_output = index_output.reshape(batch, max_sent_num * max_sent_len)
    if sep_cls:
        return output, index_output, mask, sent_mask, cls_h
    return output, index_output, mask, sent_mask
