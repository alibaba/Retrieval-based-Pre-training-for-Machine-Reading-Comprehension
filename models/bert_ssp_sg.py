import torch
from torch import nn
from torch.nn import functional as F
from transformers import BertPreTrainedModel, BertConfig

from data import MetricType
from modules.modeling_cls_bert import BertGraphModel


class SSPSGConfig(BertConfig):
    added_configs = [
        'graph_loss', 'cls_n_head', 'cls_d_head', 'graph_layers'
    ]

    def __init__(self, graph_loss=0.0, cls_n_head=12, cls_d_head=64, graph_layers=3, **kwargs):
        super().__init__(**kwargs)

        self.graph_loss = graph_loss
        self.cls_n_head = cls_n_head
        self.cls_d_head = cls_d_head
        self.graph_layers = graph_layers


class BertSSPSentenceGraph(BertPreTrainedModel):
    r"""
    Pre-trained Bert Model for self-supervised pretraining with extra
    sentence graph attention.
    """
    model_prefix = 'bert_ssp_sg'
    config_class = SSPSGConfig

    def __init__(self, config: SSPSGConfig):
        super().__init__(config)
        print(f'The model {self.__class__.__name__} is loading...')

        self.bert = BertGraphModel(config)

        self.project1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.project2 = nn.Linear(config.hidden_size, config.hidden_size)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.graph_loss = config.graph_loss

        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                sentence_spans=None, answers=None, edge_index=None, edges=None):

        cls_index = sentence_spans[:, :, 0]
        sent_mask = (cls_index == -1)
        batch, sent_num = sent_mask.size()
        cls_index[sent_mask] = 0
        cls_index = cls_index.unsqueeze(-1).expand(-1, -1, self.config.hidden_size)
        sent_mask = self.bert.get_extended_attention_mask(sent_mask, (batch, sent_num), self.device)

        outputs = self.bert(input_ids=input_ids,
                            token_type_ids=token_type_ids,
                            attention_mask=attention_mask,
                            output_attentions=True, cls_index=cls_index, cls_mask=sent_mask)
        hidden, _, attentions = outputs
        attentions = [t[1] for t in attentions[(self.config.num_hidden_layers - self.config.graph_layers):]]

        # mask: 1 for masked value and 0 for true value
        # hidden, mask, sent_mask, cls_h = split_doc_sen_que(sequence_output, sentence_spans,
        #    sep_cls=True)

        hidden = torch.gather(hidden, dim=1, index=cls_index)
        hidden = self.dropout(hidden)

        query_num = answers.size(1)

        query = self.project1(hidden[:, :query_num])
        key = self.project2(hidden)
        scores = query.bmm(key.transpose(1, 2))  # [batch, query_num, sent_num]

        output_dict = {"logits": scores}
        if answers is not None:
            assert answers.size(1) == scores.size(1)
            sent_mask = sent_mask.squeeze(1)
            scores = scores + sent_mask * -10000.0
            loss = F.cross_entropy(scores.reshape(-1, sent_num), answers.reshape(-1), ignore_index=-1,
                                   reduction='sum') / (batch * 1.0)

            if edges is not None:
                attentions = attentions[0]  # Only first layer is supervised
                attentions = attentions[:, 0]  # Only the first head is supervised
                edge_index_mask = (edge_index == -1)
                edge_index[edge_index_mask] = 0
                edge_index = edge_index.unsqueeze(-1).expand(-1, -1, sent_num)
                selected_sents = attentions.gather(dim=1, index=edge_index)

                edge_mask = (edges == -1)
                edges[edge_mask] = 0
                selected_scores = selected_sents.gather(dim=-1, index=edges)
                selected_scores[edge_mask] = 0.
                selected_scores = selected_scores.sum(dim=-1)
                selected_scores[edge_index_mask] = 1.

                selected_scores = -((selected_scores + 1e-8).log())
                knowledge_loss = self.graph_loss * selected_scores.sum() / (batch * 1.0)
                loss += knowledge_loss
                output_dict['knowledge_loss'] = knowledge_loss

            output_dict["loss"] = loss

            _, pred = scores.max(dim=-1)
            valid_num = torch.sum(answers != -1)
            acc = torch.sum(pred == answers).to(dtype=scores.dtype) / (valid_num * 1.0)
            output_dict["acc"] = acc
            output_dict["valid_num"] = valid_num

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
