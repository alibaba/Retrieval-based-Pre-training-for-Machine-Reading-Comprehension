import copy
import torch
from torch import nn
from torch.nn import functional as F
from transformers import BertPreTrainedModel, BertModel, BertConfig
from transformers.modeling_bert import BertEncoder

from general_util.utils import AverageMeter
from modules import layers


class KnowledgeSSPConfig(BertConfig):
    added_configs = [
        'graph_loss', 'gat_layers', 'gat_intermediate_size', 'gat_attn_heads'
    ]

    def __init__(self, graph_loss=0.0, gat_layers=3, gat_intermediate_size=2048, gat_attn_heads=6, **kwargs):
        super().__init__(**kwargs)

        self.graph_loss = graph_loss
        self.gat_layers = gat_layers
        self.gat_intermediate_size = gat_intermediate_size
        self.gat_attn_heads = gat_attn_heads


class BertKnowledgePreTrainedModelForSentenceReOrder(BertPreTrainedModel):
    """
    Pre-training BERT backbone or together with LinearSelfAttn.
    Use representation of [CLS] as query to make it trained for downstream task.
    Add Graph Attention Network (GAT) for sentence reasoning modeling.
    """
    config_class = KnowledgeSSPConfig
    model_prefix = 'knowledge_sentence_re_order'

    def __init__(self, config: KnowledgeSSPConfig):
        super().__init__(config)
        print(f'The model {self.__class__.__name__} is loading...')

        self.bert = BertModel(config)

        self.cls_w = nn.Linear(config.hidden_size, config.hidden_size)

        gat_config = copy.deepcopy(self.config)
        gat_config.intermediate_size = self.config.gat_intermediate_size
        gat_config.num_hidden_layers = self.config.gat_layers
        gat_config.num_attention_heads = self.config.gat_attn_heads
        self.gat = BertEncoder(gat_config)
        self.graph_loss = self.config.graph_loss

        self.project1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.project2 = nn.Linear(config.hidden_size, config.hidden_size)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                sentence_spans=None, answers=None, edge_index=None, edges=None):
        sequence_output = self.bert(input_ids=input_ids,
                                    token_type_ids=token_type_ids,
                                    attention_mask=attention_mask)[0]

        # mask: 1 for masked value and 0 for true value
        hidden, mask, sent_mask, cls_h = split_doc_sen_que(sequence_output, sentence_spans, sep_cls=True)

        batch, sent_num, seq_len = mask.size()

        hidden = self.dropout(hidden)

        cls_h = self.cls_w(cls_h)  # [batch, h]
        alpha = torch.einsum('bh,bsth->bst', cls_h, hidden)
        alpha = (alpha + mask * -10000.0).softmax(dim=-1)
        hidden = torch.einsum('bst,bsth->bsh', alpha, hidden)

        sent_mask = self.bert.get_extended_attention_mask(1 - sent_mask, (batch, sent_num), self.device)
        graph_hidden, all_attentions = self.gat(hidden, attention_mask=sent_mask,
                                                head_mask=self.bert.get_head_mask(None, self.config.gat_layers),
                                                output_attentions=True)

        hidden = self.dropout(hidden)  # Version <= 1.2
        # hidden = hidden + self.dropout(graph_hidden)  # Version 1.3

        q_sentence_num = answers.size(1)

        query = self.project1(hidden[:, :q_sentence_num])
        key = self.project2(hidden)
        scores = query.bmm(key.transpose(1, 2)).contiguous()

        output_dict = {"logits": scores}
        if answers is not None:

            scores = scores + sent_mask.squeeze(1)
            loss = F.cross_entropy(scores.reshape(-1, sent_num), answers.reshape(-1), ignore_index=-1,
                                   reduction='sum') / (batch * 1.0)
            # print(loss)

            if edges is not None:
                # Currently only supervise specific head.
                # first_layer_attn_scores = all_attentions[0][:, 0]  # normalized (batch, sent_num, sent_num)
                # edge_index_mask = (edge_index == -1)
                # edge_index[edge_index_mask] = 0
                # edge_index = edge_index[:, :, None].expand(-1, -1, sent_num)
                # selected_sents = first_layer_attn_scores.gather(dim=1, index=edge_index)  # (batch, edge_index_num, sent_num)
                # assert selected_sents.size() == (batch, edge_index_mask.size(1), sent_num)

                # edge_mask = (edges == -1)
                # edges[edge_mask] = 0
                # selected_scores = selected_sents.gather(dim=-1, index=edges)  # (batch, edge_index_num, sent_num)
                # selected_scores[edge_mask] = 0.
                # selected_scores = selected_scores.sum(dim=-1)  # (batch, edge_index_num)
                # selected_scores[edge_index_mask] = 1.  # remove sentence with no edges

                # selected_scores = -(selected_scores + 1e-8).log()
                # knowledge_loss = self.graph_loss * selected_scores.sum() / (batch * 1.0)
                # # print(knowledge_loss)
                # loss += knowledge_loss
                # output_dict["knowledge_loss"] = knowledge_loss

                # Version 1.3: For the first layer of GAT, sum the loglikelihood amond all attention heads
                # to acclerate training
                first_layer_attn_scores = all_attentions[0]  # normalized (batch, head_num, sent_num, sent_num)
                head_num = first_layer_attn_scores.size(1)
                edge_index_mask = (edge_index == -1)
                edge_index[edge_index_mask] = 0
                edge_index = edge_index[:, None, :, None].expand(-1, head_num, -1, sent_num)
                selected_sents = first_layer_attn_scores.gather(dim=2,
                                                                index=edge_index)  # (batch, head_num, edge_index_num, sent_num)
                # assert selected_sents.size() == (batch, edge_index_mask.size(1), sent_num)

                edge_mask = (edges == -1)
                edges[edge_mask] = 0
                edges = edges.unsqueeze(1).expand(-1, head_num, -1, -1)
                selected_scores = selected_sents.gather(dim=-1,
                                                        index=edges)  # (batch, head_num, edge_index_num, sent_num)
                selected_scores[edge_mask.unsqueeze(1).expand(-1, head_num, -1, -1)] = 0.
                selected_scores = selected_scores.sum(dim=-1)  # (batch, head_num, edge_index_num)
                selected_scores[
                    edge_index_mask.unsqueeze(1).expand(-1, head_num, -1)] = 1.  # remove sentence with no edges

                selected_scores = -(selected_scores + 1e-8).log()
                knowledge_loss = self.graph_loss * selected_scores.sum() / (batch * 1.0)
                # print(knowledge_loss)
                loss += knowledge_loss
                output_dict["knowledge_loss"] = knowledge_loss

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
