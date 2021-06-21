import torch
from torch import nn
from torch.nn import functional as F
from transformers import BertPreTrainedModel, BertModel, BertConfig

from data import MetricType
from modules.modeling_graph_attention import BertGraphAttention
from modules.modeling_reusable_graph_attention import BertReusableGraphAttention


class BertSSPKGConfig(BertConfig):
    added_configs = [
        'kg_layer_ids', 'cls_n_head', 'cls_d_head', 'edge_type'
    ]

    def __init__(self, kg_layer_ids='6', cls_n_head=6, cls_d_head=128,
                 edge_type='knowledge', **kwargs):
        super().__init__(**kwargs)

        self.kg_layer_ids = kg_layer_ids
        self.cls_d_head = cls_d_head
        self.cls_n_head = cls_n_head
        self.edge_type = edge_type


class BertSSPKG(BertPreTrainedModel):
    r"""
    Pre-training BERT backbone or together with LinearSelfAttn.
    Use representation of [CLS] as query to make it trained for downstream task.

    Add ConceptNet based Knowledge Graph to incorporate ralation information
    into reasoning.
    """
    config_class = BertSSPKGConfig
    model_prefix = 'ssp_cls_kg'

    def __init__(self, config: BertSSPKGConfig):
        super().__init__(config)
        print(f'The model {self.__class__.__name__} is loading...')

        self.bert = BertModel(config)
        self.gat_layer = BertGraphAttention(config)
        self.bert.encoder.layer[config.kg_layer_id].attention.self = self.gat_layer

        self.cls_w = nn.Linear(config.hidden_size, config.hidden_size)

        self.project1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.project2 = nn.Linear(config.hidden_size, config.hidden_size)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                sentence_spans=None, answers=None, edge_index=None, edges=None):
        sentence_index, mask, sent_mask = get_sentence_index(sentence_spans)
        sentence_index = sentence_index.unsqueeze(-1).expand(-1, -1, -1, self.config.hidden_size)
        self.gat_layer.graph_attention.set_edges(edge_index, edges, sentence_index,
                                                 mask, sent_mask)

        sequence_output = self.bert(input_ids=input_ids,
                                    token_type_ids=token_type_ids,
                                    attention_mask=attention_mask)[0]

        # mask: 1 for masked value and 0 for true value
        batch, sent_num, seq_len = mask.size()

        cls_h = sequence_output[:, 0]
        hidden = sequence_output.gather(dim=1, index=sentence_index.reshape(
            batch, sent_num * seq_len, -1))
        hidden = self.dropout(hidden.reshape(batch, sent_num, seq_len, -1))

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


class BertSSPKGRS(BertPreTrainedModel):
    r"""
    Pre-training BERT backbone or together with LinearSelfAttn.
    Use representation of [CLS] as query to make it trained for downstream task.

    Add ConceptNet based Knowledge Graph to incorporate ralation information
    into reasoning.

    RS: Reuse the attention from self-attention matrix
    """
    config_class = BertSSPKGConfig
    model_prefix = 'ssp_cls_kg_rs'

    def __init__(self, config: BertSSPKGConfig):
        super().__init__(config)
        print(f'The model {self.__class__.__name__} is loading...')

        self.bert = BertModel(config)
        self.kg_layer_ids = [int(idx) for idx in config.kg_layer_ids.split(',')]
        for idx in self.kg_layer_ids:
            self.bert.encoder.layer[idx].attention.self = BertReusableGraphAttention(config)
        # self.gat_layer = self.bert.encoder.layer[config.kg_layer_id].attention.self

        self.cls_w = nn.Linear(config.hidden_size, config.hidden_size)

        self.project1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.project2 = nn.Linear(config.hidden_size, config.hidden_size)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                sentence_spans=None, answers=None, edge_index=None, edges=None):

        sentence_index, mask, sent_mask = get_sentence_index(sentence_spans)
        for kg_id in self.kg_layer_ids:
            self.bert.encoder.layer[kg_id].attention.self.graph_attention.set_edges(
                sentence_index, mask, sent_mask)

        sequence_output = self.bert(input_ids=input_ids,
                                    token_type_ids=token_type_ids,
                                    attention_mask=attention_mask)[0]

        # mask: 1 for masked value and 0 for true value
        batch, sent_num, seq_len = mask.size()

        sentence_index = sentence_index.unsqueeze(-1).expand(-1, -1, -1, self.config.hidden_size)

        cls_h = sequence_output[:, 0]
        hidden = sequence_output.gather(dim=1, index=sentence_index.reshape(
            batch, sent_num * seq_len, -1))
        hidden = self.dropout(hidden.reshape(batch, sent_num, seq_len, -1))

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


class BertSSPKGMCRC(BertPreTrainedModel):
    r"""
    Pre-training BERT backbone or together with LinearSelfAttn.
    Use representation of [CLS] as query to make it trained for downstream task.

    Add ConceptNet based Knowledge Graph to incorporate ralation information
    into reasoning.
    """
    config_class = BertSSPKGConfig
    model_prefix = 'ssp_cls_kg_mcrc'

    def __init__(self, config: BertSSPKGConfig):
        super().__init__(config)
        print(f'The model {self.__class__.__name__} is loading...')

        self.bert = BertModel(config)
        self.gat_layer = BertGraphAttention(config)
        self.bert.encoder.layer[config.kg_layer_id].attention.self = self.gat_layer

        self.classifier = nn.Linear(config.hidden_size, 1)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.init_weights()

    @staticmethod
    def fold_tensor(x):
        if x is None:
            return None
        return x.reshape(x.size(0) * x.size(1), *x.size()[2:])

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                sentence_spans=None, labels=None, edge_index=None, edges=None):

        sentence_spans = self.fold_tensor(sentence_spans)
        edge_index = self.fold_tensor(edge_index)
        edges = self.fold_tensor(edges)

        sentence_index, mask, sent_mask = get_sentence_index(sentence_spans)
        sentence_index = sentence_index.unsqueeze(-1).expand(-1, -1, -1, self.config.hidden_size)
        self.gat_layer.graph_attention.set_edges(edge_index, edges, sentence_index,
                                                 mask, sent_mask)

        batch, num_choice, _ = input_ids.size()
        input_ids = self.fold_tensor(input_ids)
        token_type_ids = self.fold_tensor(token_type_ids)
        attention_mask = self.fold_tensor(attention_mask)

        sequence_output = self.bert(input_ids=input_ids,
                                    token_type_ids=token_type_ids,
                                    attention_mask=attention_mask)[0]

        cls_h = sequence_output[:, 0].contiguous()
        logits = self.classifier(self.dropout(cls_h)).view(batch, num_choice)

        # output_dict = {"logits": logits}
        outputs = (logits,)
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            # output_dict["loss"] = loss
            outputs = (loss,) + outputs

            _, pred = logits.max(dim=-1)
            acc = torch.sum(pred == labels) / (1.0 * batch)

            # output_dict["acc"] = acc
            outputs = outputs + (acc,)

        return outputs


class BertSSPKGRSMCRC(BertPreTrainedModel):
    r"""
    Pre-training BERT backbone or together with LinearSelfAttn.
    Use representation of [CLS] as query to make it trained for downstream task.

    Add ConceptNet based Knowledge Graph to incorporate ralation information
    into reasoning.
    """
    config_class = BertSSPKGConfig
    model_prefix = 'ssp_cls_kg_rs_mcrc'

    def __init__(self, config: BertSSPKGConfig):
        super().__init__(config)
        print(f'The model {self.__class__.__name__} is loading...')

        self.bert = BertModel(config)
        self.kg_layer_ids = [int(idx) for idx in config.kg_layer_ids.split(',')]
        for idx in self.kg_layer_ids:
            self.bert.encoder.layer[idx].attention.self = BertReusableGraphAttention(config)

        self.classifier = nn.Linear(config.hidden_size, 1)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.init_weights()

    @staticmethod
    def fold_tensor(x):
        if x is None:
            return None
        return x.reshape(x.size(0) * x.size(1), *x.size()[2:])

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                sentence_spans=None, labels=None, edge_index=None, edges=None):

        sentence_spans = self.fold_tensor(sentence_spans)

        sentence_index, mask, sent_mask = get_sentence_index(sentence_spans)
        for kg_id in self.kg_layer_ids:
            self.bert.encoder.layer[kg_id].attention.self.graph_attention.set_edges(
                sentence_index, mask, sent_mask)

        batch, num_choice, _ = input_ids.size()
        input_ids = self.fold_tensor(input_ids)
        token_type_ids = self.fold_tensor(token_type_ids)
        attention_mask = self.fold_tensor(attention_mask)

        sequence_output = self.bert(input_ids=input_ids,
                                    token_type_ids=token_type_ids,
                                    attention_mask=attention_mask)[0]

        cls_h = sequence_output[:, 0].contiguous()
        logits = self.classifier(self.dropout(cls_h)).view(batch, num_choice)

        # output_dict = {"logits": logits}
        outputs = (logits,)
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            # output_dict["loss"] = loss
            outputs = (loss,) + outputs

            _, pred = logits.max(dim=-1)
            acc = torch.sum(pred == labels) / (1.0 * batch)

            # output_dict["acc"] = acc
            outputs = outputs + (acc,)

        return outputs


def get_sentence_index(sentence_spans):
    max_sent_len = (sentence_spans[:, :, 1] - sentence_spans[:, :, 0] + 1).max().item()
    b, max_sent_num, _ = sentence_spans.size()

    sentence_index = sentence_spans.new_zeros(b, max_sent_num, max_sent_len)
    mask = sentence_spans.new_ones(b, max_sent_num, max_sent_len)
    sent_mask = sentence_spans.new_ones(b, max_sent_num)
    for i in range(b):
        for sent_id, sec in enumerate(sentence_spans[i]):
            start = sec[0].item()
            end = sec[1].item()
            if start == -1 and end == -1:
                break
            if start == 0:
                if end == 0:
                    continue
                else:
                    start += 1
            lens = end - start + 1
            sentence_index[i, sent_id, :lens] = torch.arange(start, end + 1)
            mask[i, sent_id, :lens] = torch.zeros(lens)
            sent_mask[i, sent_id] = 0
    return sentence_index, mask, sent_mask
