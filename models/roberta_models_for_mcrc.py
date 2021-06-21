import torch
from torch import nn
from torch.nn import functional as F
from transformers import RobertaConfig, RobertaModel, BertPreTrainedModel, RobertaForMaskedLM, XLNetModel
from transformers.modeling_bert import BertPooler

from models.roberta_models import RobertaHAMCRCConfig, RobertaPreTrainedModel
from modules.layers import simple_circle_loss, sentence_sum, mul_sentence_sum, weighted_sum, MHAToken
from modules.modeling_reusable_graph_attention import SentenceReorderDecoder, SentenceSum, SentenceReorderHead, \
    SentenceReorderWithPosition


class RobertaModelMultiQueryForMCRC(BertPreTrainedModel):
    model_prefix = 'roberta_mq_mcrc'
    config_class = RobertaHAMCRCConfig
    base_model_prefix = 'roberta'

    def __init__(self, config: RobertaHAMCRCConfig):
        super().__init__(config)

        self.roberta = RobertaModel(config)

        self.cls_w = nn.Linear(config.hidden_size, config.hidden_size)
        self.que_w = nn.Linear(config.hidden_size, config.hidden_size)

        if config.cls_type == 1:
            self.pooler = nn.Sequential(
                nn.Linear(config.hidden_size * 2, config.hidden_size),
                nn.Tanh()
            )
        else:
            self.pooler = nn.Linear(config.hidden_size * 2, config.hidden_size)

        self.classifier = nn.Linear(config.hidden_size, 1)

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
        hidden = hidden * (1 - mask.unsqueeze(-1))

        q_op_hidden = hidden[:, :2].reshape(fb, 2 * seq_len, -1)
        q_op_word_mask = mask[:, :2].reshape(fb, 2 * seq_len)
        query_q = self.cls_w(cls_h)
        query_h, _ = weighted_sum(query_q, q_op_hidden, q_op_word_mask)

        sent_q = self.que_w(query_h)
        p_hidden = hidden[:, 2:].reshape(fb, -1, hidden.size(-1))
        p_word_mask = mask[:, 2:].reshape(fb, -1)
        sent_h, _ = weighted_sum(sent_q, p_hidden, p_word_mask)

        cls_input = torch.cat([sent_q, sent_h], dim=-1)
        logits = self.classifier(self.dropout(self.pooler(cls_input))).view(batch, num_choice)

        outputs = (logits,)
        if labels is not None:
            loss = F.cross_entropy(logits, labels, ignore_index=-1)
            outputs = (loss,) + outputs

            _, pred = logits.max(dim=-1)
            acc = torch.sum(pred == labels) / (1.0 * batch)

            outputs = outputs + (acc,)

        return outputs


class RobertaModelMultiQueryHiForMCRC(BertPreTrainedModel):
    model_prefix = 'roberta_mq_h_mcrc'
    config_class = RobertaHAMCRCConfig
    base_model_prefix = 'roberta'

    def __init__(self, config: RobertaHAMCRCConfig):
        super().__init__(config)

        self.roberta = RobertaModel(config)

        self.cls_w = nn.Linear(config.hidden_size, config.hidden_size)
        self.que_w = nn.Linear(config.hidden_size, config.hidden_size)

        if config.cls_type == 1:
            self.pooler = nn.Sequential(
                nn.Linear(config.hidden_size * 2, config.hidden_size),
                nn.Tanh()
            )
        else:
            self.pooler = nn.Linear(config.hidden_size * 2, config.hidden_size)

        self.classifier = nn.Linear(config.hidden_size, 1)

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
        hidden = hidden * (1 - mask.unsqueeze(-1))

        q_op_hidden = hidden[:, :2].reshape(fb, 2 * seq_len, -1)
        q_op_word_mask = mask[:, :2].reshape(fb, 2 * seq_len)
        query_q = self.cls_w(cls_h)
        query_h, _ = weighted_sum(query_q, q_op_hidden, q_op_word_mask)

        sent_q = self.que_w(query_h)
        p_hidden = hidden[:, 2:]
        p_word_mask = mask[:, 2:]

        sent_h, _ = sentence_sum(sent_q, p_hidden, p_word_mask)
        sent_h = sent_h * (1 - sent_mask[:, 2:].unsqueeze(-1))

        attended_h, _ = weighted_sum(sent_q, sent_h, sent_mask[:, 2:])

        cls_input = torch.cat([sent_q, attended_h], dim=-1)
        logits = self.classifier(self.dropout(self.pooler(cls_input))).view(batch, num_choice)

        outputs = (logits,)
        if labels is not None:
            loss = F.cross_entropy(logits, labels, ignore_index=-1)
            outputs = (loss,) + outputs

            _, pred = logits.max(dim=-1)
            acc = torch.sum(pred == labels) / (1.0 * batch)

            outputs = outputs + (acc,)

        return outputs


class RobertaModelMultiQueryHiProForMCRC(BertPreTrainedModel):
    model_prefix = 'roberta_mq_h_pro_mcrc'
    config_class = RobertaHAMCRCConfig
    base_model_prefix = 'roberta'

    def __init__(self, config: RobertaHAMCRCConfig):
        super().__init__(config)

        self.roberta = RobertaModel(config)

        self.cls_w = nn.Linear(config.hidden_size, config.hidden_size)
        self.que_w = nn.Linear(config.hidden_size, config.hidden_size)

        self.project1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.project2 = nn.Linear(config.hidden_size, config.hidden_size)

        if config.cls_type == 1:
            self.pooler = nn.Sequential(
                nn.Linear(config.hidden_size * 2, config.hidden_size),
                nn.Tanh()
            )
        else:
            self.pooler = nn.Linear(config.hidden_size * 2, config.hidden_size)

        self.classifier = nn.Linear(config.hidden_size, 1)

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
        hidden = hidden * (1 - mask.unsqueeze(-1))

        q_op_hidden = hidden[:, :2].reshape(fb, 2 * seq_len, -1)
        q_op_word_mask = mask[:, :2].reshape(fb, 2 * seq_len)
        query_q = self.cls_w(cls_h)
        query_h, _ = weighted_sum(query_q, q_op_hidden, q_op_word_mask)  # (fb, h)

        sent_q = self.que_w(query_h)
        p_hidden = hidden[:, 2:]
        p_word_mask = mask[:, 2:]

        sent_h, _ = sentence_sum(sent_q, p_hidden, p_word_mask)  # (fb, sent_num - 2, h)
        sent_h = sent_h * (1 - sent_mask[:, 2:].unsqueeze(-1))

        sent_q = self.project1(sent_q)
        sent_h = self.project2(sent_h)

        attended_h, _ = weighted_sum(sent_q, sent_h, sent_mask[:, 2:])  # (fb, h)

        cls_input = torch.cat([sent_q, attended_h], dim=-1)
        logits = self.classifier(self.dropout(self.pooler(cls_input))).view(batch, num_choice)

        outputs = (logits,)
        if labels is not None:
            loss = F.cross_entropy(logits, labels, ignore_index=-1)
            outputs = (loss,) + outputs

            _, pred = logits.max(dim=-1)
            acc = torch.sum(pred == labels) / (1.0 * batch)

            outputs = outputs + (acc,)

        return outputs


class RobertaModelMultiQueryHiBiForMCRC(BertPreTrainedModel):
    model_prefix = 'roberta_mq_h_bi_mcrc'
    config_class = RobertaHAMCRCConfig
    base_model_prefix = 'roberta'

    def __init__(self, config: RobertaHAMCRCConfig):
        super().__init__(config)

        self.roberta = RobertaModel(config)

        self.cls_w = nn.Linear(config.hidden_size, config.hidden_size)
        self.que_w = nn.Linear(config.hidden_size, config.hidden_size)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size)

        self.half_dim = config.hidden_size // 2

        if config.cls_type == 1:
            self.pooler = nn.Sequential(
                nn.Linear(config.hidden_size * 2, config.hidden_size),
                nn.Tanh()
            )
        else:
            self.pooler = nn.Linear(config.hidden_size * 2, config.hidden_size)

        self.classifier = nn.Linear(config.hidden_size, 1)

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
        hidden = hidden * (1 - mask.unsqueeze(-1))

        q_op_hidden = hidden[:, :2].reshape(fb, 2 * seq_len, -1)
        q_op_word_mask = mask[:, :2].reshape(fb, 2 * seq_len)
        query_q = self.cls_w(cls_h)
        query_h, _ = weighted_sum(query_q, q_op_hidden, q_op_word_mask)

        sent_q = self.que_w(query_h).view(fb, 2, self.half_dim)

        p_hidden = self.proj(hidden[:, 2:]).view(fb, sent_num - 2, seq_len, 2, self.half_dim)
        p_word_mask = mask[:, 2:]

        bi_scores = torch.einsum("bxh,bstxh->bstx", sent_q, p_hidden)
        bi_alpha = (bi_scores + p_word_mask.unsqueeze(-1) * -10000.0).softmax(dim=2)
        bi_sent_h = torch.einsum("bstx,bstxh->bsxh", bi_alpha, p_hidden)

        scores = torch.einsum("bxh,bsxh->bsx", sent_q, bi_sent_h)
        alpha = (scores + sent_mask[:, 2:].unsqueeze(-1) * -10000.0).softmax(dim=1)
        attended_h = torch.einsum("bsx,bsxh->bxh", alpha, bi_sent_h)

        sent_q = sent_q.view(fb, 2 * self.half_dim)
        attended_h = attended_h.view(fb, 2 * self.half_dim)

        cls_input = torch.cat([sent_q, attended_h], dim=-1)
        logits = self.classifier(self.dropout(self.pooler(cls_input))).view(batch, num_choice)

        outputs = (logits,)
        if labels is not None:
            loss = F.cross_entropy(logits, labels, ignore_index=-1)
            outputs = (loss,) + outputs

            _, pred = logits.max(dim=-1)
            acc = torch.sum(pred == labels) / (1.0 * batch)

            outputs = outputs + (acc,)

        return outputs


class RobertaModelMultiHeadQueryBiForMCRC(RobertaPreTrainedModel):
    model_prefix = 'roberta_mhq_bi_mcrc'
    config_class = RobertaHAMCRCConfig

    def __init__(self, config: RobertaHAMCRCConfig):
        super().__init__(config)

        self.roberta = RobertaModel(config)

        self.q_sum = MHAToken(config)
        self.s_sum = MHAToken(config)
        self.project1 = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.project2 = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.value = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

        self.half_dim = config.hidden_size // 2

        if config.cls_type == 1:
            self.pooler = nn.Sequential(
                nn.Linear(config.hidden_size * 2, config.hidden_size),
                nn.Tanh()
            )
        else:
            self.pooler = nn.Linear(config.hidden_size * 2, config.hidden_size)

        self.classifier = nn.Linear(config.hidden_size, 1)
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
        hidden = hidden * (1 - mask.unsqueeze(-1))

        q_op_hidden = hidden[:, :2].reshape(fb, 1, 2 * seq_len, -1)
        q_op_word_mask = mask[:, :2].reshape(fb, 1, 2 * seq_len)

        query_h = self.q_sum(cls_h.unsqueeze(1), q_op_hidden, q_op_word_mask)
        query_h = query_h.squeeze(1)
        assert query_h.size(1) == 1
        
        sent_h = self.s_sum(query_h, hidden[:, 2:], mask[:, 2:])

        query = self.project1(query_h).view(fb, 2, self.half_dim)
        key_h = self.project2(sent_h).view(fb, sent_num - 2, 2, self.half_dim)
        value_h = self.value(sent_h).view(fb, sent_num - 2, 2, self.half_dim)
        # value_h = key_h

        scores = torch.einsum("bxh,bsxh->bxs", query, key_h)
        alpha = (scores + sent_mask[:, 2:].unsqueeze(1) * -10000.0).softmax(dim=-1)
        attended_h = torch.einsum("bxs,bsxh->bxh", alpha, value_h).view(fb, 2 * self.half_dim)

        cls_input = torch.cat([query_h.squeeze(1), attended_h], dim=-1)
        logits = self.classifier(self.dropout(self.pooler(cls_input))).view(batch, num_choice)

        outputs = (logits,)
        if labels is not None:
            loss = F.cross_entropy(logits, labels, ignore_index=-1)
            outputs = (loss,) + outputs

            _, pred = logits.max(dim=-1)
            acc = torch.sum(pred == labels) / (1.0 * batch)

            outputs = outputs + (acc,)

        return outputs


class RobertaModelMultiHeadQueryGlobalForMCRC(RobertaPreTrainedModel):
    model_prefix = 'roberta_mhq_global_mcrc'
    config_class = RobertaHAMCRCConfig

    def __init__(self, config: RobertaHAMCRCConfig):
        super().__init__(config)

        self.roberta = RobertaModel(config)

        self.q_sum = MHAToken(config)
        self.s_sum = MHAToken(config)
        self.project1 = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.project2 = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.value = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

        self.residual = not config.no_residual

        self.half_dim = config.hidden_size // 2

        if config.cls_type == 1:
            self.pooler = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.Tanh()
            )
        else:
            self.pooler = nn.Linear(config.hidden_size, config.hidden_size)

        self.classifier = nn.Linear(config.hidden_size, 1)
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
        hidden = hidden * (1 - mask.unsqueeze(-1))

        q_op_hidden = hidden[:, :2].reshape(fb, 1, 2 * seq_len, -1)
        q_op_word_mask = mask[:, :2].reshape(fb, 1, 2 * seq_len)

        query_h = self.q_sum(cls_h.unsqueeze(1), q_op_hidden, q_op_word_mask, residual=self.residual)
        query_h = query_h.squeeze(1)
        assert query_h.size(1) == 1
        
        sent_h = self.s_sum(query_h, hidden, mask, residual=self.residual)

        query = self.project1(query_h).view(fb, 2, self.half_dim)
        key_h = self.project2(sent_h).view(fb, sent_num, 2, self.half_dim)
        value_h = self.value(sent_h).view(fb, sent_num, 2, self.half_dim)
        # value_h = key_h

        scores = torch.einsum("bxh,bsxh->bxs", query, key_h)
        alpha = (scores + sent_mask.unsqueeze(1) * -10000.0).softmax(dim=-1)
        attended_h = torch.einsum("bxs,bsxh->bxh", alpha, value_h).view(fb, 2 * self.half_dim)

        # cls_input = torch.cat([query_h.squeeze(1), attended_h], dim=-1)
        logits = self.classifier(self.dropout(self.pooler(attended_h))).view(batch, num_choice)

        outputs = (logits,)
        if labels is not None:
            loss = F.cross_entropy(logits, labels, ignore_index=-1)
            outputs = (loss,) + outputs

            _, pred = logits.max(dim=-1)
            acc = torch.sum(pred == labels) / (1.0 * batch)

            outputs = outputs + (acc,)

        return outputs


class RobertaModelMultiHeadQueryBiExForMCRC(RobertaPreTrainedModel):
    model_prefix = 'roberta_mhq_bi_ex_mcrc'
    config_class = RobertaHAMCRCConfig

    def __init__(self, config: RobertaHAMCRCConfig):
        super().__init__(config)

        self.roberta = RobertaModel(config)

        self.q_sum = MHAToken(config)
        self.s_sum = MHAToken(config)
        self.project1 = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.project2 = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

        self.v_sum = nn.Linear(config.hidden_size, 1)
        self.value = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

        self.half_dim = config.hidden_size // 2

        if config.cls_type == 1:
            self.pooler = nn.Sequential(
                nn.Linear(config.hidden_size * 2, config.hidden_size),
                nn.Tanh()
            )
        else:
            self.pooler = nn.Linear(config.hidden_size * 2, config.hidden_size)

        self.classifier = nn.Linear(config.hidden_size, 1)
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
        hidden = hidden * (1 - mask.unsqueeze(-1))

        q_op_hidden = hidden[:, :2].reshape(fb, 1, 2 * seq_len, -1)
        q_op_word_mask = mask[:, :2].reshape(fb, 1, 2 * seq_len)

        query_h = self.q_sum(cls_h.unsqueeze(1), q_op_hidden, q_op_word_mask)
        query_h = query_h.squeeze(1)
        assert query_h.size(1) == 1
        
        sent_h = self.s_sum(query_h, hidden[:, 2:], mask[:, 2:])

        query = self.project1(query_h).view(fb, 2, self.half_dim)
        key_h = self.project2(sent_h).view(fb, sent_num - 2, 2, self.half_dim)
        # value_h = self.value(sent_h).view(fb, sent_num - 2, 2, self.half_dim)
        # value_h = key_h
        value_alpha = self.v_sum(hidden[:, 2:]).squeeze(-1) + mask[:, 2:] * -10000.0
        value_alpha = torch.softmax(value_alpha, dim=-1)
        value_h = torch.einsum("bst,bsth->bsh", value_alpha, hidden[:, 2:])
        value_h = self.value(value_h).view(fb, sent_num - 2, 2, self.half_dim)

        scores = torch.einsum("bxh,bsxh->bxs", query, key_h)
        alpha = (scores + sent_mask[:, 2:].unsqueeze(1) * -10000.0).softmax(dim=-1)
        attended_h = torch.einsum("bxs,bsxh->bxh", alpha, value_h).view(fb, 2 * self.half_dim)
        # attended_h = torch.einsum("bxs,bsh->bxh", alpha, value_h).view(fb, -1)

        cls_input = torch.cat([query_h.squeeze(1), attended_h], dim=-1)
        logits = self.classifier(self.dropout(self.pooler(cls_input))).view(batch, num_choice)

        outputs = (logits,)
        if labels is not None:
            loss = F.cross_entropy(logits, labels, ignore_index=-1)
            outputs = (loss,) + outputs

            _, pred = logits.max(dim=-1)
            acc = torch.sum(pred == labels) / (1.0 * batch)

            outputs = outputs + (acc,)

        return outputs


class RobertaModelMultiHeadQueryBiExGlobalForMCRC(RobertaPreTrainedModel):
    model_prefix = 'roberta_mhq_bi_exg_mcrc'
    config_class = RobertaHAMCRCConfig

    def __init__(self, config: RobertaHAMCRCConfig):
        super().__init__(config)

        self.roberta = RobertaModel(config)

        self.q_sum = MHAToken(config)
        self.s_sum = MHAToken(config)
        self.project1 = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.project2 = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

        self.v_sum = nn.Linear(config.hidden_size, 1)
        self.value = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

        self.half_dim = config.hidden_size // 2

        if config.cls_type == 1:
            self.pooler = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.Tanh()
            )
        else:
            self.pooler = nn.Linear(config.hidden_size, config.hidden_size)

        self.classifier = nn.Linear(config.hidden_size, 1)
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
        hidden = hidden * (1 - mask.unsqueeze(-1))

        q_op_hidden = hidden[:, :2].reshape(fb, 1, 2 * seq_len, -1)
        q_op_word_mask = mask[:, :2].reshape(fb, 1, 2 * seq_len)

        query_h = self.q_sum(cls_h.unsqueeze(1), q_op_hidden, q_op_word_mask)
        query_h = query_h.squeeze(1)
        assert query_h.size(1) == 1
        
        sent_h = self.s_sum(query_h, hidden, mask)

        query = self.project1(query_h).view(fb, 2, self.half_dim)
        key_h = self.project2(sent_h).view(fb, sent_num, 2, self.half_dim)
        
        value_alpha = self.v_sum(hidden).squeeze(-1) + mask * -10000.0
        value_alpha = torch.softmax(value_alpha, dim=-1)
        value_h = torch.einsum("bst,bsth->bsh", value_alpha, hidden)
        value_h = self.value(value_h).view(fb, sent_num, 2, self.half_dim)

        scores = torch.einsum("bxh,bsxh->bxs", query, key_h)
        alpha = (scores + sent_mask.unsqueeze(1) * -10000.0).softmax(dim=-1)
        attended_h = torch.einsum("bxs,bsxh->bxh", alpha, value_h).view(fb, 2 * self.half_dim)

        logits = self.classifier(self.dropout(self.pooler(attended_h))).view(batch, num_choice)

        outputs = (logits,)
        if labels is not None:
            loss = F.cross_entropy(logits, labels, ignore_index=-1)
            outputs = (loss,) + outputs

            _, pred = logits.max(dim=-1)
            acc = torch.sum(pred == labels) / (1.0 * batch)

            outputs = outputs + (acc,)

        return outputs


class RobertaModelMultiHeadQueryBiExRForMCRC(RobertaPreTrainedModel):
    model_prefix = 'roberta_mhq_bi_exr_mcrc'
    config_class = RobertaHAMCRCConfig

    def __init__(self, config: RobertaHAMCRCConfig):
        super().__init__(config)

        self.roberta = RobertaModel(config)

        self.q_sum = MHAToken(config)
        self.s_sum = MHAToken(config)
        self.project1 = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.project2 = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

        self.value = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

        self.half_dim = config.hidden_size // 2

        if config.cls_type == 1:
            self.pooler = nn.Sequential(
                nn.Linear(config.hidden_size * 2, config.hidden_size),
                nn.Tanh()
            )
        else:
            self.pooler = nn.Linear(config.hidden_size * 2, config.hidden_size)

        self.classifier = nn.Linear(config.hidden_size, 1)
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
        hidden = hidden * (1 - mask.unsqueeze(-1))

        q_op_hidden = hidden[:, :2].reshape(fb, 1, 2 * seq_len, -1)
        q_op_word_mask = mask[:, :2].reshape(fb, 1, 2 * seq_len)

        query_h = self.q_sum(cls_h.unsqueeze(1), q_op_hidden, q_op_word_mask)
        query_h = query_h.squeeze(1)
        assert query_h.size(1) == 1
        
        sent_h = self.s_sum(query_h, hidden[:, 2:], mask[:, 2:])

        query = self.project1(query_h).view(fb, 2, self.half_dim)
        key_h = self.project2(sent_h).view(fb, sent_num - 2, 2, self.half_dim)

        value_h = self.q_sum(cls_h.unsqueeze(1), hidden[:, 2:], mask[:, 2:]).squeeze(1)
        value_h = self.value(value_h).view(fb, sent_num - 2, 2, self.half_dim)

        scores = torch.einsum("bxh,bsxh->bxs", query, key_h)
        alpha = (scores + sent_mask[:, 2:].unsqueeze(1) * -10000.0).softmax(dim=-1)
        attended_h = torch.einsum("bxs,bsxh->bxh", alpha, value_h).view(fb, 2 * self.half_dim)

        cls_input = torch.cat([query_h.squeeze(1), attended_h], dim=-1)
        logits = self.classifier(self.dropout(self.pooler(cls_input))).view(batch, num_choice)

        outputs = (logits,)
        if labels is not None:
            loss = F.cross_entropy(logits, labels, ignore_index=-1)
            outputs = (loss,) + outputs

            _, pred = logits.max(dim=-1)
            acc = torch.sum(pred == labels) / (1.0 * batch)

            outputs = outputs + (acc,)

        return outputs


class RobertaModelMHQRawForMCRC(RobertaPreTrainedModel):
    model_prefix = 'roberta_mhq_raw_mcrc'
    config_class = RobertaHAMCRCConfig

    def __init__(self, config: RobertaHAMCRCConfig):
        super().__init__(config)

        self.roberta = RobertaModel(config)

        self.q_sum = MHAToken(config)
        self.s_sum = MHAToken(config)
        
        self.project1 = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.project2 = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

        if config.cls_type == 1:
            self.pooler = nn.Sequential(
                nn.Linear(config.hidden_size * 2, config.hidden_size),
                nn.Tanh()
            )
        else:
            self.pooler = nn.Linear(config.hidden_size * 2, config.hidden_size)

        self.classifier = nn.Linear(config.hidden_size, 1)
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
        hidden = hidden * (1 - mask.unsqueeze(-1))

        q_op_hidden = hidden[:, :2].reshape(fb, 1, 2 * seq_len, -1)
        q_op_word_mask = mask[:, :2].reshape(fb, 1, 2 * seq_len)

        query_h = self.q_sum(cls_h.unsqueeze(1), q_op_hidden, q_op_word_mask)
        query_h = query_h.squeeze(1)
        # assert query_h.size(1) == 1
        
        sent_h = self.s_sum(query_h, hidden[:, 2:], mask[:, 2:])

        query = self.project1(query_h).view(fb, -1)
        key_h = self.project2(sent_h).view(fb, sent_num - 2, -1)
        # value_h = self.value(sent_h).view(fb, sent_num - 2, 2, self.half_dim)
        value_h = key_h

        scores = torch.einsum("bh,bsh->bs", query, key_h)
        alpha = (scores + sent_mask[:, 2:] * -10000.0).softmax(dim=-1)
        attended_h = torch.einsum("bs,bsh->bh", alpha, value_h)

        cls_input = torch.cat([query_h.squeeze(1), attended_h], dim=-1)
        logits = self.classifier(self.dropout(self.pooler(cls_input))).view(batch, num_choice)

        outputs = (logits,)
        if labels is not None:
            loss = F.cross_entropy(logits, labels, ignore_index=-1)
            outputs = (loss,) + outputs

            _, pred = logits.max(dim=-1)
            acc = torch.sum(pred == labels) / (1.0 * batch)

            outputs = outputs + (acc,)

        return outputs


class RobertaModelMHQSumForMCRC(RobertaPreTrainedModel):
    model_prefix = 'roberta_mhq_sum_mcrc'
    config_class = RobertaHAMCRCConfig

    def __init__(self, config: RobertaHAMCRCConfig):
        super().__init__(config)

        self.roberta = RobertaModel(config)

        self.q_sum = MHAToken(config)
        self.s_sum = MHAToken(config)
        
        self.use_new_pro = config.use_new_pro
        if self.use_new_pro:
            self.que_pro = nn.Linear(config.hidden_size, config.hidden_size)
            self.key_pro = nn.Linear(config.hidden_size, config.hidden_size)
            # self.val_pro = nn.Linear(config.hidden_size, config.hidden_size)
        else:
            self.project1 = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
            self.project2 = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

        self.v_sum = nn.Linear(config.hidden_size, 1)

        if config.cls_type == 1:
            self.pooler = nn.Sequential(
                nn.Linear(config.hidden_size * 2, config.hidden_size),
                nn.Tanh()
            )
        else:
            self.pooler = nn.Linear(config.hidden_size * 2, config.hidden_size)

        self.classifier = nn.Linear(config.hidden_size, 1)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.init_weights()

    @staticmethod
    def fold_tensor(x):
        if x is None:
            return None
        return x.reshape(x.size(0) * x.size(1), *x.size()[2:])

    def project(self, _query, _key):
        if self.use_new_pro:
            return self.que_pro(_query), self.key_pro(_key)
        else:
            return self.project1(_query), self.project2(_key)

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
        hidden = hidden * (1 - mask.unsqueeze(-1))

        q_op_hidden = hidden[:, :2].reshape(fb, 1, 2 * seq_len, -1)
        q_op_word_mask = mask[:, :2].reshape(fb, 1, 2 * seq_len)

        query_h = self.q_sum(cls_h.unsqueeze(1), q_op_hidden, q_op_word_mask)
        query_h = query_h.squeeze(1)
        # assert query_h.size(1) == 1
        
        sent_h = self.s_sum(query_h, hidden[:, 2:], mask[:, 2:])

        query, key_h = self.project(query_h, sent_h)
        # query = self.project1(query_h).view(fb, -1)
        # key_h = self.project2(sent_h).view(fb, sent_num - 2, -1)
        query = query.view(fb, -1)
        key_h = key_h.view(fb, sent_num - 2, -1)

        v_scores = self.v_sum(hidden[:, 2:]).squeeze(-1)
        v_alpha = torch.softmax(v_scores + mask[:, 2:] * -10000.0, dim=-1)
        # value_h = self.value(sent_h).view(fb, sent_num - 2, 2, self.half_dim)
        value_h = torch.einsum("bst,bsth->bsh", v_alpha, hidden[:, 2:])

        scores = torch.einsum("bh,bsh->bs", query, key_h)
        alpha = (scores + sent_mask[:, 2:] * -10000.0).softmax(dim=-1)
        attended_h = torch.einsum("bs,bsh->bh", alpha, value_h)

        cls_input = torch.cat([query_h.squeeze(1), attended_h], dim=-1)
        logits = self.classifier(self.dropout(self.pooler(cls_input))).view(batch, num_choice)

        outputs = (logits,)
        if labels is not None:
            loss = F.cross_entropy(logits, labels, ignore_index=-1)
            outputs = (loss,) + outputs

            _, pred = logits.max(dim=-1)
            acc = torch.sum(pred == labels) / (1.0 * batch)

            outputs = outputs + (acc,)

        return outputs


roberta_models_for_mcrc_map = {
    RobertaModelMultiQueryForMCRC.model_prefix: RobertaModelMultiQueryForMCRC,
    RobertaModelMultiQueryHiForMCRC.model_prefix: RobertaModelMultiQueryHiForMCRC,
    RobertaModelMultiQueryHiProForMCRC.model_prefix: RobertaModelMultiQueryHiProForMCRC,
    RobertaModelMultiQueryHiBiForMCRC.model_prefix: RobertaModelMultiQueryHiBiForMCRC,

    RobertaModelMultiHeadQueryBiForMCRC.model_prefix: RobertaModelMultiHeadQueryBiForMCRC,
    RobertaModelMultiHeadQueryGlobalForMCRC.model_prefix: RobertaModelMultiHeadQueryGlobalForMCRC,
    RobertaModelMultiHeadQueryBiExForMCRC.model_prefix: RobertaModelMultiHeadQueryBiExForMCRC,
    RobertaModelMultiHeadQueryBiExRForMCRC.model_prefix: RobertaModelMultiHeadQueryBiExRForMCRC,
    RobertaModelMultiHeadQueryBiExGlobalForMCRC.model_prefix: RobertaModelMultiHeadQueryBiExGlobalForMCRC,

    RobertaModelMHQRawForMCRC.model_prefix: RobertaModelMHQRawForMCRC,
    RobertaModelMHQSumForMCRC.model_prefix: RobertaModelMHQSumForMCRC
}
