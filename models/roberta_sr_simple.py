import torch
from torch import nn
from torch.nn import functional as F
from transformers import RobertaConfig, RobertaModel
from transformers.modeling_roberta import RobertaPreTrainedModel

from models.roberta_models import RobertaHAMCRCConfig
from modules import layers
from modules.layers import simple_circle_loss, sentence_sum, mul_sentence_sum, MHAToken, fix_circle_loss, \
    cross_entropy_loss_and_accuracy, mask_scores_with_labels
from modules.modeling_reusable_graph_attention import SentenceReorderDecoder, SentenceSum, SentenceReorderHead, \
    SentenceReorderWithPosition


class RobertaModelHiForMCRC(RobertaPreTrainedModel):
    model_prefix = 'roberta_h_mcrc'
    config_class = RobertaHAMCRCConfig

    def __init__(self, config: RobertaHAMCRCConfig):
        super().__init__(config)

        self.roberta = RobertaModel(config)

        self.q_sum = nn.Linear(config.hidden_size, config.hidden_size)
        self.s_sum = nn.Linear(config.hidden_size, config.hidden_size)

        self.use_new_pro = config.use_new_pro
        if self.use_new_pro:
            self.que_pro = nn.Linear(config.hidden_size, config.hidden_size)
            self.key_pro = nn.Linear(config.hidden_size, config.hidden_size)
            # self.val_pro = nn.Linear(config.hidden_size, config.hidden_size)
        else:
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

        query_h, _ = sentence_sum(
            self.q_sum(cls_h), q_op_hidden, q_op_word_mask
        )
        query_h = query_h.squeeze(1)

        p_hidden = hidden[:, 2:]
        p_word_mask = mask[:, 2:]

        sent_h, _ = sentence_sum(
            self.s_sum(query_h), p_hidden, p_word_mask
        )

        _pro_query, _pro_key = self.project(query_h, sent_h)

        scores = torch.einsum(
            "bh,bsh->bs",
            _pro_query,
            _pro_key
        )
        alpha = (scores + sent_mask[:, 2:] * -10000.0).softmax(dim=-1)

        attended_h = torch.einsum("bs,bsh->bh", alpha, sent_h)

        cls_input = torch.cat([query_h, attended_h], dim=-1)
        logits = self.classifier(self.dropout(self.pooler(cls_input))).view(batch, num_choice)

        outputs = (logits,)
        if labels is not None:
            loss = F.cross_entropy(logits, labels, ignore_index=-1)
            outputs = (loss,) + outputs

            _, pred = logits.max(dim=-1)
            acc = torch.sum(pred == labels) / (1.0 * batch)

            outputs = outputs + (acc,)

        return outputs


class RobertaModelHiGForMCRC(RobertaPreTrainedModel):
    model_prefix = 'roberta_g_mcrc'
    config_class = RobertaHAMCRCConfig

    def __init__(self, config: RobertaHAMCRCConfig):
        super().__init__(config)

        self.roberta = RobertaModel(config)

        self.q_sum = nn.Linear(config.hidden_size, config.hidden_size)
        self.s_sum = nn.Linear(config.hidden_size, config.hidden_size)

        self.project1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.project2 = nn.Linear(config.hidden_size, config.hidden_size)

        self.v_sum = nn.Linear(config.hidden_size, 1)

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

        query_h, _ = sentence_sum(
            self.q_sum(cls_h), q_op_hidden, q_op_word_mask
        )
        query_h = query_h.squeeze(1)

        sent_h, _ = sentence_sum(
            self.s_sum(query_h), hidden, mask
        )

        scores = torch.einsum(
            "bh,bsh->bs",
            self.project1(query_h),
            self.project2(sent_h)
        )
        alpha = (scores + sent_mask * -10000.0).softmax(dim=-1)

        v_scores = self.v_sum(hidden).squeeze(-1)
        v_alpha = torch.softmax(v_scores + mask * -10000.0, dim=-1)
        v_sent_h = torch.einsum("bst,bsth->bsh", v_alpha, hidden)

        attended_h = torch.einsum("bs,bsh->bh", alpha, v_sent_h)

        # cls_input = torch.cat([query_h, attended_h], dim=-1)
        logits = self.classifier(self.dropout(self.pooler(attended_h))).view(batch, num_choice)

        outputs = (logits,)
        if labels is not None:
            loss = F.cross_entropy(logits, labels, ignore_index=-1)
            outputs = (loss,) + outputs

            _, pred = logits.max(dim=-1)
            acc = torch.sum(pred == labels) / (1.0 * batch)

            outputs = outputs + (acc,)

        return outputs


roberta_models_for_mcrc_simple_map = {
    RobertaModelHiForMCRC.model_prefix: RobertaModelHiForMCRC,
    RobertaModelHiGForMCRC.model_prefix: RobertaModelHiGForMCRC
}
