import torch
from torch import nn
from transformers.modeling_roberta import RobertaPreTrainedModel, RobertaConfig, RobertaModel, ACT2FN

from general_util.utils import LogMetric
from models.roberta_cquery_ff_models import ClsQueryFFRobertaModel, CQueryPreTrainedConfig
from modules import layers


class ClsQueryFFRobertaModelForMCRC(ClsQueryFFRobertaModel):
    model_prefix = 'cquery_ff_roberta_mcrc'
    
    def __init__(self, config: CQueryPreTrainedConfig):
        super().__init__(config)

        self.sent_sum = nn.Linear(config.hidden_size, config.hidden_size)

        if config.cls_type == 1:
            self.pooler = nn.Sequential(
                nn.Linear(config.hidden_size * 2, config.hidden_size),
                nn.Tanh()
            )
        else:
            self.pooler = nn.Linear(config.hidden_size * 2, config.hidden_size)
        
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-1)

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
        attention_mask = self.fold_tensor(attention_mask)
        sentence_index = self.fold_tensor(sentence_index)
        sent_word_mask = self.fold_tensor(sent_word_mask)
        sentence_mask = self.fold_tensor(sentence_mask)

        seq_output = self.roberta(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  token_type_ids=token_type_ids)[0]

        fb, sent_num, seq_len = sent_word_mask.size()
        sentence_index = sentence_index.unsqueeze(-1).expand(
            -1, -1, -1, self.config.hidden_size
        ).reshape(fb, sent_num * seq_len, self.config.hidden_size)

        cls_h = seq_output[:, 0]
        sent_word_hidden = seq_output.gather(dim=1, index=sentence_index).reshape(
            fb, sent_num, seq_len, -1)
        sent_word_hidden = sent_word_hidden * (1 - sent_word_mask.unsqueeze(-1))

        cls_query = self.cls_w(cls_h)
        q_op_word_hidden = sent_word_hidden[:, :2].reshape(fb, 2 * seq_len, -1)
        q_op_word_mask = sent_word_mask[:, :2].reshape(fb, 2 * seq_len)
        q_op_hidden_sent, _ = layers.weighted_sum(cls_query, q_op_word_hidden, q_op_word_mask)

        q_op_hidden_sent = self.query_dropout(q_op_hidden_sent)

        ffn_output = self.ffn_out(self.ffn_act_fn(self.ffn(q_op_hidden_sent)))
        ffn_output = self.ffn_layer_norm(self.query_dropout(ffn_output))

        # =====================================

        q_op_query = self.sent_sum(ffn_output)
        
        p_hidden_sent, _ = layers.sentence_sum(q_op_query, sent_word_hidden[:, 2:], sent_word_mask[:, 2:])
        p_hidden_sent = p_hidden_sent * (1 - sentence_mask[:, 2:].unsqueeze(-1))

        attended_h, _ = layers.weighted_sum(q_op_query, p_hidden_sent, sentence_mask[:, 2:])

        cls_input = torch.cat([ffn_output, attended_h], dim=-1)
        logits = self.classifier(self.dropout(self.pooler(cls_input))).view(batch, num_choice)

        outputs = (logits,)
        if labels is not None:
            loss = self.loss_fct(logits, labels)
            outputs = (loss,) + outputs

            _, pred = logits.max(dim=-1)
            acc = torch.sum(pred == labels) / (1.0 * batch)

            outputs = outputs + (acc,)

        return outputs



class ClsQueryFFRobertaModelForMCRCReuse(ClsQueryFFRobertaModel):
    model_prefix = 'cquery_ff_roberta_mcrc_r'
    
    def __init__(self, config: CQueryPreTrainedConfig):
        super().__init__(config)

        # self.sent_sum = nn.Linear(config.hidden_size, config.hidden_size)
        self.sr_sent_sum = nn.Linear(config.hidden_size, config.hidden_size)

        if config.cls_type == 1:
            self.pooler = nn.Sequential(
                nn.Linear(config.hidden_size * 2, config.hidden_size),
                nn.Tanh()
            )
        else:
            self.pooler = nn.Linear(config.hidden_size * 2, config.hidden_size)
        
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-1)

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
        attention_mask = self.fold_tensor(attention_mask)
        sentence_index = self.fold_tensor(sentence_index)
        sent_word_mask = self.fold_tensor(sent_word_mask)
        sentence_mask = self.fold_tensor(sentence_mask)

        seq_output = self.roberta(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  token_type_ids=token_type_ids)[0]

        fb, sent_num, seq_len = sent_word_mask.size()
        sentence_index = sentence_index.unsqueeze(-1).expand(
            -1, -1, -1, self.config.hidden_size
        ).reshape(fb, sent_num * seq_len, self.config.hidden_size)

        cls_h = seq_output[:, 0]
        sent_word_hidden = seq_output.gather(dim=1, index=sentence_index).reshape(
            fb, sent_num, seq_len, -1)
        sent_word_hidden = sent_word_hidden * (1 - sent_word_mask.unsqueeze(-1))

        cls_query = self.cls_w(cls_h)
        q_op_word_hidden = sent_word_hidden[:, :2].reshape(fb, 2 * seq_len, -1)
        q_op_word_mask = sent_word_mask[:, :2].reshape(fb, 2 * seq_len)
        q_op_hidden_sent, _ = layers.weighted_sum(cls_query, q_op_word_hidden, q_op_word_mask)

        q_op_hidden_sent = self.query_dropout(q_op_hidden_sent)

        ffn_output = self.ffn_out(self.ffn_act_fn(self.ffn(q_op_hidden_sent)))
        ffn_output = self.ffn_layer_norm(self.query_dropout(ffn_output))

        # =====================================

        # q_op_query = self.sent_sum(ffn_output)
        q_op_query = self.sr_sent_sum(ffn_output)

        p_hidden_sent, _ = layers.sentence_sum(q_op_query, sent_word_hidden[:, 2:], sent_word_mask[:, 2:])
        p_hidden_sent = p_hidden_sent * (1 - sentence_mask[:, 2:].unsqueeze(-1))

        attended_h, _ = layers.weighted_sum(q_op_query, p_hidden_sent, sentence_mask[:, 2:])

        cls_input = torch.cat([ffn_output, attended_h], dim=-1)
        logits = self.classifier(self.dropout(self.pooler(cls_input))).view(batch, num_choice)

        outputs = (logits,)
        if labels is not None:
            loss = self.loss_fct(logits, labels)
            outputs = (loss,) + outputs

            _, pred = logits.max(dim=-1)
            acc = torch.sum(pred == labels) / (1.0 * batch)

            outputs = outputs + (acc,)

        return outputs


cuqery_ff_roberta_models_mcrc_map = {
    ClsQueryFFRobertaModelForMCRC.model_prefix: ClsQueryFFRobertaModelForMCRC,
    ClsQueryFFRobertaModelForMCRCReuse.model_prefix: ClsQueryFFRobertaModelForMCRCReuse
}
