import torch
from torch import nn

import transformers
from transformers.modeling_roberta import RobertaPreTrainedModel, RobertaConfig, RobertaModel, RobertaLayer

from modules import layers
from models.roberta_hie_models import HierarchicalRobertaModel, HierarchicalRobertaPreTrainedModel
from models.roberta_models import RobertaHAMCRCConfig


class HierarchicalRobertaMCRCConfig(RobertaConfig):
    added_configs = [
        'cls_type', 'no_ff'
    ]

    def __init__(self, cls_type=0, no_ff=False, **kwargs):
        super().__init__(**kwargs)

        self.cls_type = cls_type
        self.no_ff = no_ff

    def expand_configs(self, *args):
        self.added_configs.extend(list(args))


class HierarchicalRobertaModelForMCRC(HierarchicalRobertaPreTrainedModel):

    model_prefix = "hie_roberta_mcrc"
    config_class = HierarchicalRobertaMCRCConfig
    
    def __init__(self, config: HierarchicalRobertaMCRCConfig):
        super().__init__(config)

        self.hie_roberta = HierarchicalRobertaModel(config)

        if config.cls_type == 1:
            self.pooler = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.Tanh()
            )
        else:
            self.pooler = nn.Linear(config.hidden_size, config.hidden_size)

        self.classifier = nn.Linear(config.hidden_size, 1)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.no_ff = config.no_ff
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-1)

        self.init_weights()

    @staticmethod
    def fold_tensor(x):
        if x is None:
            return None
        return x.reshape(x.size(0) * x.size(1), *x.size()[2:])

    def decompose_forward(self, _cls_h, _sent_word_hidden,
                          _sentence_mask, _sent_word_mask,
                          attention_only=False):
        
        fn_attention = self.hie_roberta.sent_transformer.attention
        fn_intermediate = self.hie_roberta.sent_transformer.intermediate
        fn_output = self.hie_roberta.sent_transformer.output

        fn_sent_sum = self.hie_roberta.sent_sum

        batch, sent_num, word_num = _sent_word_mask.size()

        # (batch, 2, word_num, h) -> (batch, 1, 2 * word_num, h)
        q_op_word_hidden = _sent_word_hidden[:, :2].reshape(batch, 1, 2 * word_num, -1)
        q_op_word_mask = _sent_word_mask[:, :2].reshape(batch, 1, 2 * word_num)
        
        q_op_hidden = fn_sent_sum(_cls_h.unsqueeze(1),
                                  q_op_word_hidden, q_op_word_mask,
                                  residual=False).squeeze(1)

        p_hidden = fn_sent_sum(_cls_h.unsqueeze(1),
                               _sent_word_hidden[:, 2:], _sent_word_mask[:, 2:],
                               residual=False).squeeze(1)
        p_hidden = p_hidden * (1 - _sentence_mask[:, 2:].unsqueeze(-1))
        
        hidden_cat = torch.cat([q_op_hidden, p_hidden], dim=1)

        if attention_only:
            return hidden_cat

        sent_mask = _sentence_mask[:, 1:].view(batch, 1, 1, sent_num - 1) * -10000.0
        
        attention_outputs = fn_attention(hidden_cat, sent_mask)
        attention_output = attention_outputs[0]

        intermediate_output = fn_intermediate(attention_output)
        layer_output = fn_output(intermediate_output, attention_output)

        return layer_output


    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                sentence_index=None, sentence_mask=None, sent_word_mask=None,
                labels=None, **kwargs):
        
        batch, num_choice, _ = input_ids.size()
        input_ids = self.fold_tensor(input_ids)
        attention_mask = self.fold_tensor(attention_mask)
        sentence_index = self.fold_tensor(sentence_index)
        sent_word_mask = self.fold_tensor(sent_word_mask)
        sentence_mask = self.fold_tensor(sentence_mask)

        cls_h, sent_word_hidden = self.hie_roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            sentence_index=sentence_index,
            sentence_mask=sentence_mask,
            sent_word_mask=sent_word_mask,
            roberta_only=True
        )

        layer_output = self.decompose_forward(
            cls_h, sent_word_hidden, sentence_mask, sent_word_mask, attention_only=self.no_ff
        )

        q_op_hidden = layer_output[:, 0]
        logits = self.classifier(self.dropout(self.pooler(q_op_hidden))).view(batch, num_choice)

        outputs = (logits,)
        if labels is not None:
            loss = self.loss_fct(logits, labels)
            outputs = (loss,) + outputs

            _, pred = logits.max(dim=-1)
            acc = torch.sum(pred == labels) / (1.0 * batch)

            outputs = outputs + (acc,)

        return outputs


hierarchical_roberta_models_mcrc_map = {

    HierarchicalRobertaModelForMCRC.model_prefix: HierarchicalRobertaModelForMCRC

}
