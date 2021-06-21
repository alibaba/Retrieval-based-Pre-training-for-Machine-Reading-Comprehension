from transformers import PreTrainedModel

from general_util.logger import get_child_logger
from .bert_cquery_models import c_query_bert_models_map
from .bert_for_cls import BertForSequenceClassification, BertForMultipleChoice
from .bert_gat_self_sup_pretrain import BertKnowledgePreTrainedModelForSentenceReOrder
from .bert_hie_self_sup_pretrain import BertSelfSupPretainClsQuery
from .bert_hie_self_sup_pretrain_test import BertSelfSupPretainClsQuery as BertSSPClsQueryTest
from .bert_iter_models import iter_bert_models_map
from .bert_query_models import query_bert_models_map
from .bert_rsg import rsg_model_map
from .bert_ssp_es import BertSelfSupPretainExtraSup
from .bert_ssp_kg import BertSSPKG, BertSSPKGMCRC, BertSSPKGRS, BertSSPKGRSMCRC
from .bert_ssp_sg import BertSSPSentenceGraph
# from .roberta_cquery_d_models import cuqery_d_roberta_models_map
# from .roberta_cquery_ff_for_mcrc import cuqery_ff_roberta_models_mcrc_map
# from .roberta_cquery_ff_models import cuqery_ff_roberta_models_map
# from .roberta_cquery_models import cquery_roberta_models_map
# from .roberta_cquery_models_for_mcrc import cquery_roberta_models_for_mcrc_map
# from .roberta_hie_models import hierarchical_roberta_models_map
# from .roberta_hie_models_for_mcrc import hierarchical_roberta_models_mcrc_map
from .roberta_iter_models import iter_roberta_models_map
# from .roberta_models import roberta_model_map
# from .roberta_models_for_mcrc import roberta_models_for_mcrc_map
# from .roberta_rsg import roberta_rsg_model_map
# from .roberta_sr_simple import roberta_models_for_mcrc_simple_map
# from .roberta_sum_models import summarized_roberta_models_map
# from .roberta_sum_models_for_mcrc import summarized_roberta_models_for_mcrc_map

logger = get_child_logger(__name__)

model_map = {
    BertSelfSupPretainClsQuery.model_prefix: BertSelfSupPretainClsQuery,
    BertForSequenceClassification.model_prefix: BertForSequenceClassification,
    BertForMultipleChoice.model_prefix: BertForMultipleChoice,
    BertKnowledgePreTrainedModelForSentenceReOrder.model_prefix: BertKnowledgePreTrainedModelForSentenceReOrder,
    BertSSPClsQueryTest.model_prefix: BertSSPClsQueryTest,
    BertSSPSentenceGraph.model_prefix: BertSSPSentenceGraph,
    BertSelfSupPretainExtraSup.model_prefix: BertSelfSupPretainExtraSup,
    BertSSPKG.model_prefix: BertSSPKG,
    BertSSPKGMCRC.model_prefix: BertSSPKGMCRC,
    BertSSPKGRS.model_prefix: BertSSPKGRS,
    BertSSPKGRSMCRC.model_prefix: BertSSPKGRSMCRC,
}

# model_map.update(rsg_model_map)
# model_map.update(roberta_rsg_model_map)
# model_map.update(roberta_model_map)
# model_map.update(roberta_models_for_mcrc_map)
# model_map.update(roberta_models_for_mcrc_simple_map)
# model_map.update(hierarchical_roberta_models_map)
# model_map.update(hierarchical_roberta_models_mcrc_map)
# model_map.update(summarized_roberta_models_map)
# model_map.update(summarized_roberta_models_for_mcrc_map)
# model_map.update(cquery_roberta_models_map)
# model_map.update(cquery_roberta_models_for_mcrc_map)
# model_map.update(cuqery_ff_roberta_models_map)
# model_map.update(cuqery_ff_roberta_models_mcrc_map)
# model_map.update(cuqery_d_roberta_models_map)
# model_map.update(c_query_bert_models_map)
# model_map.update(query_bert_models_map)
model_map.update(iter_bert_models_map)
model_map.update(iter_roberta_models_map)


def get_added_args(args, model_class: PreTrainedModel):
    opt = vars(args)
    if hasattr(model_class.config_class, 'added_configs'):
        extra_args = {
            key: opt[key] for key in model_class.config_class.added_configs if key in opt
        }

        logger.info(f"Expected extra args: {model_class.config_class.added_configs}")
        logger.info(f"Received args:")
        logger.info(extra_args)
        return extra_args
    else:
        return {}


def clean_state_dict(model_class: PreTrainedModel, state_dict):
    if model_class.base_model_prefix == 'roberta':
        for x in ['sum_roberta', 'hie_roberta']:
            new_state_dict = {}
            for key in state_dict.keys():
                if x in key:
                    new_key = key.replace(x + '.', '')
                    new_state_dict[new_key] = state_dict[key]
                else:
                    new_state_dict[key] = state_dict[key]
            # print(new_state_dict.keys())
            return new_state_dict

    return state_dict


def register_model(cls):
    if not hasattr(cls, "model_prefix"):
        raise ValueError("Model to be registered must have \"model_prefix\" attribute.")
    if not issubclass(cls, PreTrainedModel):
        raise ValueError(f"Model {cls.__name__} must extend PreTrainedModel.")
    model_map[cls.model_prefix] = cls
    return cls
