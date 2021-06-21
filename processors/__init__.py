from processors.multi_rc import MultiRCProcessor, MultiRCSentenceProcessor
from .race import RACEProcessor
from .race_sent import RACEProcessor as RACESentenceProcessor
from .wiki_pre_train import WikiPreTrainProcessor
from .wiki_pre_train_ent import WikiEntityPreTrainProcessor
from .wiki_pre_train_ent_ar import WikiEntityPreTrainProcessorAR
from .wiki_pre_train_ent_mp import WikiEntityPreTrainProcessorMP
from .wiki_pre_train_ent_mp_d import WikiEntityPreTrainProcessorMPDouble
from .wiki_pre_train_ent_baseline import WikiEntityPreTrainPureProcessor
from .wiki_pre_train_ent_mlm import WikiEntityPreTrainProcessorOnlyMLM
from .wiki_pre_train_ent_baseline_mlm import WikiPreTrainProcessorMLM
from .wiki_pre_train_ent_uni import WikiEntityPreTrainUnifiedProcessor
from .wiki_pre_train_knowledge import WikiKnowledgePreTrainProcessor
from .wiki_pre_train_knowledge_sent import WikiKnowledgePreTrainProcessor as WikiKnowledgePreTrainProcessorS

processor_map = {
    WikiKnowledgePreTrainProcessor.reader_name: WikiKnowledgePreTrainProcessor,
    WikiPreTrainProcessor.reader_name: WikiPreTrainProcessor,
    MultiRCProcessor.reader_name: MultiRCProcessor,
    MultiRCSentenceProcessor.reader_name: MultiRCSentenceProcessor,
    RACEProcessor.reader_name: RACEProcessor,
    WikiKnowledgePreTrainProcessorS.reader_name: WikiKnowledgePreTrainProcessorS,
    RACESentenceProcessor.reader_name: RACESentenceProcessor,

    WikiEntityPreTrainProcessor.reader_name: WikiEntityPreTrainProcessor,
    WikiEntityPreTrainProcessorAR.reader_name: WikiEntityPreTrainProcessorAR,
    WikiEntityPreTrainProcessorMP.reader_name: WikiEntityPreTrainProcessorMP,
    WikiEntityPreTrainProcessorMPDouble.reader_name: WikiEntityPreTrainProcessorMPDouble,
    
    WikiPreTrainProcessorMLM.reader_name: WikiPreTrainProcessorMLM,
    WikiEntityPreTrainProcessorOnlyMLM.reader_name: WikiEntityPreTrainProcessorOnlyMLM,
    WikiEntityPreTrainPureProcessor.reader_name: WikiEntityPreTrainPureProcessor,
    WikiEntityPreTrainUnifiedProcessor.reader_name: WikiEntityPreTrainUnifiedProcessor
}


def get_processor(args):
    return processor_map[args.reader_name](args)


def register_processor(cls):
    if not hasattr(cls, "reader_name"):
        raise ValueError(f"Processor to be registered should have \"reader_name\" attribute.")
    if cls.reader_name in processor_map:
        raise ValueError(f"Cannot register duplicate processor ({cls.reader_name})")
    processor_map[cls.reader_name] = cls
    return cls
