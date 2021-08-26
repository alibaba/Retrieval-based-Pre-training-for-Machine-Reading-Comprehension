# REPT

## Pre-Training Setup
### Prepare Dataset for Pre-Training

Firstly, you should download the Wikipedia dump by your self and do custom pre-processing. We provide a script to download the dump 
and do necessary pre-processing:
```
cd processors
python wiki_en_pretrain_processor.py
```
You can change the tokenizer you used and some basic settings in the script by yourself.

### Data Construction

Run the following script:
```
python processors/wiki_en_mask_shuffle_wk5_mp.py \\
    --input_file <the wikipedia data file> \\
    --seed 42 \\
    --keep_prob 0.7 \\  # keep 70% sentences as the passage.
    --ratio (0.8,0.1,0.3,0.0,0.0,0.0)  # 80% entities for mask and 10 entities for replacement. The same meaning for noun and pronouns.
```

You can change the hyper-parameters by yourself.

### Run Pre-Training

There are two pre-training scripts:
```
main_wiki_pretrain.py
main_wiki_pretrain_distrib_pai_volume.py
```
The first need ``torch==1.6.0`` while the second need ``torch==1.5.1``.
We used the second one to pre-train all the models on **Alibaba PAI**. As a result, there are some modules not available, 
since they are provided to PAI users only, e.g., 
```
torch_save_to_oss, set_bucket_dir, load_buffer_from_oss, json_save_to_oss
```
you need to use the normal operations such as ``torch.save`` or ``json.dump`` to replace them.  
Then you can run following command to pre-train your own model:
```
python main_wiki_pretrain_distrib_pai_volume.py --config <pre-training config file>
```
The config files containing the detailed hyper-parameters are listed in the next section.

### Fine-tuning
For fine-tuning, use either
```
python main_multirc_torch151.py --config <fine-tuning config>
```
or directly run the corresponding bash script under ``scripts/``.  
The configs or bash scripts for specific tasks are also listed below.

## Experiments Scripts and Configs

### Pre-Training
#### BERT

|  model name     |  path      |
|  :-------       |  :------   |
|  BERT-Q w. R/S  |  configs/bert_pretrain/bert-iter-sr-mlm1.json               |
|  BERT-Q w. R/S (40k)    |  configs/bert_pretrain/bert-iter-sr-mlm2.json       |
|  BERT-Q w. R/S (60%)    |  configs/bert_pretrain/bert-iter-sr-mlm3.json       |
|  BERT-Q w. R/S (90%)    |  configs/bert_pretrain/bert-iter-sr-mlm4.json       |
|  BERT-Q w. S (No Mask)  |  configs/bert_pretrain/bert-iter-sr-no-mask-1.json  |
|  BERT w. M              |  configs/bert_pretrain/bert-mlm-baseline2.json      |
|  BERT-Q w. S            |  configs/bert_pretrain/bert-iter-sr1.json           |
|  BERT-Q w. R            |  configs/bert_pretrain/bert-iter-mlm1.json          |

#### RoBERTa

|  model name     |  path      |
|  :-------       |  :------   |
|  RoBERTa-Q w. R/S  |  configs/roberta_pretrain/iter_roberta/roberta-iter-sr-mlm-s2.json |

### Fine-Tuning

#### RACE

|  model name     |  path      |
|  :-------       |  :------   |
|  BERT-Q                 |  configs/race/bert-base-iter-mcrc-wo-pt.json       |
|  BERT-Q w. R/S          |  configs/race/bert-base-iter-mcrc-v1-3-0-20k.json  |
|  BERT-Q w. R/S (40k)    |  configs/race/iter_sr_mlm_2/bert-base-iter-mcrc-v1-4-0-40k.json |
|  BERT-Q w. R/S (60%)    |  configs/race/iter_sr_mlm_3/bert-base-iter-mcrc-v1-5-0-20k.json |
|  BERT-Q w. R/S (90%)    |  configs/race/iter_sr_mlm_4/bert-base-iter-mcrc-v1-6-0-20k.json |
|  BERT-Q w. S (No Mask)  |  configs/race/iter_sr_no_mask/bert-base-iter-mcrc-v1-sr-wom-0-20k.json |
|  BERT-Q w. M            |  configs/race/mlm_baseline/bert-base-iter-mcrc-v1-mlm-1-0-20k.json|
|  BERT w. M              |  configs/race/mlm_baseline/bert-base-mcrc-v1-mlm-1-0-20k.json   |
|  BERT-Q w. S            |  configs/race/iter_sr_1/bert-base-iter-mcrc-v1-iter_sr_1-0-20k.json|
|  BERT-Q w. R            |  configs/race/iter_mlm_1r/bert-base-iter-mcrc-v1-iter_mlm_1r-0-20k.json|
|  RoBERTa-Q w. R/S       |  configs/race/roberta_iter_sr_mlm_s2/rob-iter-mcrc-v1-s2-0-80k.json  |

#### DREAM

|  model name     |  path      |
|  :-------       |  :------   |
|  BERT-Q                 |  scripts/dream/bert_iter/mcrc1.sh       |
|  BERT-Q w. R/S          |  configs/dream/bert-base-iter-mcrc-3-6-20k.json     |
|  BERT-Q w. M            |  configs/dream/mlm_baseline/bert-base-iter-mcrc-v1-mlm-2-1-20k.json|
|  BERT w. M              |  configs/dream/mlm_baseline/bert-base-mcrc-v1-mlm-2-1-20k.json   |
|  BERT-Q w. S            |  configs/dream/iter_bert_sr/bert-base-iter-mcrc-iter_sr_1-1-20k.json|
|  BERT-Q w. R            |  configs/dream/iter_bert_mlm/bert-base-iter-mcrc-iter_mlm_1r-1-20k.json|
|  RoBERTa-Q w. R/S       |  configs/dream/roberta/iter_mcrc/rob-iter-mcrc-v1-0-sr-mlm-s2-80k.json|

#### ReClor

|  model name     |  path      |
|  :-------       |  :------   |
|  BERT-Q                 |  configs/reclor/iter_sr_mlm_1/bert-base-iter-mcrc-wo-pt-v1-0.json|
|  BERT-Q w. R/S          |  configs/reclor/iter_sr_mlm_1/bert-base-iter-mcrc-sr-mlm-1-v1-0.json|
|  BERT-Q w. M            |  configs/reclor/mlm_baseline_2/bert-base-iter-mcrc-mlm-baseline-2-v1-0.json|
|  BERT w. M              |  configs/reclor/mlm_baseline_2/bert-base-mcrc-mlm-baseline-2-v1-0.json   |
|  BERT-Q w. S            |  configs/reclor/iter_sr_1/bert-base-iter-mcrc-sr-mlm-1-v1-0.json|
|  BERT-Q w. R            |  configs/reclor/iter_mlm_1r/bert-base-iter-mcrc-mlm-1r-v1-0.json|

#### MultiRC

|  model name     |  path      |
|  :-------       |  :------   |
|  BERT-Q                 |  scripts/multi_rc/bert/bert_iter_sc_v3.sh|
|  BERT-Q w. R/S          |  scripts/multi_rc/bert/bert_iter_sc_v3.sh|
|  BERT-Q w. R/S (40k)    |  scripts/multi_rc/bert/iter_sr_mlm_2/bert_iter_sc_v3.sh|
|  BERT-Q w. R/S (60%)    |  scripts/multi_rc/bert/iter_sr_mlm_3/bert_iter_sc_v3.sh|
|  BERT-Q w. R/S (90%)    |  scripts/multi_rc/bert/iter_sr_mlm_4/bert_iter_sc_v3.sh|
|  BERT-Q w. S (No Mask)  |  scripts/multi_rc/bert/iter_sr_no_mask/bert_iter_sc_v3.sh |
|  BERT-Q w. M            |  scripts/multi_rc/bert/mlm_baseline_2/bert_iter_sc_v3.sh|
|  BERT w. M              |  configs/multirc/mlm_baseline/bert-base-sc-v1-mlm-2-0-20k.json|
|  BERT-Q w. S            |  scripts/multi_rc/bert/iter_sr/bert_iter_sc_v3.sh|
|  BERT-Q w. R            |  scripts/multi_rc/bert/iter_mlm_1r/bert_iter_sc_v3.sh|
|  RoBERTa-Q w. R/S       |  configs/multirc/roberta/iter_sc_v3_s2/rob-sc-v3-0-sr-mlm-s2-80k.json|


#### SQuAD 2.0

|  model name     |  path      |
|  :-------       |  :------   |
|  BERT-Q w. R/S          |  scripts/squad/iter_bert_qa_v1.sh    |
|  RoBERTa-Q w. R/S       |  scripts/squad/iter_roberta_qa_v1.sh |
