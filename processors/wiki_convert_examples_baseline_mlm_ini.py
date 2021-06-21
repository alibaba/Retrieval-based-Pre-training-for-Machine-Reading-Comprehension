import argparse
import json
from collections import Counter
from functools import partial
from typing import List, Dict
import sys
import os
import logging

import numpy as np
import torch
import random
from multiprocessing import Pool
from tqdm import tqdm
from transformers import PreTrainedTokenizer, AutoTokenizer, RobertaTokenizer, BertTokenizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from oss_utils import torch_save_to_oss, load_buffer_from_oss

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(f'fangxi.{__name__}')

"""
Converting like processor `wiki_pre_train_ent_baseline_mlm`.

Mask as BERT do.
"""


def generate_pre_sent_id(example, offset):
    question = example['question']
    passage = example['passage']

    true_id2rank = {
        s['index']: idx for idx, s in enumerate(question + passage)
    }

    pre_idx = []
    for _q_sent in question:
        q_sent_id = _q_sent['index']
        pre_sent_id = q_sent_id + offset
        if pre_sent_id < 0 or pre_sent_id >= len(true_id2rank):
            pre_idx.append(-1)
        else:
            pre_idx.append(true_id2rank[pre_sent_id])
    return pre_idx


def get_sentence_spans(token_tuple, _start_offset, _sent_id_map, _sentence_spans):
    ini_sent_ids, ini_tokens = zip(*token_tuple)
    ini_sent_ids = Counter(ini_sent_ids)
    sorted_ini_sent_ids = sorted(
        ini_sent_ids.items(), key=lambda x: x[0])

    _start_offset = _start_offset
    for ini_s_id, s_len in sorted_ini_sent_ids:
        _new_t_s = _start_offset
        _new_t_e = _start_offset + s_len - 1
        _start_offset = _new_t_e + 1
        _sent_id_map[ini_s_id] = len(_sentence_spans)
        _sentence_spans.append((_new_t_s, _new_t_e))

    return ini_tokens


def process_example(example, max_seq_length, if_bpe, mask_ratio=0.15):
    true_idx2ini_idx_map = {}

    for sent_id, q_sent in enumerate(example['question']):
        true_idx = q_sent['index']
        true_idx2ini_idx_map[true_idx] = sent_id

    q_sent_num = len(example['question'])

    for sent_id, d_sent in enumerate(example['passage']):
        true_idx = d_sent['index']
        true_idx2ini_idx_map[true_idx] = sent_id + q_sent_num

    all_sentences = example['question'] + example['passage']

    true_sentences = []
    for _index in range(len(all_sentences)):
        true_sentences.append(all_sentences[true_idx2ini_idx_map[_index]])

    pure_text = ' '.join([_sent['text'] for _sent in true_sentences])

    test_input_ids = tokenizer.tokenize(pure_text)
    test_noised_input_ids = tokenizer.tokenize(noised_text)

    if not len(test_input_ids) == len(test_noised_input_ids):
        # logger.warning("Consistency checking failed.")
        return None

    tokenizer_outputs = tokenizer(pure_text, padding='max_length',
                                  max_length=max_seq_length, truncation='longest_first')

    input_ids = tokenizer_outputs['input_ids']
    if 'token_type_ids' in tokenizer_outputs:
        token_type_ids = tokenizer_outputs['token_type_ids']
    else:
        token_type_ids = [0]
    attention_mask = tokenizer_outputs['attention_mask']

    mask_token_num = int(len(input_ids) * mask_ratio)

    

    return {
        "input_ids": input_ids,
        "token_type_ids": token_type_ids,
        "attention_mask": attention_mask,
        "labels": pure_input_ids
    }


def data2tensors(all_features: List[Dict]):
    all_input_ids = torch.tensor([f["input_ids"] for f in all_features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f["token_type_ids"] for f in all_features], dtype=torch.long)
    all_attention_mask = torch.tensor([f["attention_mask"] for f in all_features], dtype=torch.long)
    all_labels = torch.tensor([f["labels"] for f in all_features], dtype=torch.long)

    all_feature_index = torch.arange(all_input_ids.size(0), dtype=torch.long)

    logger.info(f'input_ids size: {all_input_ids.size()}')
    logger.info(f'token_type_ids size: {all_token_type_ids.size()}')
    logger.info(f'attention_mask size: {all_attention_mask.size()}')
    logger.info(f'labels size: {all_labels.size()}')

    return all_input_ids, all_token_type_ids, all_attention_mask, all_labels, all_feature_index


def initializer(tokenizer_for_convert: PreTrainedTokenizer):
    global tokenizer
    tokenizer = tokenizer_for_convert


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file_list", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--tokenizer_name", type=str, required=True)
    parser.add_argument("--oss_dir", type=str, default=None)

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    MAX_SEQ_LENGTH = 512
    IF_ADD_LM = True

    cache_suffix = f'wiki_pre_train_mlm_{MAX_SEQ_LENGTH}_{IF_ADD_LM}'

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

    if isinstance(tokenizer, RobertaTokenizer):
        IF_BPE = True
    elif isinstance(tokenizer, BertTokenizer):
        IF_BPE = False
    else:
        raise RuntimeError(tokenizer.__class__.__name__)

    # set_bucket_dir(args.output_dir)
    print(cache_suffix, IF_BPE)
    print(os.cpu_count())

    input_file_list = json.load(open(args.input_file_list, "r"))
    input_file_steps = {}
    for input_file in input_file_list:

        output_file_name = f'{input_file.split("/")[-1]}_{cache_suffix}'

        logger.info(f"Reading data set from {input_file}...")

        try:
            _cache_examples, _cache_tensors = torch.load(load_buffer_from_oss(os.path.join(args.oss_dir, output_file_name)))
            logger.info("Find cache file from oss. Saving to volume...")
            input_file_steps[input_file] = _cache_tensors[0].size(0)
            torch.save((_cache_examples, _cache_tensors), os.path.join(args.output_dir, output_file_name))
            continue
        except:
            pass

        examples = json.load(open(input_file, "r", encoding="utf-8"))

        unique_id = 100000

        # features = []
        # for example in tqdm(examples, total=len(examples)):
        #     features.append(process_example(example, max_seq_length=MAX_SEQ_LENGTH, if_bpe=IF_BPE))

        with Pool(os.cpu_count(), initializer=initializer, initargs=(tokenizer,)) as p:
            annotate_ = partial(
                process_example,
                max_seq_length=MAX_SEQ_LENGTH,
                if_bpe=IF_BPE
            )
            features = list(
                tqdm(
                    p.imap(annotate_, examples, chunksize=128),
                    total=len(examples),
                    desc="convert examples to features",
                    disable=False,
                )
            )

        new_features = []
        example_index = 0
        for example_feature in tqdm(
                features, total=len(features), desc="add example index and unique id",
                disable=False
        ):
            if example_feature is None:
                continue
            example_feature["example_index"] = example_index
            example_feature["unique_id"] = unique_id
            example_index += 1
            unique_id += 1
            new_features.append(example_feature)
        features = new_features

        logger.info(f"Converting {len(features)} features.")
        logger.info(f"Starting convert features to tensors.")
        all_tensors = data2tensors(features)

        logger.info(f"Saving to oss...")

        # torch_save_to_oss((examples, all_tensors), output_file_name)
        torch.save((examples, all_tensors), os.path.join(args.output_dir, output_file_name))
        # torch_save_to_oss((examples, all_tensors), os.path.join(args.oss_dir, output_file_name))

        input_file_steps[input_file] = all_tensors[0].size(0)

        del examples
        del all_tensors

        logger.info("Finished.")

    steps_file_name = args.input_file_list[:-5] + '-' + cache_suffix + '-steps.json'
    with open(os.path.join(args.output_dir, steps_file_name), 'w') as f:
        json.dump(input_file_steps, f, indent=2)
