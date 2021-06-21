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


def process_example(example, max_seq_length, if_bpe):
    true_idx2ini_idx_map = {}

    for sent_id, q_sent in enumerate(example['question']):
        true_idx = q_sent['index']
        true_idx2ini_idx_map[true_idx] = sent_id

        # wd_pieces = tokenizer.tokenize(q_sent['text'])
        q_text = q_sent['text']
        # (ent_start_char, ent_end_char)
        masked_ents = q_sent['masked_ents']
        # (ent_start_char, ent_end_char, replaced_ent_text)
        replaced_ents = q_sent['replaced_ents']

        _text = ''

        noise = masked_ents + replaced_ents
        noise = sorted(noise, key=lambda x: x[0])

        last_end = 0
        for _tuple in noise:
            # Since the replaced entity may have different amount of sub-tokens, we use <mask> instead.
            if len(_tuple) in [2, 3]:
                if len(_tuple) == 2:
                    ent_s, ent_e = _tuple
                else:
                    ent_s, ent_e, _ = _tuple

                _text = _text + q_text[last_end:ent_s]
                _ent_text = q_text[ent_s:ent_e]
                if if_bpe:
                    if_add_prefix_space = (ent_s > 0 or true_idx > 0)
                    _mask_text = ' '.join([tokenizer.mask_token] * len(
                        tokenizer.tokenize(_ent_text, add_prefix_space=if_add_prefix_space)))
                else:
                    _mask_text = ' '.join([tokenizer.mask_token] * len(
                        tokenizer.tokenize(_ent_text)))

                _text = _text + _mask_text
                last_end = ent_e
            else:
                print(masked_ents)
                print(replaced_ents)
                raise RuntimeError()
        _text = _text + q_text[last_end:]

        q_sent['masked_text'] = _text

    q_sent_num = len(example['question'])

    for sent_id, d_sent in enumerate(example['passage']):
        true_idx = d_sent['index']
        true_idx2ini_idx_map[true_idx] = sent_id + q_sent_num
        d_sent['masked_text'] = d_sent['text']

    all_sentences = example['question'] + example['passage']

    true_sentences = []
    for _index in range(q_sent_num + len(example['passage'])):
        true_sentences.append(all_sentences[true_idx2ini_idx_map[_index]])

    pure_text = ' '.join([_sent['text'] for _sent in true_sentences])
    noised_text = ' '.join([_sent['masked_text'] for _sent in true_sentences])

    test_input_ids = tokenizer.tokenize(pure_text)
    test_noised_input_ids = tokenizer.tokenize(noised_text)

    if not len(test_input_ids) == len(test_noised_input_ids):
        # logger.warning("Consistency checking failed.")
        return None

    pure_tokenizer_outputs = tokenizer(pure_text, padding='max_length',
                                       max_length=max_seq_length, truncation='longest_first')
    noised_tokenizer_outputs = tokenizer(noised_text, padding='max_length',
                                         max_length=max_seq_length, truncation='longest_first')

    input_ids = noised_tokenizer_outputs['input_ids']
    if 'token_type_ids' in noised_tokenizer_outputs:
        token_type_ids = noised_tokenizer_outputs['token_type_ids']
    else:
        token_type_ids = [0]
    attention_mask = noised_tokenizer_outputs['attention_mask']

    pure_input_ids = pure_tokenizer_outputs['input_ids']

    assert len(input_ids) == len(pure_input_ids), (len(input_ids), len(pure_input_ids))

    for seq_idx in range(len(input_ids)):
        if pure_input_ids[seq_idx] == input_ids[seq_idx]:
            pure_input_ids[seq_idx] = -1

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
