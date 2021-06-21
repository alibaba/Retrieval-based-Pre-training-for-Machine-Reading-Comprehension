import argparse
import json
from collections import Counter
from functools import partial
from typing import List, Dict
import sys
import os
import json
import logging

import nltk
import numpy as np
import torch
import random
from multiprocessing import Pool
from tqdm import tqdm
from transformers import PreTrainedTokenizer, AutoTokenizer, RobertaTokenizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


from oss_utils import torch_save_to_oss, set_bucket_dir, json_save_to_oss


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(f'fangxi.{__name__}')


"""
Converting like processor `wiki_pre_train_ent_mp` with MLM.
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


def process_example(example, max_seq_length, if_add_lm, if_bpe):

    true_idx2ini_idx_map = {}
    truncated = 0

    pre_sent_ids = generate_pre_sent_id(example, -1)
    example['pre_sent_ids'] = pre_sent_ids

    nn_sent_ids = generate_pre_sent_id(example, 2)
    example['nn_sent_ids'] = nn_sent_ids

    pp_sent_ids = generate_pre_sent_id(example, -2)
    example['pp_sent_ids'] = pp_sent_ids

    q_tokens = []
    for sent_id, q_sent in enumerate(example['question']):
        true_idx = q_sent['index']
        true_idx2ini_idx_map[true_idx] = sent_id

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
            if if_add_lm and len(_tuple) == 3:
                # hack here. Turn the tuple for replacement to mask.
                _tuple = _tuple[:-1]

            if len(_tuple) == 2:  # mask strategy
                ent_s, ent_e = _tuple
                _text = _text + q_text[last_end:ent_s]
                _ent_text = q_text[ent_s:ent_e]
                if if_bpe:
                    if_add_prefix_space = not (ent_s == 0 and sent_id == 0)

                    _mask_text = ' '.join(
                        [tokenizer.mask_token] * len(
                            tokenizer.tokenize(_ent_text, add_prefix_space=if_add_prefix_space))
                    )
                else:
                    _mask_text = ' '.join(
                        [tokenizer.mask_token] * len(tokenizer.tokenize(_ent_text)))

                _text = _text + _mask_text
                last_end = ent_e
            elif len(_tuple) == 3:  # replace strategy
                ent_s, ent_e, _replaced_text = _tuple
                _text = _text + q_text[last_end:ent_s]

                # A bug in pre-processing, remove the "\'", '{', '}' tokens.
                _text = _text + _replaced_text

                last_end = ent_e
            else:
                print(masked_ents)
                print(replaced_ents)
                raise RuntimeError()
        _text = _text + q_text[last_end:]

        if sent_id > 0 and if_bpe:
            sub_words = tokenizer.tokenize(_text, add_prefix_space=True)
        else:
            sub_words = tokenizer.tokenize(_text)

        if len(sub_words) == 0:
            logger.warn(
                f"Found empty sentence in question: {q_sent['text']}")

        q_tokens.extend([(sent_id, t) for t in sub_words])

    q_sent_num = len(example['question'])

    d_tokens = []
    for sent_id, d_sent in enumerate(example['passage']):
        true_idx = d_sent['index']
        true_idx2ini_idx_map[true_idx] = sent_id + q_sent_num

        d_text = d_sent['text']

        if sent_id > 0 and if_bpe:
            sub_words = tokenizer.tokenize(d_text, add_prefix_space=True)
        else:
            sub_words = tokenizer.tokenize(d_text)

        if len(sub_words) == 0:
            logger.warn(
                f"Found empty sentence in passage: {d_sent['text']}")

        d_tokens.extend([(q_sent_num + sent_id, t) for t in sub_words])

    tmp = len(q_tokens) + len(d_tokens)

    # lens_to_remove = tmp + 4 - max_seq_length  # roberta == 4
    if if_bpe:
        lens_to_remove = tmp + 4 - max_seq_length
    else:
        lens_to_remove = tmp + 3 - max_seq_length

    _q_tokens, _d_tokens, _ = tokenizer.truncate_sequences(q_tokens,
                                                           pair_ids=d_tokens,
                                                           num_tokens_to_remove=lens_to_remove,
                                                           truncation_strategy='longest_first')

    if len(_q_tokens) + len(_d_tokens) < tmp:
        # logger.warn("Truncation has occurred in example index: {}".format(str(example_index)))
        truncated += 1

    sent_id_map = {}
    sentence_spans = []
    start_offset = 1  # Offset for cls_token
    # question
    ini_q_tokens = get_sentence_spans(_q_tokens, start_offset,
                                      sent_id_map, sentence_spans)

    start_offset = len(ini_q_tokens) + (3 if if_bpe else 2)

    q_sent_num = len(sentence_spans)

    # passage
    ini_p_tokens = get_sentence_spans(_d_tokens, start_offset,
                                      sent_id_map, sentence_spans)
    assert len(sentence_spans) > 0
    # print(len(ini_q_tokens), len(ini_p_tokens))

    tokenizer_outputs = tokenizer.encode_plus(ini_q_tokens, text_pair=ini_p_tokens,
                                              padding='max_length', max_length=max_seq_length)

    input_ids = tokenizer_outputs['input_ids']
    if 'token_type_ids' in tokenizer_outputs:
        token_type_ids = tokenizer_outputs['token_type_ids']
    else:
        token_type_ids = [0]
    attention_mask = tokenizer_outputs['attention_mask']

    if if_add_lm:
        cleaned_q_tokens = []
        for _idx, _tmp in enumerate(example['question']):
            cleaned_q_tokens.extend(
                tokenizer.tokenize(_tmp['text'], add_prefix_space=(_idx > 0))
            )
        # Consistency check
        if len(cleaned_q_tokens) != len(q_tokens):
            # logger.warning("Consistency checking in question failed.")
            return None

        cleaned_d_tokens = []
        for _idx, _tmp in enumerate(example['passage']):
            cleaned_d_tokens.extend(
                tokenizer.tokenize(_tmp['text'], add_prefix_space=(_idx > 0))
            )
        
        # consistency check
        if len(d_tokens) != len(cleaned_d_tokens):
            logger.warning("Consistency checking in document failed.")
            return None

        cleaned_tk_outputs = tokenizer.encode_plus(cleaned_q_tokens, text_pair=cleaned_d_tokens,
                                                   padding='max_length', max_length=max_seq_length,
                                                   truncation='longest_first')

        mlm_ids = cleaned_tk_outputs['input_ids']
        assert len(mlm_ids) == len(input_ids), (
            len(mlm_ids),
            len(input_ids)
            # tokenizer.convert_ids_to_tokens(mlm_ids),
            # tokenizer.convert_ids_to_tokens(input_ids)
        )

        for seq_idx in range(len(input_ids)):
            if mlm_ids[seq_idx] == input_ids[seq_idx]:
                mlm_ids[seq_idx] = -1

    answer = []
    q_answer = 0
    p_answer = 0
    tot_answer = 0
    for q_id, tgt in enumerate(example['answer']):

        if q_id not in sent_id_map:
            continue
        assert sent_id_map[q_id] == len(
            answer), (q_id, sent_id_map[q_id], len(answer))

        if tgt not in sent_id_map:
            answer.append(-1)
        else:
            tot_answer += 1
            _ans = sent_id_map[tgt]
            answer.append(_ans)
            if _ans < q_sent_num:
                q_answer += 1
            else:
                p_answer += 1

    pre_answer = []
    q_pre_answer = 0
    p_pre_answer = 0
    tot_pre_answer = 0
    for q_id, tgt in enumerate(example['pre_sent_ids']):
        if q_id not in sent_id_map:
            continue

        if tgt not in sent_id_map:
            pre_answer.append(-1)
        else:
            _ans = sent_id_map[tgt]
            pre_answer.append(_ans)
            tot_pre_answer += 1
            if _ans < q_sent_num:
                q_pre_answer += 1
            else:
                p_pre_answer += 1

    nn_answer = []
    for q_id, tgt in enumerate(example['nn_sent_ids']):
        if q_id not in sent_id_map:
            continue

        if tgt not in sent_id_map:
            nn_answer.append(-1)
        else:
            nn_answer.append(sent_id_map[tgt])

    pp_answer = []
    for q_id, tgt in enumerate(example['pp_sent_ids']):
        if q_id not in sent_id_map:
            continue

        if tgt not in sent_id_map:
            pp_answer.append(-1)
        else:
            pp_answer.append(sent_id_map[tgt])

    assert len(answer) == len(pre_answer) == len(
        nn_answer) == len(pp_answer)

    assert len(input_ids) == max_seq_length
    assert len(attention_mask) == max_seq_length
    if 'token_type_ids' in tokenizer_outputs:
        assert len(token_type_ids) == max_seq_length

    true_idx2ini_idx_tuples = sorted(
        true_idx2ini_idx_map.items(), key=lambda x: x[0])
    cleaned_idx = []
    for i, (true_idx, ini_idx) in enumerate(true_idx2ini_idx_tuples):
        assert i == true_idx
        if ini_idx in sent_id_map:
            cleaned_idx.append(sent_id_map[ini_idx])
    assert len(cleaned_idx) == len(sentence_spans)

    return {
        "input_ids": input_ids,
        "token_type_ids": token_type_ids,
        "attention_mask": attention_mask,
        "answers": answer,
        "pre_answers": pre_answer,
        "nn_answers": nn_answer,
        "pp_answers": pp_answer,
        "sentence_spans": sentence_spans,
        "true_sent_ids": cleaned_idx,
        "mlm_ids": mlm_ids if if_add_lm else [],
        "truncated": truncated,
        "answer_statistics": (q_answer, p_answer, tot_answer,
                              q_pre_answer, p_pre_answer, tot_pre_answer)
    }


def data2tensors(all_features: List[Dict], add_lm):
    all_input_ids = torch.LongTensor(
        [f["input_ids"] for f in all_features])
    all_token_type_ids = torch.LongTensor(
        [f["token_type_ids"] for f in all_features])
    all_attention_mask = torch.LongTensor(
        [f["attention_mask"] for f in all_features])

    if add_lm:
        all_mlm_ids = torch.LongTensor(
            [f["mlm_ids"] for f in all_features]
        )

    max_sent_num = max(
        map(lambda x: len(x['sentence_spans']), all_features))
    max_q_sent_num = max(map(lambda x: len(x['answers']), all_features))

    data_num = all_input_ids.size(0)
    all_answers = torch.zeros(
        (data_num, max_q_sent_num), dtype=torch.long).fill_(-1)
    all_pre_answers = torch.zeros(
        (data_num, max_q_sent_num), dtype=torch.long).fill_(-1)
    all_nn_answers = torch.zeros(
        (data_num, max_q_sent_num), dtype=torch.long).fill_(-1)
    all_pp_answers = torch.zeros(
        (data_num, max_q_sent_num), dtype=torch.long).fill_(-1)

    for f_id, f in enumerate(all_features):
        all_answers[f_id, :len(f["answers"])] = torch.LongTensor(
            f["answers"])
        all_pre_answers[f_id, :len(f["pre_answers"])] = torch.LongTensor(
            f["pre_answers"])
        all_nn_answers[f_id, :len(f["nn_answers"])] = torch.LongTensor(
            f["nn_answers"])
        all_pp_answers[f_id, :len(f["pp_answers"])] = torch.LongTensor(
            f["pp_answers"])

    all_sentence_spans = torch.zeros(
        (all_input_ids.size(0), max_sent_num, 2), dtype=torch.long).fill_(-1)
    for f_id, f in enumerate(all_features):
        all_sentence_spans[f_id, :len(f['sentence_spans'])] = torch.LongTensor(
            f['sentence_spans'])

    all_true_sent_ids = torch.zeros((data_num, max_sent_num), dtype=torch.long).fill_(-1)
    for f_id, f in enumerate(all_features):
        all_true_sent_ids[f_id, :len(f["true_sent_ids"])] = torch.LongTensor(
            f["true_sent_ids"]
        )

    all_feature_index = torch.arange(
        all_input_ids.size(0), dtype=torch.long)

    logger.info(f'input_ids size: {all_input_ids.size()}')
    logger.info(f'token_type_ids size: {all_token_type_ids.size()}')
    logger.info(f'attention_mask size: {all_attention_mask.size()}')
    logger.info(f'sentence_spans size: {all_sentence_spans.size()}')
    logger.info(f'answers size: {all_answers.size()}')
    logger.info(f'pre_answers size: {all_pre_answers.size()}')
    logger.info(f'nn_answers size: {all_nn_answers.size()}')
    logger.info(f'pp_answers size: {all_pp_answers.size()}')
    logger.info(f'all_true_sent_ids size: {all_true_sent_ids.size()}')
    if add_lm:
        logger.info(f'mlm_ids size: {all_mlm_ids.size()}')

    if add_lm:
        return all_input_ids, all_token_type_ids, all_attention_mask, all_sentence_spans, \
            all_answers, all_pre_answers, all_nn_answers, all_pp_answers, all_mlm_ids, \
            all_true_sent_ids, all_feature_index
    else:
        return all_input_ids, all_token_type_ids, all_attention_mask, all_sentence_spans, \
            all_answers, all_pre_answers, all_nn_answers, all_pp_answers, \
            all_true_sent_ids, all_feature_index


def initializer(tokenizer_for_convert: PreTrainedTokenizer):
    global tokenizer
    tokenizer = tokenizer_for_convert


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file_list", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--add_lm", default=False, action='store_true')

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    MAX_SEQ_LENGTH = 512
    IF_ADD_LM = args.add_lm
    IF_BPE = True

    cache_suffix = f'wiki_pre_train_ent_mp_{MAX_SEQ_LENGTH}_{IF_ADD_LM}'

    set_bucket_dir(args.output_dir)

    def read_file(_input_file, _cache_suffix):
        _cache_file_name = f'{_input_file}_{_cache_suffix}'
        
        _, _features = torch.load(_cache_file_name)

        steps = _features[0].size(0)
        del _features
        return (_input_file, steps)


    input_file_list = json.load(open(args.input_file_list, "r"))

    with Pool(16) as p:
        _annotate = partial(read_file, _cache_suffix=cache_suffix)
            
        steps = list(
            tqdm(
                p.imap(_annotate, input_file_list, chunksize=1),
                total=len(input_file_list),
                desc="convert examples to features",
                disable=False,
            )
        )

    step_num_dict = {k: v for k, v in steps}

    torch_save_to_oss(step_num_dict, args.input_file_list.split('/')[-1][:-5] + '-' + cache_suffix + '-steps')

    logger.info("finished.")

    
