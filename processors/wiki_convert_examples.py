import argparse
import json
from collections import Counter
from typing import List, Dict
import sys
import os

import nltk
import numpy as np
import torch
import random
from multiprocessing import Pool
from tqdm import tqdm
from transformers import PreTrainedTokenizer, AutoTokenizer, RobertaTokenizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


from oss_utils import torch_save_to_oss, set_bucket_dir
from general_util.logger import get_child_logger


logger = get_child_logger(__name__)


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


def process_example(example, tokenizer: PreTrainedTokenizer, max_seq_length,
                    if_add_lm, if_bpe, unique_id, example_id):
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

    tokenizer_outputs = tokenizer.encode_plus(ini_q_tokens, text_pair=ini_p_tokens,
                                              padding='max_length', max_length=max_seq_length)

    input_ids = tokenizer_outputs['input_ids']
    if 'token_type_ids' in tokenizer_outputs:
        token_type_ids = tokenizer_outputs['token_type_ids']
    else:
        token_type_ids = [0]
    attention_mask = tokenizer_outputs['attention_mask']

    if if_add_lm:
        cleaned_q_tokens = tokenizer.tokenize(
            ' '.join([tmp['text'] for tmp in example['question']]))
        # Consistency check
        if len(cleaned_q_tokens) != len(q_tokens):
            logger.warning("Consistency checking in question failed.")
            return None

        cleaned_d_tokens = tokenizer.tokenize(
            ' '.join([tmp['text'] for tmp in example['passage']]))
        # consistency check
        if len(d_tokens) != len(cleaned_d_tokens):
            logger.warning("Consistency checking in document failed.")
            return None

        cleaned_tk_outputs = tokenizer(cleaned_q_tokens, text_pair=cleaned_d_tokens,
                                       padding='max_length', max_length=max_seq_length,
                                       truncation='longest_first')

        mlm_ids = cleaned_tk_outputs['input_ids']
        assert len(mlm_ids) == len(input_ids), (
            tokenizer.convert_ids_to_tokens(mlm_ids),
            tokenizer.convert_ids_to_tokens(input_ids)
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

    # most_relevent_ids
    most_relevent = []
    for q_id, relevent_ids in enumerate(example['most_relevent']):
        if q_id not in sent_id_map:
            continue

        tmp_ls = []
        for _rel_id in relevent_ids:
            if _rel_id not in sent_id_map:
                continue
            else:
                tmp_ls.append(sent_id_map[_rel_id])

        while len(tmp_ls) < 3:
            tmp_ls.append(-1)
        assert len(tmp_ls) == 3

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
        "unique_id": unique_id,
        "example_id": example_id,
        "most_relevent": most_relevent,
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

    most_relevent = torch.zeros(
        (data_num, max_q_sent_num, 3), dtype=torch.long).fill_(-1)
    for f_id, f in enumerate(all_features):
        most_relevent[f_id, :len(f["most_relevent"])] = torch.LongTensor(
            f["most_relevent"])

    all_sentence_spans = torch.zeros(
        (all_input_ids.size(0), max_sent_num, 2), dtype=torch.long).fill_(-1)
    for f_id, f in enumerate(all_features):
        all_sentence_spans[f_id, :len(f['sentence_spans'])] = torch.LongTensor(
            f['sentence_spans'])

    all_feature_index = torch.arange(
        all_input_ids.size(0), dtype=torch.long)

    logger.info(f'input_ids size: {all_input_ids.size()}')
    logger.info(f'token_type_ids size: {all_token_type_ids.size()}')
    logger.info(f'attention_mask size: {all_attention_mask.size()}')
    logger.info(f'sentence_spans size: {all_sentence_spans.size()}')
    logger.info(f'most relevent size: {most_relevent.size()}')
    logger.info(f'answers size: {all_answers.size()}')
    logger.info(f'pre_answers size: {all_pre_answers.size()}')
    logger.info(f'nn_answers size: {all_nn_answers.size()}')
    logger.info(f'pp_answers size: {all_pp_answers.size()}')

    if add_lm:
        return all_input_ids, all_token_type_ids, all_attention_mask, all_sentence_spans, \
            all_answers, all_pre_answers, all_nn_answers, all_pp_answers, all_mlm_ids, \
            most_relevent, all_feature_index
    else:
        return all_input_ids, all_token_type_ids, all_attention_mask, all_sentence_spans, \
            all_answers, all_pre_answers, all_nn_answers, all_pp_answers, \
            most_relevent, all_feature_index


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file_list", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--tokenizer_name", type=str, required=True)

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    MAX_SEQ_LENGTH = 512
    IF_ADD_LM = True
    IF_BPE = True

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

    set_bucket_dir(args.output_dir)

    input_file_list = json.load(open(args.input_file_list, "r"))
    for input_file in input_file_list:

        output_file_name = f'{input_file.split("/")[-1]}_{MAX_SEQ_LENGTH}_{IF_ADD_LM}_{IF_BPE}_all'

        logger.info(f"Reading data set from {input_file}...")

        examples = json.load(open(input_file, "r", encoding="utf-8"))

        unique_id = 100000

        features = []

        def _call_back(_feature):
            if _feature is not None:
                features.append(_feature)

                # if len(features) % 50000 == 0:
                    # print(len(features), end='\r', flush=True)
                print(len(features), end='\r', flush=True)

        pool = Pool()
        for (example_index, example) in enumerate(tqdm(examples, total=len(examples))):

            pool.apply_async(process_example,
                            args=(example, tokenizer, MAX_SEQ_LENGTH, IF_ADD_LM,
                                IF_BPE, unique_id, example_index),
                            callback=_call_back)
            unique_id += 1

        pool.close()
        pool.join()

        logger.info(f"Converting {len(features)} features.")
        logger.info(f"Starting convert features to tensors.")
        all_tensors = data2tensors(features)

        logger.info(f"Saving to oss...")
        torch_save_to_oss((examples, all_tensors), output_file_name)

        del examples
        del all_tensors

        logger.info("Finished.")
