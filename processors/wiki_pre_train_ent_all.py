import json
from collections import Counter
from typing import List, Dict

import nltk
import torch
from multiprocessing import Pool
from tqdm import tqdm
from transformers import PreTrainedTokenizer, AutoTokenizer

from general_util.logger import get_child_logger

logger = get_child_logger(__name__)

r"""
This processor take the noisy examples file as input,
where the examples are marked with entities to be maksed and replaced.

Besides, the query sentences are shuffled.

Compared with `wiki_pre_train_ent` processor, this processor use multiprocess
to convert the examples into features and add extra `pre_answers` domain which
can be used to predict which sentence is the previous sentence.

This processor try to incoporate all possible features into the dumped file.
"""


class WikiEntityPreTrainProcessorAll(object):
    reader_name = 'wiki_pre_train_all'

    def __init__(self, args):
        super().__init__()
        self.opt = vars(args)
        self.opt['cache_suffix'] = f'{self.reader_name}_{args.max_seq_length}_{args.add_lm}'
        self.features = []

    def read(self, input_file=None, io_buffer=None):
        logger.info('Reading data set from {}...'.format(input_file))

        reader = open(
            input_file, "r", encoding='utf-8') if input_file is not None else io_buffer
        examples = json.load(reader)

        for ex in examples:
            if 'raw_entity' in ex:
                del ex['raw_entity']
            if 'raw_entity_lemma' in ex:
                del ex['raw_entity_lemma']

        logger.info('Finish reading {} examples from {}'.format(
            len(examples), input_file))
        return examples

    def process_example(self, example, tokenizer, max_seq_length, if_add_lm, if_bpe, unique_id, example_id):
        true_idx2ini_idx_map = {}
        truncated = 0

        pre_sent_ids = self.generate_pre_sent_id(example)
        example['pre_sent_ids'] = pre_sent_ids

        q_tokens = []
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
            # Space for BPE tokenizer.
            if sent_id > 0 and if_bpe:
                _text = ' '

            noise = masked_ents + replaced_ents
            noise = sorted(noise, key=lambda x: x[0])

            last_end = 0
            for _tuple in noise:
                if len(_tuple) == 2:  # mask strategy
                    ent_s, ent_e = _tuple
                    _text = _text + q_text[last_end:ent_s]
                    _ent_text = q_text[ent_s:ent_e]
                    if if_bpe:
                        # '<mask>' and ' <mask>'
                        _mask_text = []
                        for _piece in _ent_text.split():
                            _mask_text.append(
                                ''.join([tokenizer.mask_token] * len(tokenizer.tokenize(_piece))))
                        _mask_text = ' '.join(_mask_text)
                    else:
                        _mask_text = ' '.join(
                            [tokenizer.mask_token] * len(tokenizer.tokenize(_ent_text)))

                    _text = _text + _mask_text
                    last_end = ent_e
                elif len(_tuple) == 3:  # replace strategy
                    ent_s, ent_e, _replaced_text = _tuple
                    _text = _text + q_text[last_end:ent_s]

                    if len(_replaced_text) > 1 and _replaced_text[1] == '{' and _replaced_text[-1] == '}':
                        # FIXME: Here is a bug for workflow 1/2/3
                        _replaced_text = _replaced_text[2:-2]
                    # A bug in pre-processing, remove the "\'", '{', '}' tokens.
                    _text = _text + _replaced_text

                    last_end = ent_e
                else:
                    print(masked_ents)
                    print(replaced_ents)
                    raise RuntimeError()
            _text = _text + q_text[last_end:]

            # Since the replaced entities may have different amount of sub-words compared with
            # the initial entities. We tend to mask them all.
            if if_add_lm:
                raise NotImplementedError()
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
                d_text = ' ' + d_text

            # sub_words = tokenizer.tokenize(d_sent['text'])  # FIXME: `d_text` is not used. The space is ignored.
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

        sent_id_map = {}
        sentence_spans = []
        start_offset = 1  # Offset for cls_token
        # question
        ini_q_tokens = get_sentence_spans(
            _q_tokens, start_offset, sent_id_map, sentence_spans)

        start_offset = len(ini_q_tokens) + (3 if if_bpe else 2)

        q_sent_num = len(sentence_spans)

        # passage
        ini_p_tokens = get_sentence_spans(
            _d_tokens, start_offset, sent_id_map, sentence_spans)

        tokenizer_outputs = tokenizer.encode_plus(ini_q_tokens, text_pair=ini_p_tokens,
                                                  padding='max_length', max_length=max_seq_length)

        input_ids = tokenizer_outputs['input_ids']
        token_type_ids = tokenizer_outputs['token_type_ids'] if 'token_type_ids' in tokenizer_outputs else [
            0]
        attention_mask = tokenizer_outputs['attention_mask']

        answer = []
        q_answer = 0
        p_answer = 0
        tot_answer = 0
        for q_id, tgt in enumerate(example['answer']):
            # if q_id >= len(q_sentence_spans):
            # break
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
            "sentence_spans": sentence_spans,
            "true_sent_ids": cleaned_idx,
            "unique_id": unique_id,
            "example_id": example_id,
            "truncated": truncated,
            "answer_statistics": (q_answer, p_answer, tot_answer,
                                  q_pre_answer, p_pre_answer, tot_pre_answer)
        }

    def _call_back(self, _feature):
        self.features.append(_feature)
        print(len(self.features), end='\r', flush=True)

    @staticmethod
    def generate_pre_sent_id(example):
        question = example['question']
        passage = example['passage']

        true_id2rank = {
            s['index']: idx for idx, s in enumerate(question + passage)
        }

        pre_idx = []
        for _q_sent in question:
            q_sent_id = _q_sent['index']
            pre_sent_id = q_sent_id - 1
            if pre_sent_id < 0:
                pre_idx.append(-1)
            else:
                pre_idx.append(true_id2rank[pre_sent_id])
        return pre_idx

    def convert_examples_to_tensors(self, examples: List, tokenizer: PreTrainedTokenizer):
        max_seq_length = self.opt['max_seq_length']
        if_add_lm = self.opt['add_lm']
        if_bpe = 'roberta' in self.opt['base_model_type']
        logger.info(
            f'Converting args: {max_seq_length}, {if_add_lm}, {if_bpe}.')

        if_parallel = self.opt['if_parallel']

        unique_id = 1000000000
        features = []
        truncated = 0

        if not if_parallel:

            for (example_index, example) in tqdm(enumerate(examples), desc='Convert examples to features',
                                                 total=len(examples), dynamic_ncols=True):
                feature = self.process_example(example, tokenizer, max_seq_length, if_add_lm, if_bpe,
                                               unique_id, example_index)
                unique_id += 1
                truncated += feature["truncated"]
                del feature["truncated"]
                features.append(feature)

        else:

            def _call_back(_feature):
                features.append(_feature)
                print(len(_feature), ' / ', len(examples), end='\r', flush=True)

            pool = Pool()
            for (example_index, example) in tqdm(enumerate(examples), desc='Convert examples to features',
                                                 total=len(examples), dynamic_ncols=True):
                pool.apply_async(self.process_example,
                                 args=(example, tokenizer, max_seq_length, if_add_lm,
                                       if_bpe, unique_id, example_index),
                                 callback=self._call_back)
            pool.close()
            pool.join()

            features = self.features

            for _feature in features:
                truncated += _feature["truncated"]
                del _feature["truncated"]
                _feature["unique_id"] = unique_id
                unique_id += 1

        q_answer, p_answer, tot_answer = (0, 0, 0)
        q_pre_answer, p_pre_answer, tot_pre_answer = (0, 0, 0)
        all_answer = 0
        all_pre_answer = 0

        for _feature in features:
            _answer_tup = _feature["answer_statistics"]
            q_answer += _answer_tup[0]
            p_answer += _answer_tup[1]
            tot_answer += _answer_tup[2]
            q_pre_answer += _answer_tup[3]
            p_pre_answer += _answer_tup[4]
            tot_pre_answer += _answer_tup[5]
            all_answer += len(_feature["answers"])
            all_pre_answer += len(_feature["pre_answers"])
            del _feature["answer_statistics"]

        logger.info(f'answer ratio: {tot_answer * 1.0 / all_answer}')
        logger.info(f'pre answer ratio: {tot_pre_answer * 1.0 / all_pre_answer}')
        logger.info(f'q_answer_ratio: {q_answer * 1.0 / tot_answer}')
        logger.info(f'p_answer_ratio: {p_answer * 1.0 / tot_answer}')
        logger.info(f'q_pre_answer_ratio: {q_pre_answer * 1.0 / tot_pre_answer}')
        logger.info(f'p_pre_answer_ratio: {p_pre_answer * 1.0 / tot_pre_answer}')

        logger.info(f'Generate {len(features)} features.')
        logger.info(
            f'Truncated features: {truncated} / {len(features)} = {truncated * 1.0 / len(features)}. ')
        logger.info(f'Start convert features to tensors.')
        all_tensors = self.data_to_tensors(features)
        logger.info(f'Finished.')

        return all_tensors

    @staticmethod
    def data_to_tensors(all_features: List[Dict]):
        all_input_ids = torch.LongTensor(
            [f["input_ids"] for f in all_features])
        all_token_type_ids = torch.LongTensor(
            [f["token_type_ids"] for f in all_features])
        all_attention_mask = torch.LongTensor(
            [f["attention_mask"] for f in all_features])

        max_sent_num = max(
            map(lambda x: len(x['sentence_spans']), all_features))
        max_q_sent_num = max(map(lambda x: len(x['answers']), all_features))

        data_num = all_input_ids.size(0)
        all_answers = torch.zeros(
            (data_num, max_q_sent_num), dtype=torch.long).fill_(-1)
        all_pre_answers = torch.zeros(
            (data_num, max_q_sent_num), dtype=torch.long).fill_(-1)

        for f_id, f in enumerate(all_features):
            all_answers[f_id, :len(f["answers"])] = torch.LongTensor(
                f["answers"])
            all_pre_answers[f_id, :len(f["pre_answers"])] = torch.LongTensor(
                f["pre_answers"])

        all_sentence_spans = torch.zeros(
            (all_input_ids.size(0), max_sent_num, 2), dtype=torch.long).fill_(-1)
        for f_id, f in enumerate(all_features):
            all_sentence_spans[f_id, :len(f['sentence_spans'])] = torch.LongTensor(
                f['sentence_spans'])

        all_true_sent_ids = torch.zeros((data_num, max_sent_num), dtype=torch.long).fill_(-1)
        for f_id, f in enumerate(all_features):
            all_true_sent_ids[f_id, :len(f['true_sent_ids'])] = torch.LongTensor(
                f['true_sent_ids'])

        all_feature_index = torch.arange(
            all_input_ids.size(0), dtype=torch.long)

        logger.info(f'input_ids size: {all_input_ids.size()}')
        logger.info(f'token_type_ids size: {all_token_type_ids.size()}')
        logger.info(f'attention_mask size: {all_attention_mask.size()}')
        logger.info(f'sentence_spans size: {all_sentence_spans.size()}')
        logger.info(f'true_sent_ids size: {all_true_sent_ids.size()}')
        logger.info(f'answers size: {all_answers.size()}')
        logger.info(f'pre_answers size: {all_pre_answers.size()}')

        return all_input_ids, all_token_type_ids, all_attention_mask, all_sentence_spans, \
            all_true_sent_ids, all_answers, all_pre_answers, all_feature_index

    @staticmethod
    def collate_fn(batch):
        # Just process sentence spans using multi-processing.

        input_ids, token_type_ids, attention_mask, sentence_spans, \
            answers, pre_answers, feature_index = list(zip(*batch))

        input_ids = torch.stack(input_ids, dim=0)
        token_type_ids = torch.stack(token_type_ids, dim=0)
        attention_mask = torch.stack(attention_mask, dim=0)
        answers = torch.stack(answers, dim=0)
        pre_answers = torch.stack(pre_answers, dim=0)
        feature_index = torch.stack(feature_index, dim=0)
        sentence_spans = torch.stack(sentence_spans, dim=0)

        max_query_num = (answers > -1).sum(dim=1).max().item()
        answers = answers[:, :max_query_num]
        pre_answers = pre_answers[:, :max_query_num]

        max_sent_num = (sentence_spans[:, :, 0] > -1).sum(dim=1).max().item()
        sentence_spans = sentence_spans[:, :max_sent_num]

        max_sent_len = (sentence_spans[:, :, 1] -
                        sentence_spans[:, :, 0] + 1).max().item()

        batch = sentence_spans.size(0)
        sentence_index = torch.zeros(
            batch, max_sent_num, max_sent_len, dtype=torch.long)
        sentence_mask = torch.ones(batch, max_sent_num)
        sent_word_mask = torch.ones(batch, max_sent_num, max_sent_len)

        for b in range(batch):
            for sid, span in enumerate(sentence_spans[b]):
                s, e = span[0].item(), span[1].item()
                if s == -1 and e == -1:
                    break
                if s == 0:
                    if e == 0:
                        continue
                    else:
                        s += 1
                lens = e - s + 1
                sentence_index[b, sid, :lens] = torch.arange(
                    s, e + 1, dtype=torch.long)
                sentence_mask[b, sid] = 0
                sent_word_mask[b, sid, :lens] = torch.zeros(lens)

        return input_ids, token_type_ids, attention_mask, sentence_index, \
            sentence_mask, sent_word_mask, answers, pre_answers, feature_index

    def generate_inputs(self, batch, device):
        sentence_index = batch[3].to(device)
        sentence_mask = batch[4].to(device)
        sent_word_mask = batch[5].to(device)
        answers = batch[6].to(device)  # [batch, max_query_sent_num]
        pre_answers = batch[7].to(device)

        inputs = {
            'input_ids': batch[0].to(device),
            'attention_mask': batch[2].to(device),
            "sentence_index": sentence_index,
            "sentence_mask": sentence_mask,
            "sent_word_mask": sent_word_mask,
            "answers": answers,
            "pre_answers": pre_answers
        }
        if self.opt['base_model_type'] in ['bert']:
            inputs["token_type_ids"] = batch[1].to(device)

        return inputs

    def compute_metrics(self, pred, labels, examples):
        pred = torch.LongTensor(pred)
        labels = torch.LongTensor(labels)
        acc_item = (pred == labels).sum().item().float()
        total_num = (labels >= 0).sum().item().float()

        return {
            'accuracy': acc_item / total_num
        }



if __name__ == '__main__':

    from argparse import ArgumentParser

    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True)
    parser.add_argument("--base_model_type", default="roberta", type=str)
    parser.add_argument("--input_file", type=str, default=None, required=True)

    # Other parameters
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")

    
    parser.add_argument("--do_lower_case",
                        default=True,
                        action='store_true',
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")

    # Self-supervised pre-training
    parser.add_argument('--add_lm', default=False, action='store_true')
    parser.add_argument('--if_parallel', default=False, action='store_true')

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    processor = WikiEntityPreTrainProcessorAll(args)

    all_examples = processor.read(input_file=args.input_file)

    logger.info(f"Loaded {len(all_examples)} examples from {args.input_file}")

    features = []
    truncated = 0

    def _call_back(_feature):
        features.append(_feature)
        if len(features) % 50000 == 0:
            print(f"{len(features)} / {len(all_examples)}")

    max_seq_length = args.max_seq_length
    if_add_lm = args.add_lm
    if_bpe = 'roberta' in args.base_model_type

    pool = Pool()
    for (example_index, example) in tqdm(enumerate(all_examples),desc='Convert examples to features',
                                         total=len(all_examples), dynamic_ncols=True):
        pool.apply_async(processor.process_example,
                         args=(example, tokenizer, max_seq_length, if_add_lm,
                               if_bpe, unique_id, example_index),
                         callback=_call_back)
    pool.close()
    pool.join()

    for _feature in features:
        truncated += _feature["truncated"]
        del _feature["truncated"]

    q_answer, p_answer, tot_answer = (0, 0, 0)
    q_pre_answer, p_pre_answer, tot_pre_answer = (0, 0, 0)
    all_answer = 0
    all_pre_answer = 0

    for _feature in features:
        _answer_tup = _feature["answer_statistics"]
        q_answer += _answer_tup[0]
        p_answer += _answer_tup[1]
        tot_answer += _answer_tup[2]
        q_pre_answer += _answer_tup[3]
        p_pre_answer += _answer_tup[4]
        tot_pre_answer += _answer_tup[5]
        all_answer += len(_feature["answers"])
        all_pre_answer += len(_feature["pre_answers"])
        del _feature["answer_statistics"]

    logger.info(f'answer ratio: {tot_answer * 1.0 / all_answer}')
    logger.info(f'pre answer ratio: {tot_pre_answer * 1.0 / all_pre_answer}')
    logger.info(f'q_answer_ratio: {q_answer * 1.0 / tot_answer}')
    logger.info(f'p_answer_ratio: {p_answer * 1.0 / tot_answer}')
    logger.info(f'q_pre_answer_ratio: {q_pre_answer * 1.0 / tot_pre_answer}')
    logger.info(f'p_pre_answer_ratio: {p_pre_answer * 1.0 / tot_pre_answer}')
    
    logger.info(f'Generate {len(features)} features.')
    logger.info(f'Truncated features: {truncated} / {len(features)} = {truncated * 1.0 / len(features)}. ')
    logger.info(f'Start convert features to tensors.')
    all_tensors = processor.data_to_tensors(features)
    logger.info(f'Finished.')

    del features

    cached_train_features_file = train_file + '_' + processor.opt['cache_suffix']

    torch.save((all_examples, all_tensors), cached_train_features_file)

    logger.info(f"Saved tensors to {cached_train_features_file}.")

