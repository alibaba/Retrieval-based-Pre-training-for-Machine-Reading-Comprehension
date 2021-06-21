import json
from multiprocessing import Pool
from typing import List, Dict

import torch
from tqdm import tqdm
from transformers import PreTrainedTokenizer, RobertaTokenizer

from general_util.logger import get_child_logger

logger = get_child_logger(__name__)

r"""
This processor take the noisy examples file as input,
where the examples are marked with entities to be masked and replaced.

Besides, the query sentences are shuffled.

Compared with `wiki_pre_train_ent` processor, this processor use multiprocess
to convert the examples into features and add extra `pre_answers` domain which
can be used to predict which sentence is the previous sentence.
"""


class WikiPreTrainProcessorMLM(object):
    reader_name = 'wiki_pre_train_mlm'

    def __init__(self, args):
        super().__init__()
        self.opt = vars(args)
        self.opt['cache_suffix'] = f'{self.reader_name}_{args.max_seq_length}_{args.add_lm}'
        self.features = []

    @staticmethod
    def read(input_file=None, io_buffer=None):
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

    @staticmethod
    def process_example(example, tokenizer: PreTrainedTokenizer, max_seq_length, if_bpe, unique_id, example_id):
        true_idx2ini_idx_map = {}
        truncated = 0

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
            return -1

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
            "labels": pure_input_ids,
            "unique_id": unique_id,
            "example_id": example_id,
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
        if_bpe = 'roberta' in self.opt['base_model_type']
        logger.info(f'Converting args: {max_seq_length}, {if_bpe}.')

        if_parallel = self.opt['if_parallel']

        unique_id = 1000000000
        features = []

        dropped = 0

        if not if_parallel:

            for (example_index, example) in tqdm(enumerate(examples), desc='Convert examples to features',
                                                 total=len(examples), dynamic_ncols=True):
                feature = self.process_example(example, tokenizer, max_seq_length, if_bpe,
                                               unique_id, example_index)
                if feature == -1:
                    dropped += 1
                    continue

                unique_id += 1
                features.append(feature)

        else:

            pool = Pool()
            for (example_index, example) in tqdm(enumerate(examples), desc='Convert examples to features',
                                                 total=len(examples), dynamic_ncols=True):
                pool.apply_async(self.process_example,
                                 args=(example, tokenizer, max_seq_length,
                                       if_bpe, unique_id, example_index),
                                 callback=self._call_back)
            pool.close()
            pool.join()

            features = self.features

            for _feature in features:
                _feature["unique_id"] = unique_id
                unique_id += 1

        logger.info(f'Generate {len(features)} features.')
        logger.info(f'Drop {dropped} features for non-matching.')
        logger.info(f'Start convert features to tensors.')
        all_tensors = self.data_to_tensors(features)
        logger.info(f'Finished.')

        return all_tensors

    @staticmethod
    def data_to_tensors(all_features: List[Dict]):
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

    @staticmethod
    def collate_fn(batch):
        input_ids, token_type_ids, attention_mask, labels, \
            feature_index = list(zip(*batch))

        input_ids = torch.stack(input_ids, dim=0)
        token_type_ids = torch.stack(token_type_ids, dim=0)
        attention_mask = torch.stack(attention_mask, dim=0)
        labels = torch.stack(labels, dim=0)
        feature_index = torch.stack(feature_index, dim=0)

        return input_ids, token_type_ids, attention_mask, \
            labels, feature_index

    def generate_inputs(self, batch, device):

        inputs = {
            'input_ids': batch[0].to(device),
            'attention_mask': batch[2].to(device),
            'labels': batch[3].to(device)
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
