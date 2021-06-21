import json
from collections import Counter
from typing import List, Dict

import nltk
import torch
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from general_util.logger import get_child_logger

logger = get_child_logger(__name__)


class WikiEntityPreTrainProcessorOnlyMLM(object):
    r"""
    This processor doesn't incorporate mask and replacement into processing.
    In other words, the model only need to re-order the pure sentences.

    This processor will re-rank the shuffled sentences as what they were
    and add the MLM task.
    """

    reader_name = 'wiki_pre_train_ent_only_mlm'

    def __init__(self, args):
        super().__init__()
        self.opt = vars(args)
        self.opt['cache_suffix'] = f'{self.reader_name}_{args.max_seq_length}'

    def read(self, input_file):
        logger.info('Reading data set from {}...'.format(input_file))

        with open(input_file, "r", encoding='utf-8') as reader:
            examples = json.load(reader)

        # Remove the unused key to save memory
        for ex in examples:
            if 'raw_entity' in ex:
                del ex['raw_entity']
            if 'raw_entity_lemma' in ex:
                del ex['raw_entity_lemma']
            if 'ini_article' in ex:
                del ex['ini_article']
            if 'answer' in ex:
                del ex['answer']

        logger.info('Finish reading {} examples from {}'.format(
            len(examples), input_file))
        return examples

    def convert_examples_to_tensors(self, examples: List, tokenizer: PreTrainedTokenizer):
        max_seq_length = self.opt['max_seq_length']
        if_bpe = 'roberta' in self.opt['base_model_type']
        logger.info(f'Converting args: {max_seq_length}, {if_bpe}.')

        all_input_ids = []
        all_token_type_ids = []
        all_attention_mask = []
        all_labels = []

        for (example_index, example) in tqdm(enumerate(examples), desc='Convert examples to features',
                                             total=len(examples), dynamic_ncols=True):

            masked_piece_num = 0
            # Replace the selected tokens with <mask>
            for sent_id, q_sent in enumerate(example['question']):

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
                    # Since replacement will make the new text has a different length
                    # with the initial text, we mask the tokens to be replaced, too.
                    if len(_tuple) == 3:
                        _tuple = _tuple[:-1]
                    if len(_tuple) == 2:  # mask strategy
                        ent_s, ent_e = _tuple
                        _text = _text + q_text[last_end:ent_s]
                        _ent_text = q_text[ent_s:ent_e]
                        if if_bpe:
                            # '<mask>' and ' <mask>'
                            if_add_prefix_space = not (ent_s == 0 and sent_id == 0)
                            _piece_len = len(tokenizer.tokenize(_ent_text, add_prefix_space=if_add_prefix_space))
                            _mask_text = ' '.join([tokenizer.mask_token] * _piece_len)
                            masked_piece_num += _piece_len
                        else:
                            _piece_len = len(tokenizer.tokenize(_ent_text))
                            masked_piece_num += _piece_len
                            _mask_text = ' '.join([tokenizer.mask_token] * _piece_len)

                        _text = _text + _mask_text
                        last_end = ent_e
                    else:
                        print(masked_ents)
                        print(replaced_ents)
                        raise RuntimeError()
                _text = _text + q_text[last_end:]

                q_sent['masked_text'] = _text

            re_rank_sentences = sorted(
                example['question'] + example['passage'], key=lambda x: x['index'])
            initial_text = ' '.join(
                map(lambda x: x['text'], re_rank_sentences))
            masked_text = ' '.join(map(
                lambda x: x['masked_text'] if 'masked_text' in x else x['text'], re_rank_sentences))

            initial_tokens = []
            masked_tokens = []
            for _sent_id, _sent in enumerate(re_rank_sentences):
                _initial_text = _sent['text']
                _masked_text = _sent['masked_text'] if 'masked_text' in _sent else _sent['text']
                initial_tokens.extend(tokenizer.tokenize(_initial_text, add_prefix_space=(_sent_id > 0)))
                masked_tokens.extend(tokenizer.tokenize(_masked_text, add_prefix_space=(_sent_id > 0)))
            
            if len(initial_tokens) != len(masked_tokens):
                continue

            initial_outputs = tokenizer.encode_plus(initial_tokens, padding='max_length',
                                                    truncation='only_first', max_length=max_seq_length,
                                                    return_tensors='pt')
            masked_outputs = tokenizer.encode_plus(masked_tokens, padding='max_length',
                                                    truncation='only_first', max_length=max_seq_length,
                                                    return_tensors='pt')

            # initial_outputs = tokenizer(initial_text, padding='max_length',
            #                             truncation='only_first', max_length=max_seq_length,
            #                             return_tensors='pt')
            # masked_outputs = tokenizer(masked_text, padding='max_length',
            #                            truncation='only_first', max_length=max_seq_length,
            #                            return_tensors='pt')

            assert initial_outputs['input_ids'].size(1) == masked_outputs['input_ids'].size(1)
            label_mask = initial_outputs['input_ids'] == masked_outputs['input_ids']

            all_input_ids.append(masked_outputs['input_ids'])
            all_attention_mask.append(masked_outputs['attention_mask'])
            if 'token_type_ids' in masked_outputs:
                all_token_type_ids.append(masked_outputs['token_type_ids'])
            else:
                all_token_type_ids.append(torch.Tensor([[0]]))
            label = initial_outputs['input_ids']
            label[label_mask] = -1
            all_labels.append(label)
            # print(all_input_ids[-1].size())
            
            # assert (label != -1).sum().item() == masked_piece_num, ((label != -1).sum().item(), masked_piece_num)

        all_input_ids = torch.cat(all_input_ids, dim=0)
        all_attention_mask = torch.cat(all_attention_mask, dim=0)
        all_token_type_ids = torch.cat(all_token_type_ids, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        all_tokens_num = (all_attention_mask == 1).sum().item()
        masked_tokens_num = (all_labels != -1).sum().item()
        ratio = masked_tokens_num * 1.0 / all_tokens_num

        all_feature_index = torch.arange(all_input_ids.size(0), dtype=torch.long)

        logger.info(f'Finished.')
        logger.info(f'Mask ratio: {masked_tokens_num} / {all_tokens_num} = {ratio}.')

        logger.info(f'input ids size: {all_input_ids.size()}')
        logger.info(f'labels size: {all_labels.size()}')

        return all_input_ids, all_token_type_ids, all_attention_mask, all_labels, all_feature_index

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
