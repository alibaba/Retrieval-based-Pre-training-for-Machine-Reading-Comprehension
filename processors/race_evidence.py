import json
import logging
from collections import defaultdict, Counter

import torch
from nltk import sent_tokenize
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from processors.utils import evidence_pipeline, combine_sentence

logger = logging.getLogger(__name__)


class RACEProcessor(object):
    reader_name = 'race_evidence'

    def __init__(self, args):
        super(RACEProcessor, self).__init__()
        self.opt = vars(args)
        self.opt['cache_suffix'] = f"{self.reader_name}_{args.base_model_type}_" \
                                   f"{args.max_seq_length}_{args.max_query_length}"

    def read(self, input_file):
        data = json.load(open(input_file, 'r', encoding='utf-8'))

        examples = []
        for instance in tqdm(data, desc='reading examples', dynamic_ncols=True):
            passage = instance['article']
            article_id = instance['id']

            questions = instance['questions']
            answers = list(map(lambda x: {'A': 0, 'B': 1, 'C': 2, 'D': 3}[x], instance['answers']))
            options = instance['options']

            for q_id, (question, answer, option_list) in enumerate(zip(questions, answers, options)):
                qas_id = f"{article_id}-{q_id}"
                examples.append({
                    'qas_id': qas_id,
                    'passage': passage,
                    'question': question,
                    'answer': answer,
                    'options': option_list
                })
        return examples

    def convert_examples_to_tensors(self, examples, tokenizer: PreTrainedTokenizer):
        max_seq_length = self.opt['max_seq_length']
        max_query_length = self.opt['max_query_length']
        if_bpe = 'roberta' in self.opt['base_model_type']

        features = []
        for example_index, example in enumerate(
                tqdm(examples, desc='convert examples to features...', dynamic_ncols=True)):
            question = example['question']
            passage = example['passage']
            options = example['options']

            # q_sentences = sent_tokenize(question)
            q_sentences = [question]
            if not isinstance(passage, list):
                p_sentences = combine_sentence(sent_tokenize(passage))
            else:
                p_sentences = passage

            if if_bpe:
                q_sentences = [_q if q_id == 0 else ' ' + _q for q_id, _q in q_sentences]
                p_sentences = [_p if p_id == 0 else ' ' + _p for p_id, _p in p_sentences]

            q_raw_tokens = [tokenizer.tokenize(q) for q in q_sentences]
            p_raw_tokens = [tokenizer.tokenize(p) for p in p_sentences]
            q_token_tuple = [(idx, t) for idx, tokens in enumerate(q_raw_tokens) for t in tokens]

            option_features = []
            for op_id, option in enumerate(options):
                sent_offset = len(q_sentences)

                # op_sentences = sent_tokenize(option)
                op_sentences = [option]
                if if_bpe:
                    op_sentences = [_op if op_id == 0 else ' ' + _op for op_id, op in enumerate(op_sentences)]
                op_raw_tokens = [tokenizer.tokenize(op) for op in op_sentences]
                op_token_tuple = [(sent_offset + idx, t) for idx, tokens in enumerate(op_raw_tokens) for t in tokens]

                sent_offset = len(q_sentences) + len(op_sentences)
                p_token_tuple = [(sent_offset + idx, t) for idx, tokens in enumerate(p_raw_tokens) for t in tokens]

                sent_offset = sent_offset + len(p_sentences)
                # edges = pattern_pipeline(q_sentences + op_sentences + p_sentences, split_sent=False)
                evidences = evidence_pipeline(q_sentences + op_sentences, p_sentences)

                q_op_token_tuple = q_token_tuple + [(q_token_tuple[-1][0], tokenizer.sep_token)] + op_token_tuple

                q_op_lens_to_remove = len(q_op_token_tuple) - max_query_length
                q_op_token_tuple, _, _ = tokenizer.truncate_sequences(q_op_token_tuple,
                                                                      num_tokens_to_remove=q_op_lens_to_remove,
                                                                      truncation_strategy='only_first')

                if if_bpe:
                    lens_to_remove = len(q_op_token_tuple) + len(p_token_tuple) + 4 - max_seq_length
                else:
                    lens_to_remove = len(q_op_token_tuple) + len(p_token_tuple) + 3 - max_seq_length

                e_token_tuple = []
                edges = defaultdict(list)
                if evidences:
                    for evi_id, (evi, a, b) in enumerate(evidences):
                        if evi_id > 0 and if_bpe:
                            evi = ' ' + evi
                        evi_tokens = tokenizer.tokenize(evi)
                        if evi_id == 0:
                            evi_tokens = [tokenizer.sep_token] + evi_tokens
                        if lens_to_remove + len(evi_tokens) <= 0:
                            lens_to_remove += evi_tokens
                            sent_id = sent_offset + evi_id
                            e_token_tuple.extend([(sent_id, t) for t in evi_tokens])
                            # edges.append((sent_id, a))
                            # edges.append((sent_id, b))
                            edges[sent_id].append(a)
                            edges[sent_id].append(b)
                        else:
                            break
                    p_token_tuple.extend(e_token_tuple)

                _q_op_tuple, _p_tuple, _ = tokenizer.truncate_sequences(q_op_token_tuple,
                                                                        p_token_tuple, lens_to_remove,
                                                                        "longest_first")

                def get_sentence_spans(token_tuple, _start_offset, _sent_id_map, _sentence_spans):
                    ini_sent_ids, ini_tokens = zip(*token_tuple)
                    ini_sent_ids = Counter(ini_sent_ids)
                    sorted_ini_sent_ids = sorted(ini_sent_ids.items(), key=lambda x: x[0])

                    _start_offset = _start_offset
                    for ini_s_id, s_len in sorted_ini_sent_ids:
                        _new_t_s = _start_offset
                        _new_t_e = _start_offset + s_len - 1
                        _start_offset = _new_t_e + 1
                        _sent_id_map[ini_s_id] = len(_sentence_spans)
                        _sentence_spans.append((_new_t_s, _new_t_e))

                    return ini_tokens

                sentence_spans = []
                sent_id_map = {}
                start_offset = 1  # offset for [CLS]
                # question and option
                q_op_tokens = get_sentence_spans(_q_op_tuple, start_offset, sent_id_map, sentence_spans)

                if if_bpe:
                    start_offset = len(q_op_tokens) + 3  # <s> and </s></s>
                else:
                    start_offset = len(q_op_tokens) + 2  # [CLS] and [SEP]    

                # passage
                p_tokens = get_sentence_spans(_p_tuple, start_offset, sent_id_map, sentence_spans)

                tokenizer_outputs = tokenizer.encode_plus(q_op_tokens, text_pair=p_tokens, padding='max_length',
                                                          max_length=max_seq_length)

                input_ids = tokenizer_outputs["input_ids"]
                token_type_ids = tokenizer_outputs["token_type_ids"] if 'token_type_ids' in tokenizer_outputs else [0]
                attention_mask = tokenizer_outputs["attention_mask"]

                edge_src = []
                edge_tgt = []
                for edge in edges:
                    edge_src.append(edge)
                    edge_tgt.append(edges[edge])
                    assert len(edges[edge]) == 2

                option_features.append({
                    'input_ids': input_ids,
                    'token_type_ids': token_type_ids,
                    'attention_mask': attention_mask,
                    'sentence_spans': sentence_spans,
                    'edge_src': edge_src,
                    'edge_tgt': edge_tgt
                })

            features.append({
                'op_features': option_features,
                'label': example['answer']
            })

        logger.info(f'Convert {len(features)} features.')
        logger.info(f'Start converting features into tensors...')

        all_input_ids, all_token_type_ids, all_attention_mask, all_labels = [], [], [], []

        max_sent_num = 0
        max_edge_src_num = 0
        # max_edge_tgt_num = 0
        for op_features in features:
            all_labels.append(op_features['label'])
            ops = op_features['op_features']
            all_input_ids.append([f['input_ids'] for f in ops])
            all_token_type_ids.append([f['token_type_ids'] for f in ops])
            all_attention_mask.append([f['attention_mask'] for f in ops])
            max_sent_num = max(max_sent_num, max(map(lambda x: len(x['sentence_spans']), ops)))
            max_edge_src_num = max(max_edge_src_num, max(map(lambda x: len(x['edge_src']), ops)))
            # max_edge_tgt_num = max(max_edge_tgt_num, max(map(get_len, ops)))

        all_input_ids = torch.LongTensor(all_input_ids)
        all_token_type_ids = torch.LongTensor(all_token_type_ids)
        all_attention_mask = torch.LongTensor(all_attention_mask, dim=0)
        all_labels = torch.LongTensor(all_labels)

        data_num, num_choice, _ = all_input_ids.size()
        all_sentence_spans = torch.zeros(data_num, num_choice, max_sent_num, 2, dtype=torch.long).fill_(-1)
        all_edge_src = torch.zeros(data_num, num_choice, max_edge_src_num, dtype=torch.long).fill_(-1)
        all_edge_tgt = torch.zeros(data_num, num_choice, max_edge_src_num, 2, dtype=torch.long).fill_(-1)

        for f_id, op_features in enumerate(features):
            for op_id, op_f in enumerate(op_features['op_features']):
                _sent_num = len(op_f['sentence_spans'])
                all_sentence_spans[f_id, op_id, :_sent_num] = torch.LongTensor(op_f['sentence_spans'])

                _edge_src_num = len(op_f['edge_src'])
                all_edge_src[f_id, op_id, :_edge_src_num] = torch.LongTensor(op_f['edge_src'])
                all_edge_tgt[f_id, op_id, :_edge_src_num] = torch.LongTensor(op_f['edge_tgt'])

        all_feature_index = torch.arange(data_num, dtype=torch.long)

        logger.info(f'Converting tensors finished.')

        logger.info(f'all_input_ids size: {all_input_ids.size()}')
        logger.info(f'all_token_type_ids size: {all_token_type_ids.size()}')
        logger.info(f'all_attention_mask size: {all_attention_mask.size()}')
        logger.info(f'all_sentence_spans size: {all_sentence_spans.size()}')
        logger.info(f'all_edge_src size: {all_edge_src.size()}')
        logger.info(f'all_edge_tgt size: {all_edge_tgt.size()}')
        logger.info(f'all_labels size: {all_labels.size()}')

        return all_input_ids, all_token_type_ids, all_attention_mask, \
            all_sentence_spans, all_edge_src, all_edge_tgt, \
            all_labels, all_feature_index

    @staticmethod
    def collate_fn(batch):
        input_ids, token_type_ids, attention_mask, \
            sentence_spans, edge_src, edge_tgt, \
            labels, feature_index = list(zip(*batch))

        input_ids = torch.stack(input_ids, dim=0)
        token_type_ids = torch.stack(token_type_ids, dim=0)
        attention_mask = torch.stack(attention_mask, dim=0)
        labels = torch.stack(labels, dim=0)
        feature_index = torch.stack(feature_index, dim=0)
        sentence_spans = torch.stack(sentence_spans)

        batch, num_choice, max_seq_len = input_ids.size()

        max_sent_num = (sentence_spans[:, :, :, 0] > -1).sum(dim=2).max().item()
        sentence_spans = sentence_spans[:, :, :max_sent_num]

        max_sent_len = (sentence_spans[:, :, :, 1] - sentence_spans[:, :, :, 0] + 1).max().item()

        sentence_index = torch.zeros(batch, num_choice, max_sent_num, max_sent_len, dtype=torch.long)
        sentence_mask = torch.ones(batch, num_choice, max_sent_num)
        sent_word_mask = torch.ones(batch, num_choice, max_sent_num, max_sent_len)

        for b in range(batch):
            for c in range(num_choice):
                for sid, span in enumerate(sentence_spans[b][c]):
                    s, e = span[0].item(), span[1].item()
                    if s == -1 and e == -1:
                        break
                    if s == 0:
                        if e == 0:
                            continue
                        else:
                            s += 1
                    lens = e - s + 1
                    sentence_index[b, c, sid, :lens] = torch.arange(s, e + 1, dtype=torch.long)
                    sentence_mask[b, c, sid] = 0
                    sent_word_mask[b, c, sid, :lens] = torch.zeros(lens)

        max_src_num = (edge_src > -1).sum(dim=2).max().item()
        if max_src_num:
            edge_src = edge_src[:, :, :max_src_num]
            edge_tgt = edge_tgt[:, :, :max_src_num]

            ex_attn_mask = attention_mask.unsqueeze(2).expand(-1, -1, max_seq_len, -1)
            ex_sent_mask = sentence_mask.unsqueeze(2).expand(-1, -1, max_sent_num, -1)

            for b in range(batch):
                for c in range(num_choice):
                    for src_id, src in enumerate(edge_tgt[b, c]):
                        if src == -1:
                            break
                        tgt = edge_tgt[b, c, src_id].tolist()
                        ex_sent_mask[b, c, :, src] = torch.ones(max_sent_num)
                        ex_sent_mask[b, c, src, :] = torch.ones(max_sent_num)
                        ex_sent_mask[b, c, src, tgt] = 0
                        ex_sent_mask[b, c, tgt, src] = 0
                        ex_sent_mask[b, c, src, src] = 0

                        src_s, src_e = sentence_spans[b, c, src]
                        src_e += 1
                        lens = src_e - src_s
                        # tgt_s, tgt_e = sentence_spans[b, c, tgt]  # tensor d=2
                        ex_attn_mask[b, c, src_s: src_e, :] = torch.zeros(lens, max_seq_len)
                        ex_attn_mask[b, c, :, src_s: src_e] = torch.zeros(max_seq_len, lens)
                        ex_attn_mask[b, c, src_s: src_e, src_s: src_e] = torch.ones(lens, lens)
                        for _t in tgt:
                            _t_s, _t_e = sentence_spans[b, c, _t]
                            _t_e += 1
                            _lens = _t_e - _t_s
                            ex_attn_mask[b, c, _t_s: _t_e, src_s: src_e] = torch.ones(_lens, lens)
                            ex_attn_mask[b, c, src_s: src_e, _t_s: _t_e] = torch.ones(lens, _lens)

            attention_mask = ex_attn_mask
            sentence_mask = ex_sent_mask

        return input_ids, token_type_ids, attention_mask, \
            sentence_index, sentence_mask, sent_word_mask, \
            labels, feature_index

    def generate_inputs(self, batch, device):
        sentence_index = batch[3].to(device)
        sentence_mask = batch[4].to(device)
        sent_word_mask = batch[5].to(device)

        inputs = {
            "input_ids": batch[0].to(device),
            "attention_mask": batch[2].to(device),
            "sentence_mask": sentence_mask,
            "sentence_index": sentence_index,
            "sent_word_mask": sent_word_mask,
            "labels": batch[6].to(device)
        }
        if self.opt['base_model_type'] in ['bert']:
            inputs["token_type_ids"] = batch[1].to(device)
        return inputs

    def compute_metrics(self, pred, labels, examples):
        acc = torch.sum((torch.LongTensor(pred) == torch.LongTensor(labels))).item() * 1.0 / len(pred)

        return {
            'accuracy': acc
        }
