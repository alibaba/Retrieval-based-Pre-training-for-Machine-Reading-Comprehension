import json
import random
from collections import Counter
from typing import List, Tuple, Dict

import nltk
import torch
from tqdm import tqdm
from transformers import BertTokenizer

from data import ModelState
from general_util import utils
from general_util.logger import get_child_logger

# bert_tokenizer = BertTokenizer.from_pretrained('/home/admin/workspace/bert-base-uncased')

logger = get_child_logger(__name__)

"""
TODO:
- Discuss if add masked token prediction task.
- Text clean code. Following https://github.com/pytorch/text/blob/master/torchtext/datasets/unsupervised_learning.py
"""

lemmar = nltk.stem.WordNetLemmatizer()


def get_lemmar(x):
    return lemmar.lemmatize(x)


def workflow3(sentences, keep_prob=0.5, mask_prob=0.4, replace_prob=0.1, random_sample=False):
    # clean sentences
    sentences = [sent for sent in sentences if sent.replace(' ', '') != '']

    sentences = [(sent_id, sent) for sent_id, sent in enumerate(sentences)]

    if len(sentences) < 8:
        return []

    sample = {
        'cause': sentences[:2],
        'result': sentences[-2:]
    }

    gold_sents = sentences[2:-2]
    blank_sents = []
    candi_sents_ini = []

    if len(gold_sents) >= 4:
        for sent in gold_sents:
            r = random.random()
            if r < keep_prob:
                blank_sents.append(sent)
            else:
                candi_sents_ini.append(sent)
    else:
        candi_sents_ini = gold_sents

    if len(candi_sents_ini) < 3:
        return []

    entities = Counter()

    for sent in (sample['cause'] + sample['result'] + blank_sents):
        sent_tokens = nltk.word_tokenize(sent[1])
        sent_tokens_pos_tags = nltk.pos_tag(sent_tokens)
        for token in sent_tokens_pos_tags:
            if 'NN' in token[1]:
                entities[get_lemmar(token[0])] += 1

    if random_sample:
        random.shuffle(candi_sents_ini)
    else:
        tmp = []
        s = 0
        e = len(candi_sents_ini) - 1
        while len(tmp) < len(candi_sents_ini):
            r = random.random()
            if r < 0.5:
                tmp.append(candi_sents_ini[s])
                s += 1
            else:
                tmp.append(candi_sents_ini[e])
                e -= 1
        candi_sents_ini = tmp

    candi_sents = []
    for sent in candi_sents_ini:
        sent_tokens = nltk.word_tokenize(sent[1])
        sent_tokens_pos_tags = nltk.pos_tag(sent_tokens)
        cand_tokens = []
        for token in sent_tokens_pos_tags:
            if 'NN' in token[1]:
                ent = get_lemmar(token[0])
                if ent in entities:
                    r = random.random()
                    if r < mask_prob:
                        num = len(bert_tokenizer.tokenize(token[0]))
                        cand_tokens += ['[MASK]'] * num
                    elif r < mask_prob + replace_prob:
                        cand_tokens.append(
                            random.choice(list(entities.keys())))  # lemmar of some entity or the initial form?
                    else:
                        cand_tokens.append(token[0])  # ent or token[0]?
                else:
                    cand_tokens.append(token[0])  # ent or token[0]?
                entities[ent] += 1
            else:
                cand_tokens.append(token[0])
        candi_sents.append((sent[0], ' '.join(cand_tokens)))
    random.shuffle(candi_sents)

    sample['guide_sents'] = blank_sents
    sample['candidate_sents'] = candi_sents
    sample['gold_sents'] = gold_sents

    question = candi_sents
    passage = sample['cause'] + blank_sents + sample['result']
    sent_num = len(question) + len(passage)
    answer_matrix = []
    for ques_sent in question:
        q_sent_id = ques_sent[0]
        find = False
        for new_sent_id_q, ot_q_sent in enumerate(question):
            if q_sent_id + 1 == ot_q_sent[0]:
                find = True
                answer_matrix.append(new_sent_id_q)
                break
        if find:
            continue
        for new_sent_id_p, pass_sent in enumerate(passage):
            p_sent_id = pass_sent[0]
            if q_sent_id + 1 == p_sent_id:
                answer_matrix.append(len(question) + new_sent_id_p)
                break
    assert len(answer_matrix) == len(question)
    for answer_id in answer_matrix:
        assert answer_id < len(question) + len(passage)
    question = [q[1] for q in question]
    passage = [p[1] for p in passage]

    if question and passage and answer_matrix:
        return question, passage, answer_matrix
    else:
        return []


class WikiPreTrainProcessor(object):
    reader_name = 'wiki_pre_train'

    def __init__(self, args):
        super(WikiPreTrainProcessor, self).__init__()
        self.opt = vars(args)
        self.opt['cache_suffix'] = f'{self.reader_name}_{args.keep_prob}_{args.mask_prob}_{args.replace_prob}'

    def read(self, input_file):
        keep_prob = self.opt['keep_prob']
        mask_prob = self.opt['mask_prob']
        replace_prob = self.opt['replace_prob']
        logger.info('Reading data set from {}...'.format(input_file))
        logger.info(f'Reading options:\tkeep_prob: {keep_prob},\tmask_prob: {mask_prob},\treplace_prob: {replace_prob}')

        with open(input_file, "r", encoding='utf-8') as reader:
            input_data = json.load(reader)

        def is_whitespace(ch):
            if ch == " " or ch == "\t" or ch == "\r" or ch == "\n" or ord(ch) == 0x202F:
                return True
            return False

        examples = []
        for instance in tqdm(input_data, dynamic_ncols=True):
            sentences = instance['article']
            result = workflow3(sentences, keep_prob=keep_prob, mask_prob=mask_prob, replace_prob=replace_prob)
            if not result:
                continue
            question, passage, answer_id = result
            article_id = instance['id']

            examples.append({
                'qas_id': article_id,
                'question': question,
                'passage': passage,
                'answer': answer_id
            })

        logger.info('Finish reading {} examples from {}'.format(len(examples), input_file))
        return examples

    def convert_examples_to_features(self, examples: List, tokenizer):
        max_seq_length = self.opt['max_seq_length']

        unique_id = 1000000000
        features = []
        for (example_index, example) in tqdm(enumerate(examples), desc='Convert examples to features',
                                             total=len(examples), dynamic_ncols=True):

            q_tokens = []
            for sent_id, q_sent in enumerate(example['question']):
                """
                There are some bugs here. Sometimes the tokenized tokens are []. Cleaner pre-processing method is needed.
                Currently we ignore the relevant answer (source and target)
                """
                wd_pieces = tokenizer.tokenize(q_sent)
                q_tokens.extend([(sent_id, t) for t in wd_pieces])
            q_sent_num = len(example['question'])

            d_tokens = []
            for sent_id, d_sent in enumerate(example['passage']):
                wd_pieces = tokenizer.tokenize(d_sent)
                d_tokens.extend([(q_sent_num + sent_id, t) for t in wd_pieces])

            utils.truncate_seq_pair(q_tokens, d_tokens, max_seq_length - 3)

            sent_idx_map = {}

            q_sentence_spans = []
            # tokens = ["[CLS]"]
            tokens = []
            q_tokens = [(0, "[CLS]")] + q_tokens
            sent_s = 0
            for t_id, t in enumerate(q_tokens):
                tokens.append(t[1])
                if t_id == 0:
                    continue
                if t[0] != q_tokens[t_id - 1][0]:
                    q_sentence_spans.append((sent_s, t_id - 1))
                    sent_idx_map[q_tokens[t_id - 1][0]] = len(q_sentence_spans) - 1
                    sent_s = t_id
            q_sentence_spans.append((sent_s, len(q_tokens) - 1))
            sent_idx_map[q_tokens[-1][0]] = len(q_sentence_spans) - 1
            tokens.append("[SEP]")
            segment_ids = [0] * len(tokens)

            p_sentence_spans = []
            sent_s = 0
            p_offset = len(tokens)
            for t_id, t in enumerate(d_tokens):
                tokens.append(t[1])
                if t_id == 0:
                    continue
                if t[0] != d_tokens[t_id - 1][0]:
                    p_sentence_spans.append((p_offset + sent_s, p_offset + t_id - 1))
                    sent_idx_map[d_tokens[t_id - 1][0]] = len(q_sentence_spans) + len(p_sentence_spans) - 1
                    sent_s = t_id
            p_sentence_spans.append((p_offset + sent_s, p_offset + len(d_tokens) - 1))
            sent_idx_map[d_tokens[-1][0]] = len(q_sentence_spans) + len(p_sentence_spans) - 1
            tokens.append("[SEP]")
            segment_ids += [1] * (len(d_tokens) + 1)

            answer = []
            for q_id, tgt in enumerate(example['answer']):
                # if q_id >= len(q_sentence_spans):
                # break
                if q_id not in sent_idx_map:
                    continue
                assert sent_idx_map[q_id] == len(answer), (q_id, sent_idx_map[q_id], len(answer))

                if tgt not in sent_idx_map:
                    answer.append(-1)
                else:
                    answer.append(sent_idx_map[tgt])

            for x in answer:
                assert x == -1 or 0 <= x < len(q_sentence_spans) + len(p_sentence_spans)
            assert len(answer) == len(q_sentence_spans), (
                len(answer), len(q_sentence_spans), example['answer'], q_tokens)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            features.append({
                "qas_id": example["qas_id"],
                "input_ids": input_ids,
                "segment_ids": segment_ids,
                "input_mask": input_mask,
                "answer": answer,
                "q_sentence_spans": q_sentence_spans,
                "p_sentence_spans": p_sentence_spans,
                "unique_id": unique_id
            })
            unique_id += 1

        logger.info(f'Reading {len(features)} features.')

        return features

    def write_predictions(self):
        raise NotImplementedError

    @staticmethod
    def get_metric(examples, all_predictions):
        raise NotImplementedError

    def data_to_tensors(self, all_features):
        # Possible should be merged with data_to_tensors_sent_pretrain method.
        raise NotImplementedError

    @staticmethod
    def data_to_tensors_sent_pretrain(all_features: List[Dict]):
        """
        TODO: convert sentence spans to tensor instead of list for further distributed training. 
        """
        all_input_ids = torch.LongTensor([f["input_ids"] for f in all_features])
        all_input_mask = torch.LongTensor([f["input_mask"] for f in all_features])
        all_segment_ids = torch.LongTensor([f["segment_ids"] for f in all_features])

        max_q_sentence_num = max(map(lambda x: len(x["q_sentence_spans"]), all_features))
        max_p_sentence_num = max(map(lambda x: len(x["p_sentence_spans"]), all_features))

        all_answers = torch.zeros((all_input_ids.size(0), max_q_sentence_num), dtype=torch.long).fill_(-1)
        for f_id, f in enumerate(all_features):
            all_answers[f_id, :len(f["answer"])] = torch.LongTensor(f["answer"])

        all_feature_index = torch.arange(all_input_ids.size(0), dtype=torch.long)

        return all_features, (all_input_ids, all_segment_ids, all_input_mask, all_answers, all_feature_index)

    @staticmethod
    def generate_inputs(batch: Tuple, all_features: List[Dict], model_state):
        """
        TODO: remove this method. For further explanation, see data_to_tensors_sent_pretrain method.
        """
        assert model_state in ModelState
        feature_index = batch[-1].tolist()
        batch_features = [all_features[index] for index in feature_index]

        q_sentence_span_list = []
        p_sentence_span_list = []

        # For convenience
        for feature in batch_features:
            q_sentence_span_list.append(feature["q_sentence_spans"])
            p_sentence_span_list.append(feature['p_sentence_spans'])

        inputs = {
            "input_ids": batch[0],
            "token_type_ids": batch[1],
            "attention_mask": batch[2],
            "q_sentence_spans": q_sentence_span_list,
            "p_sentence_spans": p_sentence_span_list
        }
        if model_state == ModelState.Test:
            return inputs
        elif model_state == ModelState.Evaluate:
            inputs["answers"] = batch[3]
        elif model_state == ModelState.Train:
            inputs["answers"] = batch[3]

        return inputs
