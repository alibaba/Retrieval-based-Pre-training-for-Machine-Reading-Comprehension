import json
import logging
import math
from collections import defaultdict, Counter
from typing import List, Dict

import torch
from tqdm import tqdm
from transformers import PreTrainedTokenizer, RobertaTokenizer, BartTokenizer, LongformerTokenizer

logger = logging.getLogger(__name__)

multi_sep_token_tokenizers = [RobertaTokenizer, BartTokenizer, LongformerTokenizer]
has_space_tokenizers = [RobertaTokenizer, BartTokenizer, LongformerTokenizer]


def if_need_space(_tokenizer):
    return any(isinstance(_tokenizer, _target) for _target in has_space_tokenizers)


def if_need_multi_sep(_tokenizer):
    return any(isinstance(_tokenizer, _target) for _target in multi_sep_token_tokenizers)


class Measures:

    @staticmethod
    def per_question_metrics(dataset, output_map):
        P = []
        R = []
        # for p in dataset:
        #     for qIdx, q in enumerate(p["paragraph"]["questions"]):
        #         id = p["id"] + "==" + str(qIdx)
        #         if (id in output_map):
        #             predictedAns = output_map.get(id)
        #             correctAns = [int(a["isAnswer"]) for a in q["answers"]]
        #             predictCount = sum(predictedAns)
        #             correctCount = sum(correctAns)
        #             assert math.ceil(sum(predictedAns)) == sum(predictedAns), "sum of the scores: " + str(sum(predictedAns))
        #             agreementCount = sum([a * b for (a, b) in zip(correctAns, predictedAns)])
        #             p1 = (1.0 * agreementCount / predictCount) if predictCount > 0.0 else 1.0
        #             r1 = (1.0 * agreementCount / correctCount) if correctCount > 0.0 else 1.0
        #             P.append(p1)
        #             R.append(r1)
        #         else:
        #             print("The id " + id + " not found . . . ")
        for article_id, article in dataset.items():
            for question_id, question in article.items():
                predicted_ans = Measures.get_sorted_list(output_map[article_id][question_id])
                correct_ans = Measures.get_sorted_list(question)
                predict_count = sum(predicted_ans)
                correct_count = sum(correct_ans)
                assert math.ceil(sum(predicted_ans)) == sum(predicted_ans), "sum of the scores" + str(
                    sum(predicted_ans))
                agreement_count = sum([a * b for (a, b) in zip(correct_ans, predicted_ans)])
                p1 = (1.0 * agreement_count / predict_count) if predict_count > 0.0 else 1.0
                r1 = (1.0 * agreement_count / correct_count) if correct_count > 0.0 else 1.0
                P.append(p1)
                R.append(r1)

        pAvg = Measures.avg(P)
        rAvg = Measures.avg(R)
        f1Avg = 2 * Measures.avg(R) * Measures.avg(P) / (Measures.avg(P) + Measures.avg(R))
        return [pAvg, rAvg, f1Avg]

    @staticmethod
    def exact_match_metrics(dataset, output_map, delta):
        EM = []
        # for p in dataset:
        #     for qIdx, q in enumerate(p["paragraph"]["questions"]):
        #         id = p["id"] + "==" + str(qIdx)
        #         if (id in output_map):
        #             predictedAns = output_map.get(id)
        #             correctAns = [int(a["isAnswer"]) for a in q["answers"]]
        #             em = 1.0 if sum([abs(i - j) for i, j in zip(correctAns, predictedAns)]) <= delta  else 0.0
        #             EM.append(em)
        #         else:
        #             print("The id " + id + " not found . . . ")
        for article_id, article in dataset.items():
            for question_id, question in article.items():
                predicted_ans = Measures.get_sorted_list(output_map[article_id][question_id])
                correct_ans = Measures.get_sorted_list(question)
                em = 1.0 if sum(abs(i - j) for i, j in zip(correct_ans, predicted_ans)) <= delta else 0.0
                EM.append(em)

        return Measures.avg(EM)

    @staticmethod
    def per_dataset_metric(dataset, output_map):
        agreementCount = 0
        correctCount = 0
        predictCount = 0
        # for p in dataset:
        #     for qIdx, q in enumerate(p["paragraph"]["questions"]):
        #         id = p["id"] + "==" + str(qIdx)
        #         if (id in output_map):
        #             predictedAns = output_map.get(id)
        #             correctAns = [int(a["isAnswer"]) for a in q["answers"]]
        #             predictCount += sum(predictedAns)
        #             correctCount += sum(correctAns)
        #             agreementCount += sum([a * b for (a, b) in zip(correctAns, predictedAns)])
        #         else:
        #             print("The id " + id + " not found . . . ")
        for article_id, article in dataset.items():
            for question_id, question in article.items():
                predict_ans = Measures.get_sorted_list(output_map[article_id][question_id])
                correct_ans = Measures.get_sorted_list(question)
                predictCount += sum(predict_ans)
                correctCount += sum(correct_ans)
                agreementCount += sum([a * b for (a, b) in zip(correct_ans, predict_ans)])

        p1 = (1.0 * agreementCount / predictCount) if predictCount > 0.0 else 1.0
        r1 = (1.0 * agreementCount / correctCount) if correctCount > 0.0 else 1.0
        return [p1, r1, 2 * r1 * p1 / (p1 + r1)]

    @staticmethod
    def avg(l):
        # return reduce(lambda x, y: x + y, l) / len(l)
        return 1.0 * sum(l) / len(l)

    @staticmethod
    def get_sorted_list(dic):
        sorted_dic = sorted(dic.items(), key=lambda x: x[0], reverse=False)
        return [x[1] for x in sorted_dic]


class MultiRCProcessor(object):
    reader_name = 'multi_rc'

    def __init__(self, args):
        super(MultiRCProcessor, self).__init__()
        self.opt = vars(args)
        self.opt['cache_suffix'] = f"{self.reader_name}_{args.base_model_type}_{args.max_seq_length}"

    def read(self, input_file):
        data = json.load(open(input_file, 'r'))

        examples = []
        for p_id, passage in enumerate(tqdm(data, desc='reading examples', dynamic_ncols=True)):
            article = ' '.join(passage['article'])
            questions = passage['questions']
            options = passage['options']
            answers = passage['answers']

            for q_id, (ques, op_list, ans_list) in enumerate(zip(questions, options, answers)):
                for op_id, (op, ans) in enumerate(zip(op_list, ans_list)):
                    examples.append({
                        "guid": f'{p_id}-{q_id}-{op_id}',
                        "text_a": ques + ' [SEP] ' + op,
                        "text_b": article,
                        "label": ans
                    })
        return examples

    def convert_examples_to_tensors(self, examples, tokenizer: PreTrainedTokenizer):
        max_seq_length = self.opt['max_seq_length']

        features = []

        text_a_list = [example["text_a"] for example in examples]
        text_b_list = [example["text_b"] for example in examples]
        label_list = [example["label"] for example in examples]
        encoded_output = tokenizer(text_a_list, text_b_list, padding='max_length',
                                   truncation='longest_first', max_length=max_seq_length,
                                   return_tensors='pt')
        input_ids = encoded_output['input_ids']
        token_type_ids = encoded_output['token_type_ids']
        attention_mask = encoded_output['attention_mask']
        labels = torch.LongTensor(label_list)
        index = torch.arange(input_ids.size(0), dtype=torch.long)

        assert input_ids.size(1) == token_type_ids.size(1) == attention_mask.size(1) == max_seq_length

        return input_ids, token_type_ids, attention_mask, labels, index

    def generate_inputs(self, batch, device):
        inputs = {
            "input_ids": batch[0].to(device),
            "attention_mask": batch[2].to(device),
            "labels": batch[3].to(device)
        }
        if self.opt['base_model_type'] in ['bert']:
            inputs["token_type_ids"] = batch[1].to(device)
        return inputs

    def compute_metrics(self, pred, labels, examples):
        label_dict = {}
        pred_dict = {}
        for idx, (p, l, example) in enumerate(zip(pred, labels, examples)):
            p_id, q_id, op_id = example["guid"].split('-')
            assert l == example['label']
            if p_id not in label_dict:
                label_dict[p_id] = defaultdict(dict)
                pred_dict[p_id] = defaultdict(dict)
            label_dict[p_id][q_id][op_id] = l
            pred_dict[p_id][q_id][op_id] = p

        return {
            'f1_m': Measures.per_question_metrics(label_dict, pred_dict)[-1],
            'f1_a': Measures.per_dataset_metric(label_dict, pred_dict)[-1],
            'em0': Measures.exact_match_metrics(label_dict, pred_dict, 0),
            'em1': Measures.exact_match_metrics(label_dict, pred_dict, 1)
        }


class MultiRCSentenceProcessor(object):
    reader_name = 'multi_rc_sent'

    def __init__(self, args):
        super(MultiRCSentenceProcessor, self).__init__()
        self.opt = vars(args)
        self.opt['cache_suffix'] = f"{self.reader_name}_{args.base_model_type}_" \
                                   f"{args.max_seq_length}_{args.max_query_length}"

    @staticmethod
    def read(input_file):
        data = json.load(open(input_file, 'r'))

        examples = []
        for p_id, passage in enumerate(tqdm(data, desc='reading examples', dynamic_ncols=True)):
            article: List[str] = passage['article']
            questions = passage['questions']
            options = passage['options']
            answers = passage['answers']
            evidences = passage['evidences']

            for q_id, (ques, op_list, ans_list, evi_list) in enumerate(zip(questions, options, answers, evidences)):
                for op_id, (op, ans) in enumerate(zip(op_list, ans_list)):
                    examples.append({
                        "guid": f'{p_id}-{q_id}-{op_id}',
                        "article": article,
                        "question": ques,
                        "option": op,
                        "label": ans,
                        "evidence": evi_list
                    })
        return examples

    def convert_examples_to_tensors(self, examples, tokenizer: PreTrainedTokenizer):
        max_seq_length = self.opt['max_seq_length']
        max_query_length = self.opt['max_query_length']
        need_space = if_need_space(tokenizer)
        need_multi_sep = if_need_multi_sep(tokenizer)
        sequence_pair_added_tokens = tokenizer.model_max_length - tokenizer.max_len_sentences_pair
        logger.info(f'Converting args: {max_seq_length}, {max_query_length}, {need_space}, '
                    f'{need_multi_sep}, {sequence_pair_added_tokens}')

        features = []

        for example_index, example in enumerate(tqdm(examples, desc='converting examples to features...',
                                                     dynamic_ncols=True, total=len(examples))):
            question: str = example['question']
            passage: List[str] = example['article']
            option: str = example['option']
            label: int = example['label']

            q_tokens = tokenizer.tokenize(question)
            o_tokens = tokenizer.tokenize(option)

            q_token_tuples = [(0, _t) for _t in q_tokens]
            o_token_tuples = [(1, _t) for _t in o_tokens]

            _sep_tuple = (0, tokenizer.sep_token)

            _query_tokens_to_remove = len(q_token_tuples) + len(o_token_tuples) + (
                2 if need_multi_sep else 1) - max_query_length

            q_token_tuples, o_token_tuples, _ = tokenizer.truncate_sequences(
                ids=q_token_tuples,
                pair_ids=o_token_tuples,
                num_tokens_to_remove=_query_tokens_to_remove,
                truncation_strategy="longest_first"
            )

            sep_token = [_sep_tuple] * 2 if need_multi_sep else [_sep_tuple]
            q_op_token_tuples = q_token_tuples + sep_token + o_token_tuples

            _sent_id_offset = 2
            p_token_tuples = []
            for _sent_id, _p_sent in enumerate(passage):
                if need_space and _sent_id > 0:
                    _tokens = tokenizer.tokenize(_p_sent, add_prefix_space=True)
                else:
                    _tokens = tokenizer.tokenize(_p_sent)

                p_token_tuples.extend([(_sent_id + _sent_id_offset, _t) for _t in _tokens])

            _lens_to_remove = len(q_op_token_tuples) + len(p_token_tuples) + sequence_pair_added_tokens - max_seq_length

            _q_op_tuples, _p_tuples, _ = tokenizer.truncate_sequences(q_op_token_tuples,
                                                                      pair_ids=p_token_tuples,
                                                                      num_tokens_to_remove=_lens_to_remove,
                                                                      truncation_strategy='longest_first')

            def _get_sentence_spans(_token_tuple, _start_offset, _sent_id_map: Dict, _sentence_spans: List):
                ini_sent_ids, ini_tokens = zip(*_token_tuple)
                ini_sent_ids = Counter(ini_sent_ids)

                for ini_s_id, s_len in ini_sent_ids.items():
                    _new_t_s = _start_offset
                    _new_t_e = _start_offset + s_len - 1

                    _start_offset = _new_t_e + 1

                    _sent_id_map[ini_s_id] = len(_sentence_spans)

                    if ini_s_id == 0:
                        # hack to remove <sep> between question and option
                        # but keep the `_start_offset` consistent with previous.
                        _new_t_e -= (2 if need_multi_sep else 1)
                        assert _new_t_e > _new_t_s

                    _sentence_spans.append((_new_t_s, _new_t_e))

                return ini_tokens

            sentence_spans = []
            sent_id_map = {}
            start_offset = 1

            q_op_tokens = _get_sentence_spans(_q_op_tuples, start_offset, sent_id_map, sentence_spans)

            # start_offset += 2 if need_multi_sep else 1
            start_offset = start_offset + len(q_op_tokens) + (2 if need_multi_sep else 1)

            p_tokens = _get_sentence_spans(_p_tuples, start_offset, sent_id_map, sentence_spans)

            tokenizer_outputs = tokenizer.encode_plus(
                q_op_tokens,
                text_pair=p_tokens,
                padding='max_length',
                max_length=max_seq_length
            )

            input_ids = tokenizer_outputs['input_ids']
            token_type_ids = tokenizer_outputs['token_type_ids'] if 'token_type_ids' in tokenizer_outputs \
                else [0]
            attention_mask = tokenizer_outputs['attention_mask']

            features.append({
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'token_type_ids': token_type_ids,
                'label': label,
                'sentence_spans': sentence_spans
            })

        logger.info(f'Convert {len(features)} features.')
        logger.info(f'Start converting features into tensors.')

        all_input_ids = torch.tensor([f['input_ids'] for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f['token_type_ids'] for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f['attention_mask'] for f in features], dtype=torch.long)
        all_labels = torch.tensor([f['label'] for f in features], dtype=torch.long)

        max_sent_num = max(map(lambda x: len(x['sentence_spans']), features))
        data_num = all_input_ids.size(0)

        all_sentence_spans = torch.zeros(data_num, max_sent_num, 2, dtype=torch.long).fill_(-1)
        all_feature_index = torch.arange(data_num, dtype=torch.long)

        for f_id, f in enumerate(features):
            all_sentence_spans[f_id, :len(f['sentence_spans'])] = torch.tensor(f['sentence_spans'], dtype=torch.long)

        logger.info(f'Converting tensors finished.')

        logger.info(f'all_input_ids size: {all_input_ids.size()}')
        logger.info(f'all_token_type_ids size: {all_token_type_ids.size()}')
        logger.info(f'all_attention_mask size: {all_attention_mask.size()}')
        logger.info(f'all_sentence_spans size: {all_sentence_spans.size()}')
        logger.info(f'all_labels size: {all_labels.size()}')

        return all_input_ids, all_token_type_ids, all_attention_mask, \
            all_sentence_spans, all_labels, all_feature_index

    @staticmethod
    def collate_fn(batch):
        input_ids, token_type_ids, attention_mask, \
            sentence_spans, labels, feature_index = list(zip(*batch))

        input_ids = torch.stack(input_ids, dim=0)
        token_type_ids = torch.stack(token_type_ids, dim=0)
        attention_mask = torch.stack(attention_mask, dim=0)
        labels = torch.stack(labels, dim=0)
        sentence_spans = torch.stack(sentence_spans, dim=0)
        feature_index = torch.stack(feature_index, dim=0)

        max_sent_num = (sentence_spans[:, :, 0] > -1).sum(dim=1).max().item()
        sentence_spans = sentence_spans[:, :max_sent_num]

        max_sent_len = (sentence_spans[:, :, 1] - sentence_spans[:, :, 0] + 1).max().item()

        batch = sentence_spans.size(0)
        sentence_index = torch.zeros(batch, max_sent_num, max_sent_len, dtype=torch.long)
        sentence_mask = torch.ones(batch, max_sent_num)
        sent_word_mask = torch.ones(batch, max_sent_num, max_sent_len)

        for b in range(batch):
            for s_id, span in enumerate(sentence_spans[b]):
                s, e = span[0].item(), span[1].item()
                if s == -1 and e == -1:
                    break
                if s == 0:
                    if e == 0:
                        continue
                    else:
                        s += 1
                lens = e - s + 1
                sentence_index[b, s_id, :lens] = torch.arange(s, e + 1, dtype=torch.long)
                sentence_mask[b, s_id] = 0
                sent_word_mask[b, s_id, :lens] = torch.zeros(lens)

        return input_ids, token_type_ids, attention_mask, sentence_index, sentence_mask, sent_word_mask, \
            labels, feature_index

    def generate_inputs(self, batch, device):
        inputs = {
            "input_ids": batch[0].to(device),
            "attention_mask": batch[2].to(device),
            "sentence_index": batch[3].to(device),
            "sentence_mask": batch[4].to(device),
            "sent_word_mask": batch[5].to(device),
            "labels": batch[6].to(device)
        }
        if self.opt['base_model_type'] in ['bert']:
            inputs["token_type_ids"] = batch[1].to(device)
        return inputs

    @staticmethod
    def compute_metrics(pred, labels, examples):
        label_dict = {}
        pred_dict = {}
        for idx, (p, l, example) in enumerate(zip(pred, labels, examples)):
            p_id, q_id, op_id = example["guid"].split('-')
            assert l == example['label']
            if p_id not in label_dict:
                label_dict[p_id] = defaultdict(dict)
                pred_dict[p_id] = defaultdict(dict)
            label_dict[p_id][q_id][op_id] = l
            pred_dict[p_id][q_id][op_id] = p

        return {
            'f1_m': Measures.per_question_metrics(label_dict, pred_dict)[-1],
            'f1_a': Measures.per_dataset_metric(label_dict, pred_dict)[-1],
            'em0': Measures.exact_match_metrics(label_dict, pred_dict, 0),
            'em1': Measures.exact_match_metrics(label_dict, pred_dict, 1)
        }

    @staticmethod
    def write_predictions(pred, labels, examples, predicted_tensors: Dict[str, List] = None):
        predictions = {}
        sentence_logits = predicted_tensors["sentence_logits"] if predicted_tensors else None
        sent_word_ids = predicted_tensors["sent_word_ids"] if predicted_tensors else None
        for idx, (p, l, example) in enumerate(zip(pred, labels, examples)):
            p_id, q_id, op_id = example["guid"].split('-')
            assert l == example['label']
            if p_id not in predictions:
                predictions[p_id] = defaultdict(dict)
            predictions[p_id][q_id][op_id] = {
                "label": l,
                "pred": p,
                "evidences": example["evidence"],
            }
            if sentence_logits is not None:
                predictions[p_id][q_id][op_id]["sentence_logits"] = sentence_logits[idx]
                predictions[p_id][q_id][op_id]["sent_word_ids"] = sent_word_ids[idx]

        return predictions
