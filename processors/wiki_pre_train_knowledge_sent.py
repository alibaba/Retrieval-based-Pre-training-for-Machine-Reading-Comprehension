import json
from typing import List, Dict

import nltk
import torch
from tqdm import tqdm
from transformers import BertTokenizer, PreTrainedTokenizer

from general_util import utils
from general_util.logger import get_child_logger

logger = get_child_logger(__name__)

"""
TODO:
- Discuss if add masked token prediction task.
- Text clean code. Following https://github.com/pytorch/text/blob/master/torchtext/datasets/unsupervised_learning.py
"""


class WikiKnowledgePreTrainProcessor(object):
    reader_name = 'wiki_sr_kd_s'

    r"""
    sr: sentence re-ordering task
    kd: knowledge
    s: sentence

    This model add extra [CLS] around each sentence and the [CLS] is adopted as the
    representation of each sentence. The position of [CLS] can be marked as the first
    token of each sentence: `sentence_spans[:, :, 0]`.

    Use torch.scatter_ to re-write the hidden state of [CLS] to the initial position.
    """

    def __init__(self, args):
        super(WikiKnowledgePreTrainProcessor, self).__init__()
        self.opt = vars(args)
        self.opt['cache_suffix'] = f'{self.reader_name}_{args.max_seq_length}'

    def read(self, input_file):
        logger.info('Reading data set from {}...'.format(input_file))

        with open(input_file, "r", encoding='utf-8') as reader:
            examples = json.load(reader)

        for ex in examples:
            if 'raw_entity' in ex:
                del ex['raw_entity']
            if 'raw_entity_lemma' in ex:
                del ex['raw_entity_lemma']

        logger.info('Finish reading {} examples from {}'.format(len(examples), input_file))
        return examples

    def convert_examples_to_tensors(self, examples: List, tokenizer: PreTrainedTokenizer):
        max_seq_length = self.opt['max_seq_length']

        unique_id = 1000000000
        features = []
        truncated = 0

        for (example_index, example) in tqdm(enumerate(examples), desc='Convert examples to features',
                                             total=len(examples), dynamic_ncols=True):

            q_tokens = []
            for sent_id, q_sent in enumerate(example['question']):
                """
                There are some bugs here. Sometimes the tokenized tokens are []. 
                Cleaner pre-processing method is needed.
                Currently we ignore the relevant answer (source and target)
                """
                wd_pieces = tokenizer.tokenize(q_sent['text'])
                # if len(wd_pieces) == 0:
                #     logger.warn(f"Found empty sentence in question: {q_sent['text']}")
                if len(wd_pieces) > 0:
                    wd_pieces = [tokenizer.cls_token] + wd_pieces
                    q_tokens.extend([(sent_id, t) for t in wd_pieces])

            q_sent_num = len(example['question'])

            d_tokens = []
            for sent_id, d_sent in enumerate(example['passage']):
                wd_pieces = tokenizer.tokenize(d_sent['text'])
                # if len(wd_pieces) == 0:
                #     logger.warn(f"Found empty sentence in passage: {d_sent['text']}")
                if len(wd_pieces) > 0:
                    wd_pieces = [tokenizer.cls_token] + wd_pieces
                    d_tokens.extend([(q_sent_num + sent_id, t) for t in wd_pieces])

            tmp = len(q_tokens) + len(d_tokens)

            utils.truncate_seq_pair(q_tokens, d_tokens, max_seq_length - 3)

            if len(q_tokens) + len(d_tokens) < tmp:
                # logger.warn("Truncation has occurred in example index: {}".format(str(example_index)))
                truncated += 1

            sent_idx_map = {}

            q_sentence_spans = []
            tokens = []
            # q_tokens = [(0, "[CLS]")] + q_tokens
            sent_s = 0
            for t_id, t in enumerate(q_tokens):
                tokens.append(t[1])
                if t_id == 0:
                    assert t[1] == tokenizer.cls_token
                    continue
                if t[0] != q_tokens[t_id - 1][0]:
                    q_sentence_spans.append((sent_s, t_id - 1))
                    sent_idx_map[q_tokens[t_id - 1][0]] = len(q_sentence_spans) - 1
                    sent_s = t_id
                    assert t[1] == tokenizer.cls_token, (t[1], tokenizer.cls_token)
            q_sentence_spans.append((sent_s, len(q_tokens) - 1))
            sent_idx_map[q_tokens[-1][0]] = len(q_sentence_spans) - 1
            # tokens.append("[SEP]")
            tokens.append(tokenizer.sep_token)
            segment_ids = [0] * len(tokens)

            p_sentence_spans = []
            sent_s = 0
            p_offset = len(tokens)
            for t_id, t in enumerate(d_tokens):
                tokens.append(t[1])
                if t_id == 0:
                    assert t[1] == tokenizer.cls_token
                    continue
                if t[0] != d_tokens[t_id - 1][0]:
                    p_sentence_spans.append((p_offset + sent_s, p_offset + t_id - 1))
                    sent_idx_map[d_tokens[t_id - 1][0]] = len(q_sentence_spans) + len(p_sentence_spans) - 1
                    sent_s = t_id
                    assert t[1] == tokenizer.cls_token, (t[1], tokenizer.cls_token)
            p_sentence_spans.append((p_offset + sent_s, p_offset + len(d_tokens) - 1))
            sent_idx_map[d_tokens[-1][0]] = len(q_sentence_spans) + len(p_sentence_spans) - 1
            # tokens.append("[SEP]")
            tokens.append(tokenizer.sep_token)
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

            sentence_num = len(q_sentence_spans) + len(p_sentence_spans)
            # edges = [[] for _ in range(sentence_num)]
            edge_index = []
            edges = []
            for idx1 in range(len(example['edges'])):
                if idx1 not in sent_idx_map:
                    # logger.warn(f'sentence index {idx1} is empty. Ignore it and its edges.')
                    continue
                tmp_edge = []
                # a = sent_idx_map[idx1]
                for idx2 in range(len(example['edges'][idx1])):
                    if idx2 not in sent_idx_map:
                        # logger.warn(f'sentence index {idx2} is empty. Ignore it and its edges.')
                        continue
                    # assert len(edges[a]) == sent_idx_map[idx2]
                    # edges[a].append(example['edges'][idx1][idx2])
                    if example['edges'][idx1][idx2] == 1:
                        tmp_edge.append(sent_idx_map[idx2])
                # assert len(edges[a]) == sentence_num, (len(edges[a]), sentence_num)
                if len(tmp_edge) > 0:
                    edge_index.append(sent_idx_map[idx1])
                    edges.append(tmp_edge)

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
                # "qas_id": example["qas_id"],
                "input_ids": input_ids,
                "segment_ids": segment_ids,
                "input_mask": input_mask,
                "answer": answer,
                "edge_index": edge_index,
                "edges": edges,
                "q_sentence_spans": q_sentence_spans,
                "p_sentence_spans": p_sentence_spans,
                "unique_id": unique_id
            })
            unique_id += 1

        logger.info(f'Generate {len(features)} features.')
        logger.info(f'Truncated features: {truncated} / {len(features)} = {truncated * 1.0 / len(features)}. ')
        logger.info(f'Start convert features to tensors.')
        all_tensors = self.data_to_tensors(features)
        logger.info(f'Finished.')

        return all_tensors

    @staticmethod
    def data_to_tensors(all_features: List[Dict]):
        all_input_ids = torch.LongTensor([f["input_ids"] for f in all_features])
        all_input_mask = torch.LongTensor([f["input_mask"] for f in all_features])
        all_segment_ids = torch.LongTensor([f["segment_ids"] for f in all_features])

        max_q_sentence_num = max(map(lambda x: len(x["q_sentence_spans"]), all_features))
        max_sentence_num = max(map(lambda x: len(x["q_sentence_spans"]) + len(x["p_sentence_spans"]), all_features))

        all_answers = torch.zeros((all_input_ids.size(0), max_q_sentence_num), dtype=torch.long).fill_(-1)
        for f_id, f in enumerate(all_features):
            all_answers[f_id, :len(f["answer"])] = torch.LongTensor(f["answer"])

        all_sentence_spans = torch.zeros((all_input_ids.size(0), max_sentence_num, 2), dtype=torch.long).fill_(-1)
        for f_id, f in enumerate(all_features):
            all_sentence_spans[f_id, :len(f["q_sentence_spans"])] = torch.LongTensor(f["q_sentence_spans"])
            sentence_num = len(f["q_sentence_spans"]) + len(f["p_sentence_spans"])
            all_sentence_spans[f_id, len(f["q_sentence_spans"]):sentence_num] = torch.LongTensor(f["p_sentence_spans"])

        max_edge_index = max(map(lambda x: len(x["edge_index"]), all_features))
        max_edge_num = 0
        for f in all_features:
            if not f["edges"]:
                continue
            max_edge_num = max(max_edge_num, max(map(lambda x: len(x), f["edges"])))

        all_edge_index = torch.zeros((all_input_ids.size(0), max_edge_index), dtype=torch.long).fill_(-1)
        all_edges = torch.zeros((all_input_ids.size(0), max_edge_index, max_edge_num), dtype=torch.long).fill_(-1)
        for f_id, f in enumerate(all_features):
            sentence_num = len(f["edge_index"])
            if sentence_num == 0:
                continue
            all_edge_index[f_id, :sentence_num] = torch.LongTensor(f["edge_index"])
            for sent_id, edges in enumerate(f["edges"]):
                all_edges[f_id, sent_id, :len(edges)] = torch.LongTensor(edges)
        # all_edges = torch.zeros((all_input_ids.size(0), max_sentence_num, max_sentence_num), dtype=torch.long)
        # for f_id, f in enumerate(all_features):
        #     sentence_num = len(f["edges"])
        #     all_edges[f_id, :sentence_num, :sentence_num] = torch.LongTensor(f["edges"])

        all_feature_index = torch.arange(all_input_ids.size(0), dtype=torch.long)

        return all_input_ids, all_segment_ids, all_input_mask, all_sentence_spans, all_answers, \
            all_edge_index, all_edges, all_feature_index

    def generate_inputs(self, batch, device):
        sentences_spans = batch[3].to(device)  # [batch, max_sentence_num, 2]
        answers = batch[4].to(device)  # [batch, max_query_sent_num]
        edge_index = batch[5].to(device)
        edges = batch[6].to(device)

        max_query_num = (answers > -1).sum(dim=1).max().item()
        answers = answers[:, :max_query_num]

        max_sentence_num = (sentences_spans[:, :, 0] > -1).sum(dim=1).max().item()
        sentences_spans = sentences_spans[:, :max_sentence_num]

        max_edge_index_num = (edge_index > -1).sum(dim=1).max().item()
        if max_edge_index_num == 0:
            edge_index = None
            edges = None
        else:
            edge_index = edge_index[:, :max_edge_index_num]
            edges = edges[:, :max_edge_index_num]
            max_edge_num = (edges > -1).sum(dim=2).max().item()
            assert max_edge_num > 0
            edges = edges[:, :, :max_edge_num]

        inputs = {
            'input_ids': batch[0].to(device),
            'attention_mask': batch[2].to(device),
            # 'sentence_spans': batch[3].to(device),
            # 'answers': batch[4].to(device),
            'sentence_spans': sentences_spans,
            'answers': answers,
            'edge_index': edge_index,
            'edges': edges
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
