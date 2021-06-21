import argparse
import collections
import json
import random
from multiprocessing import Pool
from typing import List
from functools import partial
import os

import nltk
import spacy
import utils
from tqdm import tqdm

r"""
This script taked the segmented wiki corpus file as input file:
[
    {
        "id": text_id,
        "article": [
            sentence_0,
            sentence_1,
            \cdots,
            sentence_k
        ]
    },
    ...
]
"""


def workflow(sentences: List[str], keep_prob=0.5,
             _ent_mask_p=0.5, _ent_rep_p=0.1,
             _noun_mask_p=0.5, _noun_rep_p=0.1,
             _pron_mask_p=0.2, _pron_rep_p=0.0):
    if len(sentences) < 0:
        return []

    # For definition, see: https://spacy.io/api/annotation#pos-tagging
    # Importance is defined as common counts.
    # collected_pos_list_edge = ['NOUN', 'PROPN']  # noun and proper noun
    collected_pos_list_edge = ['NOUN']
    # Importance is defined as self-counts
    collected_pos_list_self = ['PRON']  # pronoun

    # Extract entity and pattern for each sentence
    all_docs = nlp.pipe(sentences)
    all_ents = []
    all_nouns = []
    all_prons = []

    sent_token_num = []

    for doc in all_docs:
        ents = []
        for ent in doc.ents:
            ents.append((ent.text, ent.lemma_.lower(),
                         ent.start_char, ent.end_char))

        nns = []
        pns = []
        for token in doc:
            if token.pos_ in collected_pos_list_edge:
                nns.append((token.text, token.lemma_.lower(),
                            token.idx, token.idx + len(token.text)))
            elif token.pos_ in collected_pos_list_self:
                pns.append((token.text, token.lemma_.lower(),
                            token.idx, token.idx + len(token.text)))

        # Remove duplicates
        nns = utils.remove_duplicate_spans(ents, nns)
        pns = utils.remove_duplicate_spans(ents, pns)

        all_ents.append(ents)
        all_nouns.append(nns)
        all_prons.append(pns)

        sent_token_num.append(len(doc))

    sentences = [
        {'index': idx, 'sentence': sent, 'ent': ent, 'nn': nn, 'pn': pn}
        for idx, (sent, ent, nn, pn) in enumerate(zip(
            sentences, all_ents, all_nouns, all_prons))
    ]

    """
    Edge importance is defined as the number of edges for each end point.
    """
    sent2edge_set = []
    for sent_edge1, sent_edge2 in zip(all_ents, all_nouns):
        tmp = [_lemma for _, _lemma, _, _ in sent_edge1]
        tmp += [_lemma for _, _lemma, _, _ in sent_edge2]
        sent2edge_set.append(set(tmp))

    edge2sent_set = collections.defaultdict(set)
    for sent_idx, sent_edge in enumerate(sent2edge_set):
        for _lemma in sent_edge:
            edge2sent_set[_lemma].add(sent_idx)

    edge_num = {_lemma: len(edge2sent_set[_lemma])
                for _lemma in edge2sent_set.keys()}

    # Calculate importance for each sentence
    sent_edge_tuples = []
    for sent_idx, sent_edge in enumerate(sent2edge_set):
        _edge_tot_num = sum(map(lambda edge: edge_num[edge], sent_edge))
        # edge importance + count importance
        importance = (_edge_tot_num - 1) + len(sentences[sent_idx]['pn'])
        sent_edge_tuples.append((sent_idx, importance))
        sentences[sent_idx]['importance'] = importance

    # Sort all sentence index according their entities
    sorted_sent_edge_tuples = sorted(
        sent_edge_tuples, key=lambda x: x[1], reverse=True)
    noisy_sent_num = int((1. - keep_prob) * len(sentences))
    noisy_sent_ids = set(
        [idx for idx, _ in sorted_sent_edge_tuples[:noisy_sent_num]])
    if noisy_sent_num < 3:
        return []

    query_tot_token_num = sum([sent_token_num[idx] for idx in noisy_sent_ids])

    noisy_sents = []
    gold_sents = []
    for sent_id, sentence in enumerate(sentences):
        if sent_id in noisy_sent_ids:
            noisy_sents.append(sentence)
        else:
            gold_sents.append(sentence)

    raw_entities = []
    for sent in sentences:
        for ent_text, _, _, _ in sent['ent']:
            raw_entities.append(ent_text)
    raw_entities = list(set(raw_entities))

    raw_nouns = []
    for sent in sentences:
        for nn_text, _, _, _ in sent['nn']:
            raw_nouns.append(nn_text)
    raw_nouns = list(set(raw_nouns))

    raw_prons = []
    for sent in sentences:
        for pn_text, _, _, _ in sent['pn']:
            raw_prons.append(pn_text)
    raw_prons = list(set(raw_prons))

    # Add noise to selected sentences
    noised_token_num = 0
    for sent in noisy_sents:
        masked_ents = []
        replaced_ents = []

        # Mask entities
        for ent in sent['ent']:

            ent_lemma = ent[1]
            if edge_num[ent_lemma] <= 1:
                continue

            r = random.random()
            if r < _ent_mask_p:
                masked_ents.append((ent[2], ent[3]))
                noised_token_num += len(ent[0].split())
            elif r < _ent_mask_p + _ent_rep_p:
                new_ent = str(random.choice(raw_entities))
                replaced_ents.append((ent[2], ent[3], new_ent))
                noised_token_num += len(ent[0].split())

        # Mask nouns
        for nn in sent['nn']:

            nn_lemma = nn[1]
            if edge_num[nn_lemma] <= 1:
                continue

            r = random.random()
            if r < _noun_mask_p:
                masked_ents.append((nn[2], nn[3]))
                noised_token_num += len(nn[0].split())
            elif r < _noun_mask_p + _noun_rep_p:
                new_nn = str(random.choice(raw_nouns))
                replaced_ents.append((nn[2], nn[3], new_nn))
                noised_token_num += len(nn[0].split())

        # Mask pronouns
        for pn in sent['pn']:
            
            pn_lemma = pn[1]
            
            r = random.random()
            if r < _pron_mask_p:
                masked_ents.append((pn[2], pn[3]))
                noised_token_num += len(pn[0].split())
            elif r < _pron_mask_p + _pron_rep_p:
                new_pn = str(random.choice(raw_prons))
                replaced_ents.append((pn[2], pn[3], new_pn))
                noised_token_num += len(pn[0].split())

        sent['masked_ents'] = masked_ents
        sent['replaced_ents'] = replaced_ents

    random.shuffle(noisy_sents)

    idx_map = {s['index']: idx for idx,
                                   s in enumerate(noisy_sents + gold_sents)}
    sent_num = len(noisy_sents) + len(gold_sents)

    answer_matrix = []
    for ques_sent in noisy_sents:
        q_sent_id = ques_sent['index']
        next_id = q_sent_id + 1
        if next_id == sent_num:
            answer_matrix.append(-1)
        else:
            answer_matrix.append(idx_map[next_id])

    return {
        'ini_article': [sent['sentence'] for sent in sentences],
        'question': [{
            'index': sent['index'],
            'text': sent['sentence'],
            'masked_ents': sent['masked_ents'],
            'replaced_ents': sent['replaced_ents'],
            'importance': sent['importance']
        } for sent in noisy_sents],
        'passage': [{
            'index': sent['index'],
            'text': sent['sentence'],
            'importance': sent['importance']
        } for sent in gold_sents],
        'answer': answer_matrix,
        'noise_ratio': noised_token_num * 1.0 / query_tot_token_num
    }


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--keep_prob', default=0.5, type=float)
    parser.add_argument('--ratio', default='(0.8, 0.1, 0.3, 0.0, 0.3, 0.0)', type=str)
    # parser.add_argument('--token_noise_ratio', default=0.15, type=float)
    args = parser.parse_args()

    random.seed(args.seed)

    nlp = spacy.load('en_core_web_sm', disable=['parser', 'textcat'])
    nlp_no_ner = spacy.load('en_core_web_sm', disable=[
        'parser', 'textcat', 'ner'])
    # nlp.add_pipe(nlp.create_pipe('sentencizer'))
    nltk_stopwords = nltk.corpus.stopwords.words('english')

    # exec(f'ratio = {args.ratio}')
    ratio = eval(args.ratio)
    assert len(ratio) == 6
    ent_mask_p, ent_rep_p, noun_mask_p, noun_rep_p, pron_mask_p, pron_rep_p = ratio
    print(f'Entity: mask {ent_mask_p}, replace {ent_rep_p}\n'
          f'Noun: mask {noun_mask_p}, replace {noun_rep_p}\n'
          f'Pronoun: mask {pron_mask_p}, replace {pron_rep_p}.')

    examples = []
    noise_ratio_avg = 0
    data = json.load(open(args.input_file, 'r'))


    # def _func(instance):
    #     id = instance['id']
    #     sentences = instance['article']

    #     example = workflow(sentences, args.keep_prob,
    #                        _ent_mask_p=ent_mask_p, _ent_rep_p=ent_rep_p,
    #                        _noun_mask_p=noun_mask_p, _noun_rep_p=noun_rep_p,
    #                        _pron_mask_p=pron_mask_p, _pron_rep_p=pron_rep_p)
    #     if example:
    #         example['id'] = id
    #         return example
    #     return None


    # def _call_back(example):

    #     if example is not None:
    #         global noise_ratio_avg

    #         examples.append(example)
    #         noise_ratio_avg += example['noise_ratio']
    #         # print(len(examples) / len(data), end='\r', flush=True)

    #         if len(examples) % 50000 == 0:
    #             print(f'Processed {len(examples)} examples.')


    # pool = Pool()
    # for instance in tqdm(data, dynamic_ncols=True):
    #     pool.apply_async(_func, args=(instance,), callback=_call_back)

    all_ids = [inst['id'] for inst in data]
    all_article = [inst['article'] for inst in data]
    del data

    with Pool(os.cpu_count()) as p:
        annotate_ = partial(
            workflow,
            keep_prob=args.keep_prob,
            _ent_mask_p=ent_mask_p,
            _ent_rep_p=ent_rep_p,
            _noun_mask_p=noun_mask_p,
            _noun_rep_p=noun_rep_p,
            _pron_mask_p=pron_mask_p,
            _pron_rep_p=pron_rep_p
        )

        examples = list(
            tqdm(
                p.imap(annotate_, all_article, chunksize=32),
                total=len(all_article),
                desc='generate examples',
                disable=False
            )
        )

    new_examples = []
    for ex_id, ex in enumerate(examples):
        if ex:
            ex['id'] = all_ids[ex_id]
            noise_ratio_avg += ex['noise_ratio']
            new_examples.append(ex)
    del examples
    examples = new_examples

    noise_ratio_avg /= len(examples)
    print(len(examples))
    print(noise_ratio_avg)
    prefix = args.input_file[:-5]
    output_file = f'{prefix}_wk5_{args.keep_prob}_{ent_mask_p}_' \
                  f'{ent_rep_p}_{noun_mask_p}_{noun_rep_p}_' \
                  f'{pron_mask_p}_{pron_rep_p}_{args.seed}.json'

    with open(output_file, 'w') as f:
        
        json.dump(examples, f, ensure_ascii=False)

    print("===================== END ===========================")
