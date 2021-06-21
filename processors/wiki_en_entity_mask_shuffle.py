import argparse
import collections
import json
import random
from collections import Counter
from typing import List

import nlp
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

parser = argparse.ArgumentParser()
parser.add_argument('--input_file', type=str, required=True)
parser.add_argument('--work_flow_id', type=int, default=0)
parser.add_argument('--seed', type=int, default=42)

parser.add_argument('--keep_prob', default=0.5, type=float)
parser.add_argument('--mask_prob', default=0.4, type=float)
parser.add_argument('--replace_prob', default=0.1, type=float)
parser.add_argument('--incommon_prob', default=0.2, type=float)
args = parser.parse_args()

random.seed(args.seed)

nlp = spacy.load('en_core_web_sm', disable=['parser', 'textcat'])
nlp_no_ner = spacy.load('en_core_web_sm', disable=['parser', 'textcat', 'ner'])
# nlp.add_pipe(nlp.create_pipe('sentencizer'))
nltk_stopwords = nltk.corpus.stopwords.words('english')


def workflow_0(sentences: List[str], keep_prob=0.5, mask_prob=0.4, replace_prob=0.1, random_sample=False):
    # pre-process
    if len(sentences) < 8:
        return []

    # Extract entity and pattern for each sentence
    all_docs = nlp.pipe(sentences)
    all_ents = []

    for doc in all_docs:
        ents = []
        for ent in doc.ents:
            ents.append((ent.text, ent.lemma_.lower(), ent.start_char, ent.end_char))

        all_ents.append(ents)

    sentences = [
        {'index': idx, 'sentence': sent, 'ent': ent}
        for idx, (sent, ent) in enumerate(zip(
            sentences, all_ents
        ))
    ]

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

    raw_entity = Counter()
    raw_entity_lemma = Counter()

    for sent in (sample['cause'] + sample['result'] + blank_sents):
        raw_entity += Counter([ent[0].lower() for ent in sent['ent']])
        raw_entity_lemma += Counter([ent[1] for ent in sent['ent']])

    # Change the reasoning order of candidate sentences
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

    # Add noise to query sentences and shuffle them
    for sent in candi_sents_ini:

        masked_ents = []
        replaced_ents = []

        for ent in sent['ent']:

            ent_text = ent[0]
            raw_ent = ent_text.lower()
            raw_ent_lemma = ent[1]

            if raw_ent in raw_entity or raw_ent_lemma in raw_entity_lemma:
                r = random.random()
                if r < mask_prob:
                    masked_ents.append((ent[2], ent[3]))
                elif r < mask_prob + replace_prob:
                    new_ent = str({random.choice(list(raw_entity.keys()))})
                    replaced_ents.append((ent[2], ent[3], new_ent))

            raw_entity[raw_ent] += 1
            raw_entity_lemma[raw_ent_lemma] += 1

        sent['masked_ents'] = masked_ents
        sent['replaced_ents'] = replaced_ents

    random.shuffle(candi_sents_ini)

    # Generate sentence reorder labels
    question = candi_sents_ini
    passage = sample['cause'] + blank_sents + sample['result']
    idx_map = {s['index']: idx for idx, s in enumerate(question + passage)}
    sent_num = len(question) + len(passage)

    answer_matrix = []
    for ques_sent in question:
        q_sent_id = ques_sent['index']
        next_id = q_sent_id + 1
        answer_matrix.append(idx_map[next_id])

    return {
        'ini_article': [sent['sentence'] for sent in sentences],
        'question': [{
            'index': sent['index'],
            'text': sent['sentence'],
            'masked_ents': sent['masked_ents'],
            'replaced_ents': sent['replaced_ents']
        } for sent in question],
        'passage': [{
            'index': sent['index'],
            'text': sent['sentence']
        } for sent in passage],
        'answer': answer_matrix,
        # 'entity_pairs': entity_pairs,
        # 'raw_entity': list(raw_entity),
        # 'raw_entity_lemma': list(raw_entity_lemma)
    }


def workflow_1(sentences: List[str], keep_prob=0.5, mask_prob=0.4, replace_prob=0.1,
               incommon_prob=0.2, random_sample=False):
    # pre-process
    if len(sentences) < 8:
        return []

    # Extract entity and pattern for each sentence
    all_docs = nlp.pipe(sentences)
    all_ents = []

    for doc in all_docs:
        ents = []
        for ent in doc.ents:
            ents.append((ent.text, ent.lemma_.lower(), ent.start_char, ent.end_char))

        all_ents.append(ents)

    sentences = [
        {'index': idx, 'sentence': sent, 'ent': ent}
        for idx, (sent, ent) in enumerate(zip(
            sentences, all_ents
        ))
    ]

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

    raw_entity = Counter()
    raw_entity_lemma = Counter()

    for sent in (sample['cause'] + sample['result'] + blank_sents):
        raw_entity += Counter([ent[0].lower() for ent in sent['ent']])
        raw_entity_lemma += Counter([ent[1] for ent in sent['ent']])

    # Change the reasoning order of candidate sentences
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

    # Add noise to query sentences and shuffle them
    for sent in candi_sents_ini:

        masked_ents = []
        replaced_ents = []

        for ent in sent['ent']:

            ent_text = ent[0]
            raw_ent = ent_text.lower()
            raw_ent_lemma = ent[1]

            if raw_ent in raw_entity or raw_ent_lemma in raw_entity_lemma:
                r = random.random()
                if r < mask_prob:
                    masked_ents.append((ent[2], ent[3]))
                elif r < mask_prob + replace_prob:
                    new_ent = str({random.choice(list(raw_entity.keys()))})
                    replaced_ents.append((ent[2], ent[3], new_ent))
            else:
                r = random.random()
                if r < incommon_prob:
                    masked_ents.append((ent[2], ent[3]))

            raw_entity[raw_ent] += 1
            raw_entity_lemma[raw_ent_lemma] += 1

        sent['masked_ents'] = masked_ents
        sent['replaced_ents'] = replaced_ents

    random.shuffle(candi_sents_ini)

    # Generate sentence reorder labels
    question = candi_sents_ini
    passage = sample['cause'] + blank_sents + sample['result']
    idx_map = {s['index']: idx for idx, s in enumerate(question + passage)}
    sent_num = len(question) + len(passage)

    answer_matrix = []
    for ques_sent in question:
        q_sent_id = ques_sent['index']
        next_id = q_sent_id + 1
        answer_matrix.append(idx_map[next_id])

    sent_next = []
    for pass_sent in passage:
        p_sent_id = pass_sent['index']
        next_id = p_sent_id + 1
        if next_id == sent_num:
            sent_next.append(-1)
        else:
            sent_next.append(idx_map[next_id])

    return {
        'ini_article': [sent['sentence'] for sent in sentences],
        'question': [{
            'index': sent['index'],
            'text': sent['sentence'],
            'masked_ents': sent['masked_ents'],
            'replaced_ents': sent['replaced_ents']
        } for sent in question],
        'passage': [{
            'index': sent['index'],
            'text': sent['sentence']
        } for sent in passage],
        'answer': answer_matrix,
        'sent_next': sent_next
        # 'entity_pairs': entity_pairs,
        # 'raw_entity': list(raw_entity),
        # 'raw_entity_lemma': list(raw_entity_lemma)
    }


def workflow_2(sentences: List[str], keep_prob=0.5, mask_prob=0.4, replace_prob=0.1):
    r"""
    For each entity in each sentence, the importance of the entity is defined as the total number
    of times that it is included in other sentences (multi-times in single sentence is seen as once).
    The the importance of each sentence is defined as the sum of the importance of the entities
    occurred in the sentence.
    """

    if len(sentences) < 0:
        return []

    # Extract entity and pattern for each sentence
    all_docs = nlp.pipe(sentences)
    all_ents = []

    for doc in all_docs:
        ents = []
        for ent in doc.ents:
            ents.append((ent.text, ent.lemma_.lower(), ent.start_char, ent.end_char))

        all_ents.append(ents)

    sentences = [
        {'index': idx, 'sentence': sent, 'ent': ent}
        for idx, (sent, ent) in enumerate(zip(sentences, all_ents))
    ]

    assert len(all_ents) == len(sentences)

    sent2ent_set = []
    for sent_ents in all_ents:
        tmp = [ent_lemma_low for _, ent_lemma_low, _, _ in sent_ents]
        sent2ent_set.append(set(tmp))

    ent2sent_set = collections.defaultdict(set)
    for sent_idx, sent_ents in enumerate(sent2ent_set):
        for ent_lemma in sent_ents:
            ent2sent_set[ent_lemma].add(sent_idx)

    ent_edge_num = {ent_lem: len(ent2sent_set[ent_lem]) for ent_lem in ent2sent_set.keys()}

    sent_edge_tuples = []
    for sent_idx, sent_ents in enumerate(sent2ent_set):
        _edge_tot_num = sum(map(lambda ent: ent_edge_num[ent], sent_ents))
        sent_edge_tuples.append((sent_idx, _edge_tot_num - 1))

    # Sort all sentence index according their entities
    sorted_sent_edge_tuples = sorted(sent_edge_tuples, key=lambda x: x[1], reverse=True)
    noisy_sent_num = int((1. - keep_prob) * len(sentences))
    noisy_sent_ids = set([idx for idx, _ in sorted_sent_edge_tuples[:noisy_sent_num]])
    if noisy_sent_num < 3:
        return []

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

    # Add noise to selected sentences
    for sent in noisy_sents:
        masked_ents = []
        replaced_ents = []

        for ent in sent['ent']:

            ent_lemma = ent[1]
            if ent_edge_num[ent_lemma] <= 1:
                continue

            r = random.random()
            if r < mask_prob:
                masked_ents.append((ent[2], ent[3]))
            elif r < mask_prob + replace_prob:
                new_ent = str(random.choice(raw_entities))
                replaced_ents.append((ent[2], ent[3], new_ent))

        sent['masked_ents'] = masked_ents
        sent['replaced_ents'] = replaced_ents

    random.shuffle(noisy_sents)

    idx_map = {s['index']: idx for idx, s in enumerate(noisy_sents + gold_sents)}
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
            'replaced_ents': sent['replaced_ents']
        } for sent in noisy_sents],
        'passage': [{
            'index': sent['index'],
            'text': sent['sentence']
        } for sent in gold_sents],
        'answer': answer_matrix
    }


def workflow_3(sentences: List[str], keep_prob=0.5, mask_prob=0.4, replace_prob=0.1):
    if len(sentences) < 0:
        return []

    # For definition, see: https://spacy.io/api/annotation#pos-tagging
    # Importance is defined as common counts.
    collected_pos_list_edge = ['NOUN', 'PROPN']  # noun and proper noun
    # Importance is defined as self-counts
    collected_pos_list_self = ['PRON']  # pronoun

    # Extract entity and pattern for each sentence
    all_docs = nlp.pipe(sentences)
    all_ents = []
    all_pos_self = []

    for doc in all_docs:
        ents = []
        for ent in doc.ents:
            ents.append((ent.text, ent.lemma_.lower(), ent.start_char, ent.end_char))

        nns = []
        pns = []
        for token in doc:
            if token.pos_ in collected_pos_list_edge:
                nns.append((token.text, token.lemma_.lower(), token.idx, token.idx + len(token.text)))
            elif token.pos_ in collected_pos_list_self:
                pns.append((token.text, token.lemma_.lower(), token.idx, token.idx + len(token.text)))

        # Remove duplicates
        nns = utils.remove_duplicate_spans(ents, nns)
        pns = utils.remove_duplicate_spans(ents, pns)
        ents.extend(nns)

        all_ents.append(ents)
        all_pos_self.append(pns)

    sentences = [
        {'index': idx, 'sentence': sent, 'ent': ent, 'count': pn}
        for idx, (sent, ent, pn) in enumerate(zip(sentences, all_ents, all_pos_self))
    ]

    assert len(all_ents) == len(sentences)

    sent2ent_set = []
    for sent_ents in all_ents:
        tmp = [ent_lemma_low for _, ent_lemma_low, _, _ in sent_ents]
        sent2ent_set.append(set(tmp))

    ent2sent_set = collections.defaultdict(set)
    for sent_idx, sent_ents in enumerate(sent2ent_set):
        for ent_lemma in sent_ents:
            ent2sent_set[ent_lemma].add(sent_idx)

    ent_edge_num = {ent_lem: len(ent2sent_set[ent_lem]) for ent_lem in ent2sent_set.keys()}

    # Calculate importance for each sentence
    sent_edge_tuples = []
    for sent_idx, sent_ents in enumerate(sent2ent_set):
        _edge_tot_num = sum(map(lambda ent: ent_edge_num[ent], sent_ents))
        importance = _edge_tot_num - 1 + len(sentences[sent_idx]['count'])
        sent_edge_tuples.append((sent_idx, importance))
        sentences[sent_idx]['importance'] = importance

    # Sort all sentence index according their entities
    sorted_sent_edge_tuples = sorted(sent_edge_tuples, key=lambda x: x[1], reverse=True)
    noisy_sent_num = int((1. - keep_prob) * len(sentences))
    noisy_sent_ids = set([idx for idx, _ in sorted_sent_edge_tuples[:noisy_sent_num]])
    if noisy_sent_num < 3:
        return []

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

    raw_pronouns = []
    for sent in sentences:
        for pn_text, _, _, _ in sent['count']:
            raw_pronouns.append(pn_text)
    raw_pronouns = list(set(raw_pronouns))

    # Add noise to selected sentences
    for sent in noisy_sents:
        masked_ents = []
        replaced_ents = []

        # Mask entities / nouns / proper nouns
        for ent in sent['ent']:

            ent_lemma = ent[1]
            if ent_edge_num[ent_lemma] <= 1:
                continue

            r = random.random()
            if r < mask_prob:
                masked_ents.append((ent[2], ent[3]))
            elif r < mask_prob + replace_prob:
                new_ent = str(random.choice(raw_entities))
                replaced_ents.append((ent[2], ent[3], new_ent))

        # Mask pronouns
        for pn in sent['count']:

            pn_lemma = pn[1]

            r = random.random()
            if r < mask_prob:
                masked_ents.append((pn[2], pn[3]))
            elif r < mask_prob + replace_prob:
                new_pn = str(random.choice(raw_pronouns))
                replaced_ents.append((pn[2], pn[3], new_pn))

        sent['masked_ents'] = masked_ents
        sent['replaced_ents'] = replaced_ents

    random.shuffle(noisy_sents)

    idx_map = {s['index']: idx for idx, s in enumerate(noisy_sents + gold_sents)}
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
        'answer': answer_matrix
    }


data = json.load(open(args.input_file, 'r'))
workflow = {
    0: workflow_0,
    1: workflow_1,
    2: workflow_2,
    3: workflow_3
}[args.work_flow_id]

examples = []
for instance in tqdm(data, dynamic_ncols=True):
    id = instance['id']
    sentences = instance['article']

    example = workflow(sentences, args.keep_prob,
                       args.mask_prob, args.replace_prob)
    if example:
        example['id'] = id
        examples.append(example)

print(len(examples))
if args.work_flow_id == 0:
    output_file = f'{args.input_file}_{args.keep_prob}_{args.mask_prob}_' \
                  f'{args.replace_prob}_{args.seed}.json'
elif args.work_flow_id == 1:
    output_file = f'{args.input_file}_wk{args.work_flow_id}_{args.keep_prob}_' \
                  f'{args.mask_prob}_{args.replace_prob}_{args.incommon_prob}_{args.seed}.json'
elif args.work_flow_id in [2, 3]:
    output_file = f'{args.input_file}_wk{args.work_flow_id}_{args.keep_prob}_' \
                  f'{args.mask_prob}_{args.replace_prob}_{args.seed}.json'
else:
    raise RuntimeError()

with open(output_file, 'w') as f:
    json.dump(examples, f, ensure_ascii=False)
