import argparse
import json
import random
from collections import Counter
from typing import List

import nltk
import spacy
from tqdm import tqdm
from transformers import AutoTokenizer

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
parser.add_argument('--conceptnet', type=str, default=None)
parser.add_argument('--concept_pattern', type=str, default=None)
parser.add_argument('--tf_tokenizer_name',
                    default='bert-base-uncased', type=str)
parser.add_argument('--seed', type=int, default=42)

parser.add_argument('--keep_prob', default=0.5, type=float)
parser.add_argument('--mask_prob', default=0.4, type=float)
parser.add_argument('--replace_prob', default=0.1, type=float)
args = parser.parse_args()

random.seed(args.seed)

nlp = spacy.load('en_core_web_sm', disable=['parser', 'textcat'])
nlp_no_ner = spacy.load('en_core_web_sm', disable=['parser', 'textcat', 'ner'])
# nlp.add_pipe(nlp.create_pipe('sentencizer'))
nltk_stopwords = nltk.corpus.stopwords.words('english')
transformer_tokenizer = AutoTokenizer.from_pretrained(args.tf_tokenizer_name)

# process ConceptNet
# lemma_vocab = {}

# concept_vocab = {}
# concepts = collections.defaultdict(lambda: collections.defaultdict(list))
# with open(args.conceptnet, 'r') as f:
#     for line in tqdm(f.readlines(), desc='Reading conceptNet', dynamic_ncols=True):
#         x = line.strip().split('\t')
#         r, a, b, w = x

#         # if a not in lemma_vocab:
#         #     lemma_vocab[a] = lemmatize(a)
#         # a = lemma_vocab[a]

#         # if b not in lemma_vocab:
#         #     lemma_vocab[b] = lemmatize(b)
#         # b = lemma_vocab[b]

#         if a not in concept_vocab:
#             pattern = create_pattern(nlp(a.replace('_', ' ')))
#             concept_vocab[a] = pattern
#         a = concept_vocab[a]

#         if b not in concept_vocab:
#             pattern = create_pattern(nlp(b.replace('_', ' ')))
#             concept_vocab[b] = pattern
#         b = concept_vocab[b]

#         concepts[a][b].append((r, float(w)))

# with open('concept_pattern.json', 'w') as f:
#     json.dump(concepts, f)


concepts = json.load(open(args.concept_pattern, 'r'))


def create_pattern(doc):
    pronoun_list = {"my", "you", "it", "its", "your", "i", "he", "she", "his", "her", "they", "them", "their", "our",
                    "we"}
    # Filtering concepts consisting of all stop words and longer than four words.
    if len(doc) >= 5 or doc[0].text in pronoun_list or doc[-1].text in pronoun_list or all(
            [(token.text in nltk_stopwords or token.lemma_ in nltk_stopwords) for token in doc]):  #
        return None  # ignore this concept as pattern

    # pattern = []
    # for token in doc:  # a doc is a concept
    #     pattern.append({"LEMMA": token.lemma_})
    # return pattern
    pattern = '_'.join([token.lemma_ for token in doc])
    return pattern


def lemmatize(concept):
    doc = nlp(concept.replace("_", " "))
    # lcs = set()
    # lcs.add("_".join([token.lemma_ for token in doc])) # all lemma
    # return lcs
    return "_".join([token.lemma_ for token in doc])


def remove_extra_space(s):
    return ' '.join(s.split())


def replace_segment(s, start_char, end_char, replace_ment: str):
    new_s = s[:start_char] + replace_ment + s[end_char:]
    offset = len(new_s) - len(s)
    return new_s, offset


def pre_process_sentence(sentences: List[str]):
    sent_ents = []
    sent_ent_docs = []
    sent_ent_doc_sec = []
    all_docs = nlp.pipe(sentences, batch_size=100)
    for doc in tqdm(all_docs, total=len(sentences), desc='pipeline for sentences'):
        ls_a = []
        ents = []
        for ent in doc.ents:
            if ent.label_ in ['CARDINAL']:
                continue
            ents.append(ent)
            ls_a.append(ent.text.lower())

        _s = len(sent_ent_docs)
        sent_ent_docs.extend(ls_a)
        _e = len(sent_ent_docs)
        # sent_ent_docs.append(nlp_no_ner.pipe(ls_a, batch_size=100))
        sent_ent_doc_sec.append((_s, _e))
        sent_ents.append(ents)

    all_ent_docs = nlp_no_ner.pipe(sent_ent_docs, batch_size=300)
    all_ent_docs = [x for x in tqdm(all_ent_docs, total=len(
        sent_ent_docs), desc='pipeline for all entities')]

    sentences = [
        {'index': sent_id, 'sentence': sent, 'ents': ents,
         'ent_docs': all_ent_docs[ent_doc_sec[0]:ent_doc_sec[1]]}
        for sent_id, (sent, ents, ent_doc_sec) in enumerate(zip(
            sentences, sent_ents, sent_ent_doc_sec))
    ]

    return sentences


def workflow_0(sentences: List[str], keep_prob=0.5, mask_prob=0.4, replace_prob=0.1, max_hop=1, random_sample=False):
    # pre-process
    sentences = [remove_extra_space(sent) for sent in sentences if sent.strip() != '']
    if len(sentences) < 8:
        return []

    # Extract entity and pattern for each sentence
    all_docs = nlp.pipe(sentences)
    all_ents = []
    all_ent_docs = []
    all_ent_doc_secs = []
    for doc in all_docs:
        ents = []
        ls_a = []
        for ent in doc.ents:
            # if ent.label_ in ['CARDINAL']:
            #     continue
            ents.append((ent.text, ent.lemma_.lower(), ent.start_char, ent.end_char))
            ls_a.append(ent.text.lower())

        _s = len(all_ent_docs)
        all_ent_docs.extend(ls_a)
        _e = len(all_ent_docs)
        all_ent_doc_secs.append((_s, _e))
        all_ents.append(ents)

    all_ent_docs = nlp_no_ner.pipe(all_ent_docs)
    all_ent_docs = [x for x in all_ent_docs]
    all_ent_patterns = []
    for _s, _e in all_ent_doc_secs:
        tmp = []
        for x in all_ent_docs[_s:_e]:
            ptn = create_pattern(x)
            if ptn and ptn in concepts:
                tmp.append(ptn)
        all_ent_patterns.append(set(tmp))

    sentences = [
        {'index': idx, 'sentence': sent, 'ent': ent, 'pattern': ptn}
        for idx, (sent, ent, ptn) in enumerate(zip(
            sentences, all_ents, all_ent_patterns
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

    # entity_sent_ls = collections.defaultdict(set)
    sent_pattern_sets = {}

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
        new_sent_text = ''
        last_start = 0

        for ent in sent['ent']:

            ent_text = ent[0]
            raw_ent = ent_text.lower()
            raw_ent_lemma = ent[1]

            new_sent_text += sent['sentence'][last_start:ent[2]]
            last_start = ent[3]

            if raw_ent in raw_entity or raw_ent_lemma in raw_entity_lemma:
                r = random.random()
                if r < mask_prob:
                    num = len(transformer_tokenizer.tokenize(raw_ent))
                    rep = ' '.join(['[MASK]'] * num)
                    new_sent_text += rep
                    # new_sent_text += f' {"[MASK]"} '
                elif r < mask_prob + replace_prob:
                    new_sent_text += str({random.choice(list(raw_entity.keys()))})
                else:
                    new_sent_text += ent_text
            else:
                new_sent_text += ent_text

            raw_entity[raw_ent] += 1
            raw_entity_lemma[raw_ent_lemma] += 1

        new_sent_text += sent['sentence'][last_start:]
        sent['noisy_sentence'] = new_sent_text

        # Add all relevant patterns to the set of each sentence
        # TODO: Considering mask all concepts. But it's a time-consuming operation.
        tmp = set()
        for ptn in sent['pattern']:
            tmp.update(concepts[ptn].keys())
        sent_pattern_sets[sent['index']] = tmp

    random.shuffle(candi_sents_ini)

    # Generate sentence reorder labels
    question = candi_sents_ini
    passage = sample['cause'] + blank_sents + sample['result']
    idx_map = {s['index']: idx for idx, s in enumerate(question + passage)}
    # q_idx_map = {s['index']: idx for idx, s in enumerate(question)}
    # p_idx_map = {s['index']: idx for idx, s in enumerate(passage)}
    sent_num = len(question) + len(passage)

    answer_matrix = []
    for ques_sent in question:
        q_sent_id = ques_sent['index']
        next_id = q_sent_id + 1
        # if next_id in q_idx_map:
        #     answer_matrix.append(q_idx_map[next_id])
        # else:
        #     answer_matrix.append(p_idx_map[next_id])
        answer_matrix.append(idx_map[next_id])

    # Generate entity edges
    edges = [[0] * sent_num for _ in range(sent_num)]

    for sent in passage:
        tmp = set()
        for ptn in sent['pattern']:
            tmp.update(concepts[ptn].keys())
        sent_pattern_sets[sent['index']] = tmp

    for i, sent1 in enumerate(sentences):
        for j, sent2 in enumerate(sentences):
            if i == j:
                continue
            for ptn in sent1['pattern']:
                if ptn in sent_pattern_sets[sent2['index']]:
                    idx1 = idx_map[sent1['index']]
                    idx2 = idx_map[sent2['index']]
                    edges[idx1][idx2] = edges[idx2][idx1] = 1

    return {
        'ini_article': [sent['sentence'] for sent in sentences],
        'question': [{
            'index': sent['index'],
            'text': sent['noisy_sentence']
        } for sent in question],
        'passage': [{
            'index': sent['index'],
            'text': sent['sentence']
        } for sent in passage],
        'answer': answer_matrix,
        'edges': edges,
        # 'entity_pairs': entity_pairs,
        'raw_entity': list(raw_entity),
        'raw_entity_lemma': list(raw_entity_lemma)
    }


data = json.load(open(args.input_file, 'r'))

examples = []
for instance in tqdm(data, dynamic_ncols=True):
    id = instance['id']
    sentences = instance['article']

    example = workflow_0(sentences, args.keep_prob,
                         args.mask_prob, args.replace_prob)
    if example:
        example['id'] = id
        examples.append(example)

print(len(examples))
output_file = f'{args.input_file}_{args.tf_tokenizer_name.split("-")[0]}_{args.keep_prob}_' \
              f'{args.mask_prob}_{args.replace_prob}_{args.seed}.json'

with open(output_file, 'w') as f:
    json.dump(examples, f, ensure_ascii=False)
