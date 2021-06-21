import json
import re
from collections import defaultdict

import nltk
import spacy
from nltk import sent_tokenize, word_tokenize


nltk.data.path.append('nltk_data')

try:
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'textcat'])
    nlp_no_ner = spacy.load('en_core_web_sm', disable=['parser', 'textcat', 'ner'])
    nltk_stopwords = nltk.corpus.stopwords.words('english')

    concepts = json.load(open('processors/concept_pattern.json'))
except:
    nlp = None
    nlp_no_ner = None
    nltk_stopwords = None
    
    concepts = None


def custom_replace(replace_pattern):
    r"""A transform to convert text string.
    Examples:
        >>> from torchtext.data.functional import custom_replace
        >>> custom_replace_transform = custom_replace([(r'S', 's'), (r'\s+', ' ')])
        >>> list_a = ["Sentencepiece encode  aS  pieces", "exampleS to   try!"]
        >>> list(custom_replace_transform(list_a))
            ['sentencepiece encode as pieces', 'examples to try!']
    """

    _patterns = list((re.compile(p), r)
                     for (p, r) in replace_pattern)

    def _internal_func(txt_iter):
        for line in txt_iter:
            for pattern_re, replaced_str in _patterns:
                line = pattern_re.sub(replaced_str, line)
            yield line

    return _internal_func


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


def pattern_pipeline(article, split_sent=True):
    if split_sent:
        article = sent_tokenize(article)

    assert isinstance(article, list)
    sent_num = len(article)
    all_docs = nlp.pipe(article)
    all_ent_docs = []
    all_ent_doc_secs = []
    for doc in all_docs:
        tmp = []
        for ent in doc.ents:
            tmp.append(ent.text.lower())
        _s = len(all_ent_docs)
        all_ent_docs.extend(tmp)
        _e = len(all_ent_docs)
        all_ent_doc_secs.append((_s, _e))
    assert len(all_ent_doc_secs) == sent_num, (len(all_ent_doc_secs), sent_num)

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

    edges = [[0] * sent_num for _ in range(sent_num)]

    sent_edges = []
    for sent_id, sent_pattern in enumerate(all_ent_patterns):
        tmp = set()
        for ptn in sent_pattern:
            tmp.update(concepts[ptn].keys())
        sent_edges.append(tmp)

    for i, sent_1_pattern in enumerate(all_ent_patterns):
        for j, sent_2_pattern in enumerate(all_ent_patterns):
            if i == j:
                continue
            for ptn in sent_1_pattern:
                if ptn in sent_edges[j]:
                    edges[i][j] = edges[j][i] = 1
                    break

    return edges


def extract_concept_patterns(sentences):
    sent_num = len(sentences)
    all_docs = nlp.pipe(sentences)
    all_ent_docs = []
    all_ent_doc_secs = []
    for doc in all_docs:
        tmp = []
        for ent in doc.ents:
            tmp.append(ent.text.lower())
        _s = len(all_ent_docs)
        all_ent_docs.extend(tmp)
        _e = len(all_ent_docs)
        all_ent_doc_secs.append((_s, _e))

    assert len(all_ent_doc_secs) == sent_num, (len(all_ent_doc_secs), sent_num)

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

    assert len(all_ent_patterns) == sent_num

    return all_ent_patterns


def evidence_pipeline(question, passage):
    assert isinstance(question, list)
    assert isinstance(passage, list)

    ques_patterns = extract_concept_patterns(question)
    pass_patterns = extract_concept_patterns(passage)
    vis = defaultdict(dict)

    def is_marked(a, b):
        if a in vis and b in vis[a]:
            return True
        if b in vis and a in vis[b]:
            return True
        return False

    evidences = []
    for q_sent_id, q_sent in enumerate(ques_patterns):
        for ptn1 in q_sent:
            ptn_1_lm = ptn1.replace('_', ' ')

            for p_sent_id, p_sent in enumerate(pass_patterns):
                for ptn2 in p_sent:

                    if is_marked(ptn1, ptn2):
                        continue

                    ptn_2_lm = ptn2.replace('_', ' ')

                    if ptn1 in concepts[ptn2]:
                        r, _ = concepts[ptn2][ptn1]
                        ed = f'{ptn_2_lm} {r} {ptn_1_lm}'
                        evidences.append((ed, p_sent_id, q_sent_id))
                        vis[ptn2][ptn1] = 1
                        vis[ptn1][ptn2] = 1
                    elif ptn2 in concepts[ptn1]:
                        r, _ = concepts[ptn1][ptn2]
                        ed = f'{ptn_1_lm} {r} {ptn_2_lm}'
                        evidences.append((ed, q_sent_id, p_sent_id))
                        vis[ptn2][ptn1] = 1
                        vis[ptn1][ptn2] = 1

    return evidences


def combine_sentence(sentences, _len=6, drop=False, remove_long=-1):
    new_sents = []
    sent_words = []

    for sent in sentences:
        tks = word_tokenize(sent)
        if len(tks) <= _len:
            if len(new_sents) == 0:
                if drop:
                    continue
                else:
                    new_sents.append(sent)
                    sent_words.append(tks)
            else:
                new_sents[-1] += ' ' + sent
                sent_words[-1].extend(tks)
        else:
            new_sents.append(sent)
            sent_words.append(tks)

    assert len(new_sents) == len(sent_words)

    if remove_long > 0:
        new_sents = [new_sents[i] for i in range(len(new_sents)) if len(sent_words[i]) <= remove_long]

    return new_sents


def if_intersect(span1, span2):
    # span: (start, end)

    # Make span1[0] <= span2[0]
    if span1[0] >= span2[0]:
        span1, span2 = span2, span1

    if span1[1] > span2[0]:
        return True
    return False


def remove_duplicate_spans(tuple_ls1, tuple_ls2):
    # tuples in ls1 have larger span than tuples in ls2
    # tuple: (text, text.lemma_, text_start_offset, text_end_offset)

    sorted_tuple_ls1 = sorted(tuple_ls1, key=lambda x: x[2])
    sorted_tuple_ls2 = sorted(tuple_ls2, key=lambda x: x[2])

    new_ls2 = []

    ls1_p_index = 0
    ls2_p_index = 0

    while ls1_p_index < len(sorted_tuple_ls1) and ls2_p_index < len(sorted_tuple_ls2):
        ls1_p_span = (sorted_tuple_ls1[ls1_p_index][2], sorted_tuple_ls1[ls1_p_index][3])
        ls2_p_span = (sorted_tuple_ls2[ls2_p_index][2], sorted_tuple_ls2[ls2_p_index][3])

        if not if_intersect(ls1_p_span, ls2_p_span):
            if ls1_p_span[0] < ls2_p_span[0]:
                ls1_p_index += 1
            elif ls1_p_span[0] > ls2_p_span[0]:
                new_ls2.append(sorted_tuple_ls2[ls2_p_index])
                ls2_p_index += 1
            else:
                raise RuntimeError(f'{ls1_p_span}, {ls2_p_span}')
        else:
            ls2_p_index += 1

    if ls2_p_index < len(sorted_tuple_ls2):
        new_ls2.extend(sorted_tuple_ls2[ls2_p_index:])

    return new_ls2
