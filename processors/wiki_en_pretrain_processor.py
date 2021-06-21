import json

import datasets
import nltk
from tqdm import tqdm
from transformers import RobertaTokenizerFast, BertTokenizerFast
from utils import custom_replace, combine_sentence

_patterns = [(r'<.*>', ''),
             (r'&amp;', '&'),
             (r'&lt;', '<'),
             (r'&gt;', '>'),
             (r'<ref[^<]*<\/ref>', ''),
             (r'<[^>]*>', ''),
             (r'\[http:[^] ]*', '['),
             (r'\|thumb', ''),
             (r'\|left', ''),
             (r'\|right', ''),
             (r'\|\d+px', ''),
             (r'\[\[image:[^\[\]]*\|', ''),
             (r'\[\[category:([^|\]]*)[^]]*\]\]', '[[$1]]'),
             (r'\[\[[a-z\-]*:[^\]]*\]\]', ''),
             (r'\[\[[^\|\]]*\|', '[['),
             (r'\{\{[^\}]*\}\}', ''),
             (r'\{[^\}]*\}', ''),
             (r'\[', ''),
             (r'\]', ''),
             (r'&[^;]*;', ' '),
             #  (r'A', 'a'), (r'B', 'b'), (r'C', 'c'),
             #  (r'D', 'd'), (r'E', 'e'), (r'F', 'f'),
             #  (r'G', 'g'), (r'H', 'h'), (r'I', 'i'),
             #  (r'J', 'j'), (r'K', 'k'), (r'L', 'l'),
             #  (r'M', 'm'), (r'N', 'n'), (r'O', 'o'),
             #  (r'P', 'p'), (r'Q', 'q'), (r'R', 'r'),
             #  (r'S', 's'), (r'T', 't'), (r'U', 'u'),
             #  (r'V', 'v'), (r'W', 'w'), (r'X', 'x'),
             #  (r'Y', 'y'), (r'Z', 'z'),
             #  (r'0', ' zero '), (r'1', ' one '), (r'2', ' two '),
             #  (r'3', ' three '), (r'4', ' four '), (r'5', ' five '),
             #  (r'6', ' six '), (r'7', ' seven '), (r'8', ' eight '),
             #  (r'9', ' nine '),
             #  (r'[^a-z\n]+', ' '),
             #  (r'[^a-zA-Z0-9\n\.\?\!\'\"]+', ' '),
             (r'[^a-zA-Z0-9\n\.\?\!]+', ' '),
             (r'\n ', ''),
             (r'\s+', ' '),
             (r'\n\s*\n', r'\n')
             ]

# wiki_en = datasets.load_dataset('wikipedia', '20200501.en')
wiki_en = datasets.load_dataset('/home/admin/workspace/project/datasets/datasets/datasets/wikipedia/wikipedia.py',
                                '20200501.en')
wiki_en = wiki_en['train']

# tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
bert_tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
custom_replace_transform = custom_replace(_patterns)

examples = []

debug_f = open('pre-process-log.txt', 'w')


def extract(example):
    title = example['title']
    text = example['text']

    text = text.replace('\n', ' ')
    sentences = nltk.sent_tokenize(text)

    para_id = 0
    article = []
    total_cnt = 0

    sentences = combine_sentence(custom_replace_transform(sentences), _len=5, drop=True, remove_long=50)

    for sent in sentences:
        if not sent.strip():
            continue
        # tokens = tokenizer.tokenize(sent)
        tokens = bert_tokenizer.tokenize(sent)
        if len(tokens) == 0:
            continue
        if len(tokens) <= 5:
            print(sent, file=debug_f, flush=True)
            print(tokens, file=debug_f, flush=True)
            # print(bert_tokenizer.tokenize(sent), file=debug_f, flush=True)
            print(' ', file=debug_f, flush=True)

        if total_cnt + len(tokens) < 500:
            article.append(sent)
            total_cnt += len(tokens)
        else:
            examples.append({
                'id': f'{title}-{para_id}',
                'article': article
            })
            article = [sent]
            total_cnt = len(tokens)
            para_id += 1

    if total_cnt > 200:
        examples.append({
            'id': f'{title}-{para_id}',
            'article': article
        })


cnt = 0
bool_dev = False
for example_id, example in tqdm(enumerate(wiki_en), total=len(wiki_en)):
    extract(example)

    # if not bool_dev and len(examples) >= 10000:
    #     print(
    #         f'Dump {len(examples)} examples into ../datasets/wiki_en_bpe_c/wiki_en_dev_10k.json .')
    #     with open(f'../datasets/wiki_en_bpe_c/wiki_en_dev_10k.json', 'w') as f:
    #         json.dump(examples, f, ensure_ascii=False)
    #     examples = []  # The first 10k samples are divided into dev set.
    #     bool_dev = True

    # if len(examples) >= 300000:
    #     print(
    #         f'Dump {len(examples)} examples into ../datasets/wiki_en_bpe_c/wiki_en_train_300k_{cnt}.json .')
    #     if cnt > 18:
    #         with open(f'../datasets/wiki_en_bpe_c/wiki_en_train_300k_{cnt}.json', 'w') as f:
    #             json.dump(examples, f, ensure_ascii=False)
    #     examples = []
    #     cnt += 1

    # BERT
    if not bool_dev and len(examples) >= 10000:
        print(
            f'Dump {len(examples)} examples into ../datasets/wiki_en_c/wiki_en_dev_10k.json .')
        with open(f'../datasets/wiki_en_c/wiki_en_dev_10k.json', 'w') as f:
            json.dump(examples, f, ensure_ascii=False)
        examples = []  # The first 10k samples are divided into dev set.
        bool_dev = True

    if len(examples) >= 300000:
        print(f'Dump {len(examples)} examples into ../datasets/wiki_en_c/wiki_en_train_300k_{cnt}.json .')
        
        with open(f'../datasets/wiki_en_c/wiki_en_train_300k_{cnt}.json', 'w') as f:
            json.dump(examples, f, ensure_ascii=False)
        examples = []
        cnt += 1
