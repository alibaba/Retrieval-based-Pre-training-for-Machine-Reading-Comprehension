import argparse
import json
from multiprocessing import Pool, Manager, Lock

from transformers import BertTokenizer
from utils import custom_replace

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
             (r'[^a-zA-Z0-9\n\.\?\!\'\"]+', ' '),
             (r'\n ', ''),
             (r'\s+', ' '),
             (r'\n\s*\n', r'\n')
             ]

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
input_file_tem = '../datasets/wiki_en_train_300k_{}.json'
# input_file_tem = '../datasets/wiki_en_dev_{}k.json'

parser = argparse.ArgumentParser()
parser.add_argument('--file_id', type=int, default=0)
args = parser.parse_args()

custom_replace_transform = custom_replace(_patterns)

data = json.load(open(input_file_tem.format(str(args.file_id)), 'r'))

manager = Manager()
examples = manager.list()
lock = Lock()

t = manager.dict()
# t = dict()
# examples = []

t['avg_len'] = 0


def interface(instance):
    text = instance['article']

    new_text = []
    for sent in custom_replace_transform(text):

        wd_pieces = tokenizer.tokenize(sent)
        if len(wd_pieces) == 0:
            continue

        new_text.append(sent)
        with lock:
            t['avg_len'] += len(wd_pieces)

    if len(new_text) == 0:
        return

    with lock:
        instance['article'] = new_text
        examples.append(instance)


pool = Pool(10)
pool.map(interface, data)
pool.close()
pool.join()

# for ist in tqdm(data, dynamic_ncols=True):
#     interface(ist)

examples = [ex for ex in examples]

with open(f'../datasets/wiki_en_train_300k_{args.file_id}_cl.json', 'w') as f:
    json.dump(examples, f, ensure_ascii=False)
# with open(f'../datasets/wiki_en_dev_10k_cl.json', 'w') as f:
#     json.dump(examples, f, ensure_ascii=False)

print('Read ', len(examples), ' examples in total.')
print('Average word-pieces length: ', t['avg_len'] * 1.0 / len(examples))
