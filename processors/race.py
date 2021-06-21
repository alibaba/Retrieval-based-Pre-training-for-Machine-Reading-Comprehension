import json
import logging

import torch
from tqdm import tqdm
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


class RACEProcessor(object):
    reader_name = 'race'

    def __init__(self, args):
        super(RACEProcessor, self).__init__()
        self.opt = vars(args)
        self.opt['cache_suffix'] = f"{self.reader_name}_{args.base_model_type}_{args.max_seq_length}"

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
        # max_query_length = self.opt['max_query_length']
        if_bpe = 'roberta' in self.opt['base_model_type']

        features = []
        all_input_ids, all_token_type_ids, all_attention_mask, all_labels = [], [], [], []
        for example_index, example in enumerate(
                tqdm(examples, desc='convert examples to tensors...', dynamic_ncols=True)):
            text_a = [example['question'] + f' {tokenizer.sep_token} ' + op for op in example['options']]
            text_b = [example['passage']] * len(example['options'])
            encoded_output = tokenizer(text_a, text_b, padding='max_length',
                                       truncation='longest_first', max_length=max_seq_length)
            all_input_ids.append(encoded_output['input_ids'])
            all_token_type_ids.append(encoded_output['token_type_ids'])
            all_attention_mask.append(encoded_output['attention_mask'])
            all_labels.append(example['answer'])
        input_ids = torch.LongTensor(all_input_ids)
        token_type_ids = torch.LongTensor(all_token_type_ids)
        attention_mask = torch.LongTensor(all_attention_mask)
        labels = torch.LongTensor(all_labels)
        index = torch.arange(input_ids.size(0))

        assert input_ids.size(2) == token_type_ids.size(2) == attention_mask.size(2) == max_seq_length

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
        acc = torch.sum((torch.LongTensor(pred) == torch.LongTensor(labels))).item() * 1.0 / len(pred)

        return {
            'accuracy': acc
        }
