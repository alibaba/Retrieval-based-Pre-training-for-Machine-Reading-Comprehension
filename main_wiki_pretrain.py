# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import random
from typing import Tuple

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup, AutoTokenizer

from data.data_instance import ModelState
from general_util.logger import setting_logger
from general_util.utils import AverageMeter
from models import model_map
from processors import get_processor

"""
torch==1.6.0
TODO: Add distributed training logic
"""


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    # parser.add_argument("--bert_model", default=None, type=str, required=True,
    #                     help="Bert pre-trained model selected in the list: bert-base-uncased, "
    #                          "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--vocab_file", default='bert-base-uncased-vocab.txt', type=str, required=True)
    parser.add_argument("--model_file", default='bert-base-uncased.tar.gz', type=str, required=True)
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints and predictions will be written.")
    parser.add_argument("--predict_dir", default=None, type=str, required=True,
                        help="The output directory where the predictions will be written.")

    # Other parameters
    parser.add_argument("--train_file", default=None, type=str, help="SQuAD json for training. E.g., train-v1.1.json")
    parser.add_argument("--predict_file", default=None, type=str,
                        help="SQuAD json for predictions. E.g., dev-v1.1.json or test-v1.1.json")
    parser.add_argument("--test_file", default=None, type=str)
    parser.add_argument("--max_seq_length", default=384, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--doc_stride", default=128, type=int,
                        help="When splitting up a long document into chunks, how much stride to take between chunks.")
    parser.add_argument("--max_query_length", default=64, type=int,
                        help="The maximum number of tokens for the question. Questions longer than this will "
                             "be truncated to this length.")
    parser.add_argument("--do_train", default=False, action='store_true', help="Whether to run training.")
    parser.add_argument("--do_predict", default=False, action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--do_label", default=False, action='store_true', help="For preprocess only.")
    parser.add_argument("--train_batch_size", default=32, type=int, help="Total batch size for training.")
    parser.add_argument("--predict_batch_size", default=8, type=int, help="Total batch size for predictions.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=2.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10% "
                             "of training.")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--do_lower_case",
                        default=True,
                        action='store_true',
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")

    # Base setting
    parser.add_argument('--pretrain', type=str, default=None)
    parser.add_argument('--bert_name', type=str, default='self_sup_pretrain_cls_query')
    parser.add_argument('--reader_name', type=str, default='wiki_pre_train')
    parser.add_argument('--per_eval_step', type=int, default=200)

    # Self-supervised pre-training
    parser.add_argument('--keep_prob', default=0.6, type=float)
    parser.add_argument('--mask_prob', default=0.2, type=float)
    parser.add_argument('--replace_prob', default=0.05, type=float)

    # Self-supervised pre-training input_file setting
    parser.add_argument('--input_files_template', type=str, required=True)
    parser.add_argument('--input_files_start', type=int, default=0)
    parser.add_argument('--input_files_end', type=int, default=10)
    parser.add_argument('--save_each_epoch', default=False, action='store_true')

    args = parser.parse_args()

    logger = setting_logger(args.output_dir)
    logger.info('================== Program start. ========================')

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}".format(
        device, n_gpu, bool(args.local_rank != -1)))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_predict and not args.do_label:
        raise ValueError("At least one of `do_train` or `do_predict` or `do_label` must be True.")

    if args.do_train:
        if not args.train_file and not args.input_files_template:
            raise ValueError(
                "If `do_train` is True, then `train_file` must be specified.")
    if args.do_predict:
        if not args.predict_file:
            raise ValueError(
                "If `do_predict` is True, then `predict_file` must be specified.")

    if args.do_train:
        if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
            raise ValueError("Output directory () already exists and is not empty.")
        os.makedirs(args.output_dir, exist_ok=True)

    if args.do_predict or args.do_label:
        os.makedirs(args.predict_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.vocab_file)

    data_reader = get_processor(args)

    num_train_steps = None
    if args.do_train or args.do_label:

        # Generate cache files
        input_files_format = args.input_files_template
        total_features = 0
        for idx in range(args.input_files_start, args.input_files_end + 1):
            train_file = input_files_format.format(str(idx))
            cached_train_features_file = train_file + '_' + data_reader.opt['cache_suffix']

            if not os.path.exists(cached_train_features_file):
                train_examples = data_reader.read(input_file=train_file)
                train_features = data_reader.convert_examples_to_features(examples=train_examples, tokenizer=tokenizer)

                _, train_tensors = data_reader.data_to_tensors_sent_pretrain(train_features)

                if args.local_rank == -1 or torch.distributed.get_rank() == 0:
                    logger.info(f" Saving train features into cached file {cached_train_features_file}")
                    torch.save((train_features, train_tensors), cached_train_features_file)
            else:
                train_features, train_tensors = torch.load(cached_train_features_file)

            total_features += train_tensors[0].size(0)

        del train_features
        del train_tensors
        num_train_steps = int(
            total_features / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

    # Prepare model
    if args.pretrain is not None:
        # TODO: Use new api in transformers compatible with model.save_pretrained() method.
        logger.info('Load pretrained model from {}'.format(args.pretrain))
        model_state_dict = torch.load(args.pretrain, map_location='cuda:0')
        model = model_map[args.bert_name].from_pretrained(args.model_file, state_dict=model_state_dict)
    else:
        model = model_map[args.bert_name].from_pretrained(args.model_file)

    model.to(device)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())

    # Remove frozen parameters
    param_optimizer = [n for n in param_optimizer if n[1].requires_grad]

    # hack to remove pooler, which is not used
    # thus it produce None grad that break apex
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    t_total = num_train_steps if num_train_steps else -1
    if args.local_rank != -1:
        t_total = t_total // torch.distributed.get_world_size()

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, correct_bias=False)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(t_total * args.warmup_proportion),
                                                num_training_steps=t_total)
    scaler = torch.cuda.amp.GradScaler()

    # Prepare data
    eval_examples = data_reader.read(input_file=args.predict_file)
    eval_features = data_reader.convert_examples_to_features(examples=eval_examples, tokenizer=tokenizer)

    eval_features, eval_tensors = data_reader.data_to_tensors_sent_pretrain(eval_features)
    eval_data = TensorDataset(*eval_tensors)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.predict_batch_size, num_workers=8)

    if args.do_train:
        logger.info("***** Running training *****")
        # logger.info("  Num orig examples = %d", len(train_examples))
        logger.info("  Num split examples = %d", total_features)
        logger.info("  Batch size = %d", args.train_batch_size)

        summary_writer = SummaryWriter(log_dir=args.output_dir)
        global_step = 0

        eval_loss = AverageMeter()
        train_loss = AverageMeter()
        best_loss = 1000000000
        eval_acc = AverageMeter()

        # Separate dataloader for huge dataset
        data_loader_id_range = list(range(args.input_files_start, args.input_files_end + 1))

        for epoch in range(int(args.num_train_epochs)):
            logger.info(f'Running at Epoch {epoch}')

            # Load training dataset separately

            random.shuffle(data_loader_id_range)
            for idx in data_loader_id_range:

                train_file = args.input_files_template.format(str(idx))
                cached_train_features_file = train_file + '_' + data_reader.opt['cache_suffix']

                logger.info(f"Load features and tensors from {cached_train_features_file}")
                train_features, train_tensors = torch.load(cached_train_features_file)

                train_data = TensorDataset(*train_tensors)
                if args.local_rank == -1:
                    train_sampler = RandomSampler(train_data)
                else:
                    train_sampler = DistributedSampler(train_data)
                train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

                # Train
                for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration", dynamic_ncols=True)):
                    model.train()
                    if n_gpu == 1:
                        batch = batch_to_device(batch, device)  # multi-gpu does scattering it-self
                    inputs = data_reader.generate_inputs(batch, train_features, model_state=ModelState.Train)

                    with torch.cuda.amp.autocast():
                        output_dict = model(**inputs)
                        loss = output_dict['loss']

                    train_loss.update(val=loss.item(), n=inputs['answers'].size(0))

                    if n_gpu > 1:
                        loss = loss.mean()  # mean() to average on multi-gpu.
                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps

                    scaler.scale(loss).backward()

                    if (step + 1) % args.gradient_accumulation_steps == 0:
                        # scaler.unscale_(optimizer)
                        # torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                        scaler.step(optimizer)
                        scaler.update()

                        scheduler.step()
                        # optimizer.step()
                        optimizer.zero_grad()
                        global_step += 1

                        lr_this_step = scheduler.get_lr()[0]
                        summary_writer.add_scalar('lr', lr_this_step, global_step)

                        batch_size = inputs["answers"].size(0)
                        summary_writer.add_scalar('train/loss', train_loss.avg, global_step)
                        # print(f'loss: {train_loss.avg}')
                        # print(inputs["answers"])

                        if global_step % args.per_eval_step == 0:
                            # Evaluation
                            model.eval()
                            logger.info("Start evaluating")
                            for _, eval_batch in enumerate(
                                    tqdm(eval_dataloader, desc="Evaluating", dynamic_ncols=True)):
                                if n_gpu == 1:
                                    eval_batch = batch_to_device(eval_batch,
                                                                 device)  # multi-gpu does scattering it-self
                                inputs = data_reader.generate_inputs(eval_batch, eval_features,
                                                                     model_state=ModelState.Train)

                                with torch.no_grad():
                                    with torch.cuda.amp.autocast():
                                        output_dict = model(**inputs)
                                        loss = output_dict['loss']

                                    batch_size = inputs["answers"].size(0)
                                    eval_loss.update(loss.item(), batch_size)
                                    eval_acc.update(output_dict["acc"].item(), output_dict["valid_num"])

                            eval_epoch_loss = eval_loss.avg
                            eval_epoch_acc = eval_acc.avg
                            summary_writer.add_scalar('eval/loss', eval_epoch_loss, global_step)
                            summary_writer.add_scalar('eval/acc', eval_epoch_acc, global_step)
                            eval_loss.reset()
                            eval_acc.reset()

                            if eval_epoch_loss < best_loss:
                                best_loss = eval_epoch_loss
                                # Only save the model it-self
                                model_to_save = model.module if hasattr(model, 'module') else model
                                output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")
                                torch.save(model_to_save.state_dict(), output_model_file)

                            logger.info(
                                f'Global step: {global_step}, eval_loss: {eval_epoch_loss}, best_loss: {best_loss}, '
                                f'eval_acc: {eval_epoch_acc}')
                            logger.info(f'Train loss: {train_loss.avg}')

                # Free memory
                del train_features
                del train_tensors
                del train_data
                del train_sampler
                del train_dataloader

            if args.save_each_epoch:
                target_dir = os.path.join(args.output_dir, f'epoch_{epoch}')
                os.makedirs(target_dir, exist_ok=True)
                model.save_pretrained(target_dir)

        summary_writer.close()


def batch_to_device(batch: Tuple[torch.Tensor], device):
    # batch[-1] don't move to gpu.
    output = []
    for t in batch[:-1]:
        output.append(t.to(device))
    output.append(batch[-1])
    return output


if __name__ == "__main__":
    main()
