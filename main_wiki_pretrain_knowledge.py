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
import json
import os
import random

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup, AutoTokenizer

from general_util.logger import setting_logger
from general_util.utils import AverageMeter
from models import model_map, get_added_args
from processors import get_processor

"""
torch==1.6.0
TODO: Add distributed training logic
"""


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--model_name_or_path",
                        default=None, type=str, required=True)
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints and predictions will be written.")
    parser.add_argument("--predict_dir", default=None, type=str, required=True,
                        help="The output directory where the predictions will be written.")
    parser.add_argument("--tensorboard", default='tensorboard', type=str)

    # Other parameters
    parser.add_argument("--predict_file", default=None, type=str,
                        help="SQuAD json for predictions. E.g., dev-v1.1.json or test-v1.1.json")
    parser.add_argument("--test_file", default=None, type=str)
    parser.add_argument("--max_seq_length", default=384, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--do_train", default=False,
                        action='store_true', help="Whether to run training.")
    parser.add_argument("--do_predict", default=False,
                        action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--do_label", default=False,
                        action='store_true', help="For preprocess only.")
    parser.add_argument("--train_batch_size", default=32,
                        type=int, help="Total batch size for training.")
    parser.add_argument("--predict_batch_size", default=8,
                        type=int, help="Total batch size for predictions.")
    parser.add_argument("--learning_rate", default=5e-5,
                        type=float, help="The initial learning rate for Adam.")
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
    parser.add_argument('--num_workers', default=12, type=int)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument("--adam_epsilon", default=1e-8,
                        type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0,
                        type=float, help="Max gradient norm.")
    parser.add_argument("--adam_betas", default="(0.9, 0.999)", type=str)

    # Base setting
    parser.add_argument('--pretrain', type=str, default=None)
    parser.add_argument('--bert_name', type=str,
                        default='self_sup_pretrain_cls_query')
    parser.add_argument('--reader_name', type=str, default='wiki_pre_train')
    parser.add_argument('--per_eval_step', type=int, default=200)

    # Self-supervised pre-training
    parser.add_argument('--keep_prob', default=0.6, type=float)
    parser.add_argument('--mask_prob', default=0.2, type=float)
    parser.add_argument('--replace_prob', default=0.05, type=float)
    parser.add_argument('--add_lm', default=False, action='store_true')
    parser.add_argument('--if_parallel', default=False, action='store_true')

    # Self-supervised pre-training input_file setting
    parser.add_argument('--input_file_list', type=str, default=None)
    parser.add_argument('--per_save_step', type=int, default=-1)

    # Bert GAT for knolwedge based sentence re-ordering task.
    parser.add_argument("--graph_loss", type=float, default=0.1)
    parser.add_argument("--gat_layers", type=int, default=3)
    parser.add_argument("--gat_intermediate_size", type=int, default=2048)
    parser.add_argument("--gat_attn_heads", type=int, default=6)

    # Bert SSP SG
    parser.add_argument("--graph_layers", type=int, default=3)
    parser.add_argument("--cls_n_head", type=int, default=12)
    parser.add_argument("--cls_d_head", type=int, default=64)

    # Bert SSP ES
    parser.add_argument("--es_layer_id", type=int, default=9)

    # Bert SSP KG
    parser.add_argument("--kg_layer_ids", type=str, default='6')
    parser.add_argument("--edge_type", type=str, default='knowledge')

    # Auto-regressive sentence re-order decoder
    parser.add_argument("--max_targets", type=int, default=80)
    parser.add_argument("--pos_emb_size", type=int, default=200)
    parser.add_argument("--decoder_inter_size", type=int, default=1536)

    args = parser.parse_args()

    logger = setting_logger(args.output_dir)
    logger.info('================== Program start. ========================')

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
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

    args.train_batch_size = int(
        args.train_batch_size / args.gradient_accumulation_steps)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_predict and not args.do_label:
        raise ValueError(
            "At least one of `do_train` or `do_predict` or `do_label` must be True.")

    if args.do_train:
        if not args.input_file_list:
            raise ValueError(
                "If `do_train` is True, then `train_file` must be specified.")
    if args.do_predict:
        if not args.predict_file:
            raise ValueError(
                "If `do_predict` is True, then `predict_file` must be specified.")

    if args.do_train:
        if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
            raise ValueError(
                "Output directory () already exists and is not empty.")
        os.makedirs(args.output_dir, exist_ok=True)

    if args.do_predict or args.do_label:
        os.makedirs(args.predict_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    # Prepare model
    ModelClass = model_map[args.bert_name]
    args.base_model_type = ModelClass.config_class.model_type

    if args.pretrain is not None:
        logger.info('Load pretrained model from {}'.format(args.pretrain))
        model_state_dict = torch.load(args.pretrain, map_location='cuda:0')
        model = ModelClass.from_pretrained(args.model_name_or_path, **get_added_args(args, ModelClass),
                                           state_dict=model_state_dict)
    else:
        model = ModelClass.from_pretrained(
            args.model_name_or_path, **get_added_args(args, ModelClass))

    model.to(device)

    if args.do_label:
        del model

    data_reader = get_processor(args)

    num_train_steps = None
    if args.do_train or args.do_label:

        # Generate cache files
        input_file_list = json.load(open(args.input_file_list, 'r'))
        input_files = []

        step_num_file = f'{args.input_file_list[:-5]}-{data_reader.opt["cache_suffix"]}-steps.json'
        step_update = False
        if not os.path.exists(step_num_file):
            step_num_map = {}
        else:
            step_num_map = json.load(open(step_num_file, 'r'))

        total_features = 0
        for train_file in input_file_list:
            cached_train_features_file = train_file + \
                '_' + data_reader.opt['cache_suffix']
            input_files.append(cached_train_features_file)

            if train_file not in step_num_map:
                # Use try-except structure to avoid none ending of the file.
                try:
                    train_examples, train_tensors = torch.load(
                        cached_train_features_file)
                    logger.info(
                        f" Load pre-processed examples and tensors from {cached_train_features_file}")
                except:
                    train_examples = data_reader.read(input_file=train_file)
                    train_tensors = data_reader.convert_examples_to_tensors(examples=train_examples,
                                                                            tokenizer=tokenizer)

                    if args.local_rank == -1 or torch.distributed.get_rank() == 0:
                        logger.info(
                            f" Saving train features into cached file {cached_train_features_file}")
                        torch.save((train_examples, train_tensors),
                                   cached_train_features_file)

                # if not os.path.exists(cached_train_features_file):
                #     train_examples = data_reader.read(input_file=train_file)
                #     train_tensors = data_reader.convert_examples_to_tensors(examples=train_examples, tokenizer=tokenizer)

                #     if args.local_rank == -1 or torch.distributed.get_rank() == 0:
                #         logger.info(f" Saving train features into cached file {cached_train_features_file}")
                #         torch.save((train_examples, train_tensors), cached_train_features_file)
                # else:
                #     logger.info(f" Load pre-processed examples and tensors from {cached_train_features_file}")
                #     train_examples, train_tensors = torch.load(cached_train_features_file)

                step_num_map[train_file] = train_tensors[0].size(0)
                step_update = True
                # total_features += train_tensors[0].size(0)

                del train_examples
                del train_tensors
                del cached_train_features_file

            logger.info(
                f' Num features from {train_file}: {step_num_map[train_file]}.')

            total_features += step_num_map[train_file]

        num_train_steps = int(
            total_features / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

        if step_update:
            json.dump(step_num_map, open(step_num_file, 'w'), indent=2)

        if args.do_label:
            logger.info(f'Finished pre-processing.')
            return

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

    exec('args.adam_betas = ' + args.adam_betas)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    t_total = num_train_steps if num_train_steps else -1
    if args.local_rank != -1:
        t_total = t_total // torch.distributed.get_world_size()

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, correct_bias=False,
                      eps=args.adam_epsilon, betas=args.adam_betas)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(t_total * args.warmup_proportion),
                                                num_training_steps=t_total)
    scaler = torch.cuda.amp.GradScaler()

    # Prepare data
    _eval_cache_file = args.predict_file + \
        '_' + data_reader.opt['cache_suffix']

    if os.path.exists(_eval_cache_file):
        eval_examples, eval_tensors = torch.load(_eval_cache_file)
    else:
        eval_examples = data_reader.read(input_file=args.predict_file)
        eval_tensors = data_reader.convert_examples_to_tensors(
            examples=eval_examples, tokenizer=tokenizer)

    eval_data = TensorDataset(*eval_tensors)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.predict_batch_size,
                                 collate_fn=getattr(data_reader, "collate_fn", None), num_workers=args.num_workers)

    if args.do_train:
        logger.info("***** Running training *****")
        logger.info("  Num split examples = %d", total_features)
        logger.info("  Num Epochs = %d", args.num_train_epochs)
        logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)
        logger.info(
            "  Total train batch size (w. parallel, distributed & accumulation) = %d",
            args.train_batch_size
            * args.gradient_accumulation_steps
            * (torch.distributed.get_world_size()
               if args.local_rank != -1 else 1),
        )
        logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)

        summary_writer = SummaryWriter(log_dir=args.tensorboard)
        if not os.path.exists(args.tensorboard):
            os.makedirs(args.tensorboard, exist_ok=True)
        global_step = 0

        eval_loss = AverageMeter()
        train_loss = AverageMeter()
        best_loss = 1000000000
        eval_acc = AverageMeter()

        # Separate dataloader for huge dataset
        for epoch in range(int(args.num_train_epochs)):
            logger.info(f'Running at Epoch {epoch}')

            # Load training dataset separately
            random.shuffle(input_files)
            # FIXME: Wrong name here !!!
            for cached_train_features_file in input_files:

                logger.info(f"Load features and tensors from {cached_train_features_file}")
                _, train_tensors = torch.load(cached_train_features_file)
                logger.info(f"Num features: {train_tensors[0].size(0)}")

                train_data = TensorDataset(*train_tensors)
                if args.local_rank == -1:
                    train_sampler = RandomSampler(train_data)
                else:
                    train_sampler = DistributedSampler(train_data)
                train_dataloader = DataLoader(train_data, sampler=train_sampler,
                                              batch_size=args.train_batch_size,
                                              collate_fn=getattr(data_reader, "collate_fn", None),
                                              num_workers=args.num_workers)

                # Train
                for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration", dynamic_ncols=True)):
                    model.train()

                    inputs = data_reader.generate_inputs(batch, device)

                    with torch.cuda.amp.autocast():
                        output_dict = model(**inputs)
                        loss = output_dict['loss']

                    train_loss.update(val=loss.item(), n=batch[0].size(0))

                    if n_gpu > 1:
                        loss = loss.mean()  # mean() to average on multi-gpu.
                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps

                    scaler.scale(loss).backward()

                    if (step + 1) % args.gradient_accumulation_steps == 0:
                        if args.max_grad_norm != -1:
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                        scaler.step(optimizer)
                        scaler.update()

                        scheduler.step()
                        # optimizer.step()
                        optimizer.zero_grad()
                        global_step += 1

                        lr_this_step = scheduler.get_lr()[0]
                        summary_writer.add_scalar('lr', lr_this_step, global_step)

                        batch_size = batch[0].size(0)
                        summary_writer.add_scalar('train/loss', train_loss.avg, global_step)

                        if global_step % args.per_eval_step == 0:
                            # Evaluation
                            model.eval()
                            logger.info("Start evaluating")
                            for _, eval_batch in enumerate(
                                    tqdm(eval_dataloader, desc="Evaluating", dynamic_ncols=True)):
                                inputs = data_reader.generate_inputs(eval_batch, device)

                                with torch.no_grad():
                                    with torch.cuda.amp.autocast():
                                        output_dict = model(**inputs)
                                        loss = output_dict['loss']

                                    batch_size = eval_batch[0].size(0)
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
                                model_to_save = (
                                    model.module if hasattr(model, "module") else model
                                )  # Take care of distributed/parallel training
                                model_to_save.save_pretrained(args.output_dir)
                                tokenizer.save_pretrained(args.output_dir)

                                # Good practice: save your training arguments together with the trained model
                                torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

                            logger.info(
                                f'Global step: {global_step}, eval_loss: {eval_epoch_loss}, '
                                f'eval_acc: {eval_epoch_acc}, best_loss: {best_loss}. '
                            )
                            if hasattr(model, "eval_metrics"):
                                _eval_metric_log = model.eval_metrics.get_log()
                                _eval_metric_log = '\t'.join([f"{k}: {v}" for k, v in _eval_metric_log.items()])
                                logger.info(_eval_metric_log)
                                model.eval_metrics.reset()

                            logger.info(f'Train loss: {train_loss.avg}')
                            torch.cuda.empty_cache()

                        if args.local_rank in [-1, 0] and args.per_save_step > 0 and global_step % args.per_save_step == 0:
                            # Save model checkpoint
                            output_dir = os.path.join(
                                args.output_dir, "checkpoint-{}".format(global_step))
                            if not os.path.exists(output_dir):
                                os.makedirs(output_dir)
                            model_to_save = (
                                model.module if hasattr(
                                    model, "module") else model
                            )  # Take care of distributed/parallel training
                            model_to_save.save_pretrained(output_dir)
                            tokenizer.save_pretrained(output_dir)

                            torch.save(args, os.path.join(
                                output_dir, "training_args.bin"))
                            logger.info(
                                "Saving model checkpoint to %s", output_dir)

                            torch.save(optimizer.state_dict(), os.path.join(
                                output_dir, "optimizer.pt"))
                            torch.save(scheduler.state_dict(), os.path.join(
                                output_dir, "scheduler.pt"))
                            torch.save(scaler.state_dict(), os.path.join(
                                output_dir, "scaler.pt"))
                            logger.info(
                                "Saving optimizer and scheduler states to %s", output_dir)

                # Free memory
                del train_tensors
                del train_data
                del train_sampler
                del train_dataloader

        summary_writer.close()

    if args.do_predict:
        eval_loss = AverageMeter()
        eval_acc = AverageMeter()

        model.eval()
        logger.info("Start predicting.")
        for _, eval_batch in enumerate(tqdm(eval_dataloader, desc="Evaluating", dynamic_ncols=True)):
            inputs = data_reader.generate_inputs(eval_batch, device)

            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    output_dict = model(**inputs)
                    loss = output_dict['loss']

            batch_size = eval_batch[0].size(0)
            eval_loss.update(loss.item(), batch_size)
            eval_acc.update(output_dict["acc"].item(), output_dict["valid_num"])

        logger.info("=============== Predict Results ====================")
        logger.info(f"Accuracy: {eval_acc.avg}\tLoss: {eval_loss.avg}.")


if __name__ == "__main__":
    main()
