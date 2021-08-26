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
from multiprocessing import cpu_count
import math
import gc

import numpy as np
import torch
from oss_utils import torch_save_to_oss, set_bucket_dir, load_buffer_from_oss, json_save_to_oss
from torch import distributed as dist
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from transformers.tokenization_auto import AutoTokenizer

from general_util.logger import setting_logger_pai
from general_util.utils import AverageMeter
from models import model_map, get_added_args
from processors import get_processor

"""
torch==1.5.1
Use apex

This script is used for PAI distribution training, in which
multi-node with multi-gpu is used.
"""

volume_output_dir = "/data/output1/"


def update_config(_args):
    json_config = json.load(open(_args.config, "r"))

    opt = vars(_args)
    opt.update(json_config)


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--model_name_or_path",
                        default="roberta-large", type=str)
    parser.add_argument("--output_dir", default="outputs", type=str,
                        help="The output directory where the model checkpoints and predictions will be written.")
    parser.add_argument("--predict_dir", default="outputs", type=str,
                        help="The output directory where the predictions will be written.")
    parser.add_argument("--tensorboard", default="tensorboard", type=str)

    # Json config file, which will overwrite the other parameters
    parser.add_argument("--config", default=None, type=str)

    # Other parameters
    parser.add_argument("--predict_file", default="/data/volume2/wiki_en_dev_20k_from11.json_wk4_0.8_0.1_0.3_0.0.json",
                        type=str,
                        help="SQuAD json for predictions. E.g., dev-v1.1.json or test-v1.1.json")
    parser.add_argument("--test_file", default=None, type=str)
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--do_train", default=True,
                        action='store_true', help="Whether to run training.")
    parser.add_argument("--do_predict", default=False,
                        action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--do_label", default=False,
                        action='store_true', help="For preprocess only.")
    parser.add_argument("--train_batch_size", default=256,
                        type=int, help="Total batch size for training.")
    parser.add_argument("--predict_batch_size", default=16,
                        type=int, help="Total batch size for predictions.")
    parser.add_argument("--learning_rate", default=2e-5,
                        type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=5.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.06, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10% "
                             "of training.")
    parser.add_argument("--warmup_steps", default=-1, type=int)
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
                        default=4,
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
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--adam_betas", default="(0.9, 0.999)", type=str)
    parser.add_argument('--fp16', default=True, action='store_true')
    parser.add_argument('--fp16_opt_level', default='O1', type=str)
    parser.add_argument('--optim', default='adamw', type=str)
    parser.add_argument('--scheduler', default='linear', type=str)
    parser.add_argument('--poly_power', default=0.5, type=float)

    # Base setting
    parser.add_argument('--pretrain', type=str, default=None)
    parser.add_argument('--bert_name', type=str,
                        default='roberta_sr')
    parser.add_argument('--reader_name', type=str, default='wiki_pre_train_ent')
    parser.add_argument('--per_eval_step', type=int, default=250)
    parser.add_argument('--logging_step', type=int, default=2)
    parser.add_argument('--max_steps', default=-1, type=int)

    # Self-supervised pre-training input_file setting
    parser.add_argument('--input_file_list', type=str, default="/data/volume4/file_list_wk4_0.6_0.8_0.1_0.3_0.0.json")
    parser.add_argument('--per_save_step', type=int, default=-1)

    # # Bert GAT for knowledge based sentence re-ordering task.
    # parser.add_argument("--graph_loss", type=float, default=0.1)
    # parser.add_argument("--gat_layers", type=int, default=3)
    # parser.add_argument("--gat_intermediate_size", type=int, default=2048)
    # parser.add_argument("--gat_attn_heads", type=int, default=6)

    # # Bert SSP SG
    # parser.add_argument("--graph_layers", type=int, default=3)
    # parser.add_argument("--cls_n_head", type=int, default=12)
    # parser.add_argument("--cls_d_head", type=int, default=64)

    # # Bert SSP ES
    # parser.add_argument("--es_layer_id", type=int, default=9)

    # # Bert SSP KG
    # parser.add_argument("--kg_layer_ids", type=str, default='6')
    # parser.add_argument("--edge_type", type=str, default='knowledge')

    # Auto-regressive sentence re-order decoder
    parser.add_argument("--max_targets", type=int, default=80)
    parser.add_argument("--pos_emb_size", type=int, default=200)
    parser.add_argument("--decoder_inter_size", type=int, default=1536)
    parser.add_argument("--correct_bias", default=False, action='store_true')

    # Query based pre-training config
    parser.add_argument("--query_dropout", default=0.2, type=float)
    parser.add_argument("--query_ff_size", default=1536, type=int)
    parser.add_argument("--sr_query_dropout", default=0.2, type=float)
    parser.add_argument("--lm_query_dropout", default=0.1, type=float)

    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--resume", type=int, default=None)

    args = parser.parse_args()

    update_config(args)

    dist.init_process_group(backend='nccl')
    print(f"local rank: {args.local_rank}")
    print(f"global rank: {dist.get_rank()}")
    print(f"world size: {dist.get_world_size()}")

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
        dist.init_process_group(backend='nccl')

    global_rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size > 1:
        args.local_rank = global_rank

    set_bucket_dir(args.output_dir)
    torch_save_to_oss(args, "/training_args.bin")

    args.output_dir = os.path.join(volume_output_dir, args.output_dir)
    args.predict_dir = os.path.join(volume_output_dir, args.predict_dir)
    log_dir = os.path.join(volume_output_dir, "log_dir/")
    tensorboard = os.path.join(volume_output_dir, "tensorboard/")

    if args.local_rank not in [-1, 0]:
        dist.barrier()

    if args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(tensorboard, exist_ok=True)

        if args.local_rank == 0:
            dist.barrier()

    logger = setting_logger_pai(log_dir, args.local_rank)

    logger.info('================== Program start. ========================')
    logger.info(args)

    logger.info("device: {} n_gpu: {}, distributed training: {}".format(
        device, n_gpu, bool(args.local_rank != -1)))
    logger.info("global rank: {}, world size: {}, cpu kernel: {}".format(
        dist.get_rank(), dist.get_world_size(), cpu_count()))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    if n_gpu > 0:
        args.train_batch_size = int(
            args.train_batch_size / (n_gpu * args.gradient_accumulation_steps))
    else:
        args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

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

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    # Prepare model
    model_class = model_map[args.bert_name]
    args.base_model_type = model_class.config_class.model_type

    if args.pretrain is not None:
        logger.info('Load pretrained model from {}'.format(args.pretrain))
        model_state_dict = torch.load(args.pretrain, map_location='cpu')
        model = model_class.from_pretrained(args.model_name_or_path, **get_added_args(args, model_class),
                                            state_dict=model_state_dict)
    elif args.resume is not None:
        model_checkpoint = f"pytorch_model_{args.resume}.bin"
        _buffer = load_buffer_from_oss(model_checkpoint)
        model_state_dict = torch.load(_buffer, map_location='cpu')
        model = model_class.from_pretrained(args.model_name_or_path, **get_added_args(args, model_class),
                                            state_dict=model_state_dict)
        logger.info(f"Load resumed checkpoint from {model_checkpoint}")
    else:
        model = model_class.from_pretrained(args.model_name_or_path, **get_added_args(args, model_class))

    # Save config for fine-tuning.
    json_save_to_oss(model.config.to_dict(), '/config.json')

    model.to(device)

    if args.do_label:
        del model

    data_reader = get_processor(args)

    num_train_steps = None
    total_features = 0
    if args.do_train or args.do_label:

        input_file_list = json.load(open(args.input_file_list, "r"))
        input_files = []
        input_file_steps = dict()

        step_num_file = f'{args.input_file_list[:-5]}-{data_reader.opt["cache_suffix"]}-steps.json'
        if os.path.exists(step_num_file):
            step_num_map = json.load(open(step_num_file, 'r'))
        else:
            step_num_map = dict()

        for train_file in input_file_list:
            cached_train_features_file = train_file + '_' + data_reader.opt['cache_suffix']
            input_files.append(cached_train_features_file)

            if train_file in step_num_map:
                total_features += step_num_map[train_file]
                input_file_steps[cached_train_features_file] = step_num_map[train_file]
                logger.info(f"Num features: {step_num_map[train_file]}")
                continue

            try:
                train_examples, train_tensors = torch.load(cached_train_features_file)
                logger.info(f" Load pre-processed examples and tensors from {cached_train_features_file}")
            except:
                assert args.cache_dir is not None
                train_examples = data_reader.read(input_file=train_file)
                train_tensors = data_reader.convert_examples_to_tensors(examples=train_examples,
                                                                        tokenizer=tokenizer)

                if args.local_rank == -1 or dist.get_rank() == 0:
                    cached_train_features_file = cached_train_features_file.split('/')[-1]
                    cached_train_features_file = os.path.join(args.cache_dir, cached_train_features_file)
                    logger.info(f" Saving train features into cached file {cached_train_features_file}")
                    torch_save_to_oss((train_examples, train_tensors), '/' + cached_train_features_file)
                    # torch.save((train_examples, train_tensors), cached_train_features_file)

                    if args.local_rank == 0:
                        dist.barrier()
                # raise RuntimeError("Please preprocess the tensors first.")

            logger.info(f"Num features: {train_tensors[0].size(0)}")
            total_features += train_tensors[0].size(0)
            input_file_steps[cached_train_features_file] = train_tensors[0].size(0)

        num_train_steps = int(
            total_features / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())

    # Remove frozen parameters
    param_optimizer = [n for n in param_optimizer if n[1].requires_grad]

    # hack to remove pooler, which is not used
    # thus it produce None grad that break apex
    # param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

    exec('args.adam_betas = ' + args.adam_betas)
    no_decay = ['bias', 'LayerNorm.weight', 'layer_norm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]

    t_total = num_train_steps if num_train_steps else -1
    if args.local_rank != -1:
        t_total = t_total // dist.get_world_size()
    if args.max_steps != -1:
        t_total = min(t_total, args.max_steps)

    if args.optim == 'adamw':
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=args.learning_rate, correct_bias=args.correct_bias,
                          eps=args.adam_epsilon, betas=args.adam_betas)
    elif args.optim == 'lamb':
        from apex.optimizers import FusedLAMB

        optimizer = FusedLAMB(optimizer_grouped_parameters,
                              lr=args.learning_rate, betas=args.adam_betas,
                              eps=args.adam_epsilon, max_grad_norm=args.max_grad_norm)
    else:
        raise ValueError(f"Unsupported optimizer for {args.optim}")

    logger.info(optimizer)

    num_warmup_steps = int(t_total * args.warmup_proportion)
    if args.warmup_steps != -1:
        num_warmup_steps = min(num_warmup_steps, args.warmup_steps)

    if args.scheduler == 'linear':
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                    num_training_steps=t_total)
    elif args.scheduler == 'poly':
        from transformers import get_polynomial_decay_schedule_with_warmup

        scheduler = get_polynomial_decay_schedule_with_warmup(optimizer,
                                                              num_warmup_steps=num_warmup_steps,
                                                              num_training_steps=t_total,
                                                              power=args.poly_power)
    else:
        scheduler = None

    if args.fp16 and n_gpu:
        from apex import amp

        if args.fp16_opt_level == 'O1':
            # amp.register_float_function(torch, "einsum")
            amp.register_half_function(torch, "einsum")

        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    if args.local_rank != -1:
        if args.fp16_opt_level == 'O2':
            try:
                import apex
                model = apex.parallel.DistributedDataParallel(model, delay_allreduce=True)
            except ImportError:
                model = torch.nn.parallel.DistributedDataParallel(model,
                                                                  find_unused_parameters=True)
        else:
            model = torch.nn.parallel.DistributedDataParallel(model,
                                                              find_unused_parameters=True)

    # Prepare validation data
    if args.local_rank not in [-1, 0]:
        dist.barrier()

    if args.local_rank in [-1, 0]:
        _eval_cache_file = args.predict_file + '_' + data_reader.opt['cache_suffix']

        if os.path.exists(_eval_cache_file):
            eval_examples, eval_tensors = torch.load(_eval_cache_file)
            logger.info(f"Cached evaluation dataset is loaded from {_eval_cache_file}.")
        else:
            eval_examples = data_reader.read(input_file=args.predict_file)
            eval_tensors = data_reader.convert_examples_to_tensors(
                examples=eval_examples, tokenizer=tokenizer)

        eval_data = TensorDataset(*eval_tensors)
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.predict_batch_size,
                                     collate_fn=getattr(data_reader, "collate_fn", None),
                                     num_workers=args.num_workers)

        if args.local_rank == 0:
            dist.barrier()

    if args.resume is not None:
        _amp_state_dict = torch.load(load_buffer_from_oss(f"amp_{args.resume}.pt"))
        _optimizer_state_dict = torch.load(load_buffer_from_oss(f"optimizer_{args.resume}.pt"))
        _scheduler_state_dict = torch.load(load_buffer_from_oss(f"scheduler_{args.resume}.pt"))

        amp.load_state_dict(_amp_state_dict)
        optimizer.load_state_dict(_optimizer_state_dict)
        scheduler.load_state_dict(_scheduler_state_dict)

        logger.info(f"Loaded resumed state dict of step {args.resume}")

    if args.do_train:

        logger.info("***** Running training *****")
        logger.info("  Num split examples = %d", total_features)
        logger.info("  Num Epochs = %d", args.num_train_epochs)
        logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)
        logger.info(
            "  Total train batch size (w. parallel, distributed & accumulation) = %d",
            args.train_batch_size
            * args.gradient_accumulation_steps
            * (dist.get_world_size() if args.local_rank != -1 else 1),
        )
        logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)

        global_step = 0

        eval_loss = AverageMeter()
        train_loss = AverageMeter()
        best_loss = 1000000000
        eval_acc = AverageMeter()

        get_eval_log_fn = None
        if hasattr(model, "get_eval_log"):
            get_eval_log_fn = model.get_eval_log
        elif hasattr(model, "module") and hasattr(model.module, "get_eval_log"):
            get_eval_log_fn = model.module.get_eval_log

        # Separate dataloader for huge dataset
        for epoch in range(int(args.num_train_epochs)):
            logger.info(f'Running at Epoch {epoch}')

            # Load training dataset separately
            random.shuffle(input_files)
            for cached_train_features_file in input_files:

                # if args.resume is not None and global_step < args.resume:
                #     _steps_per_file = int(math.ceil(input_file_steps[cached_train_features_file] * 1.0
                #                                     / args.train_batch_size
                #                                     / (dist.get_world_size() if args.local_rank != -1 else 1)))
                #     _steps_per_file = _steps_per_file // args.gradient_accumulation_steps
                #     if global_step + _steps_per_file <= args.resume:
                #         global_step += _steps_per_file
                #         continue

                torch.cuda.empty_cache()
                logger.info(f"Load features and tensors from {cached_train_features_file}")
                _, train_tensors = torch.load(cached_train_features_file)

                train_data = TensorDataset(*train_tensors)
                if args.local_rank == -1:
                    train_sampler = RandomSampler(train_data)
                else:
                    train_sampler = DistributedSampler(train_data)
                train_dataloader = DataLoader(train_data, sampler=train_sampler,
                                              batch_size=args.train_batch_size,
                                              collate_fn=getattr(data_reader, "collate_fn", None),
                                              num_workers=args.num_workers,
                                              pin_memory=True)
                if args.local_rank != -1:
                    train_dataloader.sampler.set_epoch(epoch)

                # Train
                for step, batch in enumerate(train_dataloader):
                    if args.resume is not None and global_step < args.resume:
                        if (step + 1) % args.gradient_accumulation_steps == 0:
                            global_step += 1
                        continue

                    model.train()

                    inputs = data_reader.generate_inputs(batch, device)

                    output_dict = model(**inputs)
                    loss = output_dict['loss']

                    train_loss.update(val=loss.item(), n=batch[0].size(0))

                    if n_gpu > 1:
                        loss = loss.mean()  # mean() to average on multi-gpu.
                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps

                    if args.fp16:
                        with amp.scale_loss(loss, optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss.backward()

                    if (step + 1) % args.gradient_accumulation_steps == 0:
                        if args.max_grad_norm != -1 and args.optim != 'lamb':
                            if args.fp16:
                                torch.nn.utils.clip_grad_norm_(
                                    amp.master_params(optimizer), args.max_grad_norm)
                            else:
                                torch.nn.utils.clip_grad_norm_(
                                    model.parameters(), args.max_grad_norm)

                        optimizer.step()
                        scheduler.step()
                        model.zero_grad()
                        global_step += 1

                        if global_step % args.logging_step == 0:
                            logger.info(f"Global step: {global_step}\tTrain loss: {train_loss.avg}\t"
                                        f"Learning rate: {scheduler.get_lr()[0]}")

                            # if args.local_rank in [-1, 0]:
                            #     lr_this_step = scheduler.get_lr()[0]
                            #     summary_writer.add_scalar('lr', lr_this_step, global_step)

                            #     batch_size = batch[0].size(0)
                            #     summary_writer.add_scalar('train/loss', train_loss.avg, global_step)

                        if global_step % args.per_eval_step == 0:

                            if args.local_rank not in [-1, 0]:
                                dist.barrier()

                            if args.local_rank in [-1, 0]:
                                # Evaluation
                                model.eval()

                                if get_eval_log_fn:
                                    get_eval_log_fn(reset=True)

                                logger.info("Start evaluating")
                                for _, eval_batch in enumerate(eval_dataloader):
                                    inputs = data_reader.generate_inputs(
                                        eval_batch, device)

                                    with torch.no_grad():
                                        output_dict = model(**inputs)
                                        loss = output_dict['loss']

                                        batch_size = eval_batch[0].size(0)
                                        eval_loss.update(loss.item(), batch_size)
                                        eval_acc.update(output_dict["acc"].item(), output_dict["valid_num"])

                                eval_epoch_loss = eval_loss.avg
                                eval_epoch_acc = eval_acc.avg
                                # summary_writer.add_scalar('eval/loss', eval_epoch_loss, global_step)
                                # summary_writer.add_scalar('eval/acc', eval_epoch_acc, global_step)
                                eval_loss.reset()
                                eval_acc.reset()

                                if eval_epoch_loss < best_loss:
                                    best_loss = eval_epoch_loss
                                    model_to_save = (
                                        model.module if hasattr(
                                            model, "module") else model
                                    )  # Take care of distributed/parallel training
                                    model_to_save.save_pretrained(args.output_dir)
                                    tokenizer.save_pretrained(args.output_dir)

                                    # Good practice: save your training arguments together with the trained model
                                    torch.save(args, os.path.join(args.output_dir, "training_args.bin"))
                                    logger.info(f"Model and configuration saved in {args.output_dir}")

                                logger.info(
                                    f'Global step: {global_step}, eval_loss: {eval_epoch_loss}, '
                                    f'eval_acc: {eval_epoch_acc}, best_loss: {best_loss}. '
                                )

                                if get_eval_log_fn:
                                    logger.info(get_eval_log_fn(reset=True))

                                if args.local_rank == 0:
                                    dist.barrier()

                        if args.local_rank in [-1, 0] and \
                                args.per_save_step > 0 and global_step % args.per_save_step == 0:
                            # Save model checkpoint
                            # output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                            # Take care of distributed/parallel training
                            model_to_save = (model.module if hasattr(model, "module") else model)

                            # save to oss to avoid error                            
                            torch_save_to_oss(optimizer.state_dict(), f"/optimizer_{global_step}.pt")
                            torch_save_to_oss(scheduler.state_dict(), f"/scheduler_{global_step}.pt")
                            torch_save_to_oss(amp.state_dict(), f"/amp_{global_step}.pt")
                            logger.info("Saving optimizer and scheduler states to oss.")

                            state_dict = model_to_save.state_dict()
                            torch_save_to_oss(state_dict, f"/pytorch_model_{global_step}.bin")
                            logger.info("Saving model state dict to oss.")

                        if global_step >= t_total:
                            break

                # Free memory
                del train_dataloader
                del train_sampler
                del train_data
                del train_tensors
                torch.cuda.empty_cache()
                gc.collect()

                if global_step >= t_total:
                    break

            if global_step >= t_total:
                break

        # if args.local_rank in [-1, 0]:
        # summary_writer.close()


if __name__ == "__main__":
    main()
