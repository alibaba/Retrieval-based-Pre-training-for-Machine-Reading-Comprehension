# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
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
""" Finetuning multi-lingual models on XNLI (e.g. Bert, DistilBERT, XLM).
    Adapted from `examples/text-classification/run_glue.py`"""

import argparse
import glob
import logging
import json
import os
import random
from typing import Dict, List

import numpy as np
import torch
from torch import distributed as dist
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    AutoTokenizer,
    get_linear_schedule_with_warmup
)

from general_util.logger import setting_logger
from general_util.mixin import PredictionMixin
from models import model_map, get_added_args, clean_state_dict
from processors import get_processor
from oss_utils import load_buffer_from_oss

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, model, tokenizer, processor):
    """ Train the model """
    tb_writer = None
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter(logdir=('runs/' + '-'.join(args.output_dir.split('/')[1:])))

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,
                                  collate_fn=getattr(processor, "collate_fn", None), num_workers=args.num_workers)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    exec('args.adam_betas = ' + args.adam_betas)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0
        },
    ]

    if args.warmup_steps == 0:
        args.warmup_steps = int(t_total * args.warmup_proportion)

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate,
                      eps=args.adam_epsilon, betas=args.adam_betas,
                      correct_bias=(not args.no_bias_correction))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )
    logger.info(optimizer)

    from torch.cuda.amp.grad_scaler import GradScaler
    scaler = GradScaler()

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
            os.path.join(args.model_name_or_path, "scheduler.pt")
    ) and args.resume:
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (dist.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if args.resume and os.path.exists(args.model_name_or_path) and 'checkpoint' in args.model_name_or_path:
        # set global_step to global_step of last saved checkpoint from model path
        tmp = args.model_name_or_path.split("-")[1].split("/")[0]
        if isinstance(tmp, int):
            global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
            epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info("  Continuing training from global step %d", global_step)
            logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

    best_results = {}
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    set_seed(args)  # Added here for reproducibility
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0],
                              dynamic_ncols=True)
        for step, batch in enumerate(epoch_iterator):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            inputs = processor.generate_inputs(batch, args.device)
            with torch.cuda.amp.autocast():
                outputs = model(**inputs)
                loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            scaler.scale(loss).backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()

                scheduler.step()  # Update learning rate schedule
                optimizer.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0]:
                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar("loss", (tr_loss - logging_loss), global_step)
                    logging_loss = tr_loss

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if (
                            args.local_rank == -1 and args.evaluate_during_training
                    ):  # Only evaluate when single GPU otherwise metrics may not average well
                        torch.cuda.empty_cache()
                        results = evaluate(args, model, tokenizer, processor, if_eval=True)
                        for key, value in results.items():
                            tb_writer.add_scalar("eval_{}".format(key), value, global_step)

                    # Save best model
                    if args.save_metric not in best_results or results[args.save_metric] > best_results[
                            args.save_metric]:
                        logger.info(f"Saving model checkpoint with best {args.save_metric} to {args.output_dir}")
                        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
                        # They can then be reloaded using `from_pretrained()`
                        model_to_save = (
                            model.module if hasattr(model, "module") else model
                        )  # Take care of distributed/parallel training
                        model_to_save.save_pretrained(args.output_dir)
                        tokenizer.save_pretrained(args.output_dir)

                        # Good practice: save your training arguments together with the trained model
                        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

                        best_results.update(results)

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)

            if 0 < args.max_steps < global_step:
                epoch_iterator.close()
                break
        
        torch.cuda.empty_cache()
        
        if 0 < args.max_steps < global_step:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, processor, prefix="", if_eval=True):
    eval_outputs_dirs = (args.output_dir,)

    has_prediction_mixin = False
    predict_mix_reset = None
    predict_mix_get_predict = None
    if isinstance(model, PredictionMixin):
        has_prediction_mixin = True
        predict_mix_reset = model.reset_predict_tensors
        predict_mix_get_predict = model.get_predict_tensors
    elif hasattr(model, "module") and isinstance(model.module, PredictionMixin):
        has_prediction_mixin = True
        predict_mix_reset = model.module.reset_predict_tensors
        predict_mix_get_predict = model.module.get_predict_tensors

    results = {}
    for eval_output_dir in eval_outputs_dirs:
        eval_dataset, examples = load_and_cache_examples(args, tokenizer, processor, evaluate=1 if if_eval else 2)

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size,
                                     collate_fn=getattr(processor, "collate_fn", None), num_workers=args.num_workers)

        # multi-gpu eval
        if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0

        if has_prediction_mixin:
            predict_mix_reset()

        preds = None
        out_label_ids = None
        indexes = []
        for batch in tqdm(eval_dataloader, desc="Evaluating", dynamic_ncols=True):
            model.eval()
            inputs = processor.generate_inputs(batch, args.device)

            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    outputs = model(**inputs)
                    tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item() * batch[0].size(0)
            # nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)
            indexes.extend(batch[-1].tolist())

        eval_loss = eval_loss / len(eval_dataloader)
        preds = np.argmax(preds, axis=1)
        preds = preds.tolist()
        out_label_ids = out_label_ids.tolist()
        reranked_examples = [examples[idx] for idx in indexes]
        result = processor.compute_metrics(preds, out_label_ids, reranked_examples)
        results.update(result)
        results['loss'] = eval_loss

        if not os.path.exists(os.path.join(eval_output_dir, prefix)):
            os.mkdir(os.path.join(eval_output_dir, prefix))

        output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results {} *****".format(prefix))
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

        if has_prediction_mixin:
            predicted_tensors: Dict[str, List] = predict_mix_get_predict()
            predict_mix_reset()
        else:
            predicted_tensors = {}

        predictions = processor.write_predictions(preds, out_label_ids, reranked_examples,
                                                  predicted_tensors=predicted_tensors)
        prediction_file = os.path.join(eval_output_dir, prefix, "eval_predictions.json")
        with open(prediction_file, "w") as f:
            json.dump(predictions, f, indent=2)
        # For ReClor
        if isinstance(predictions, list):
            prediction_file = os.path.join(eval_output_dir, prefix, "eval_predictions.npy")
            np.save(prediction_file, predictions)

    return results


def load_and_cache_examples(args, tokenizer, processor, evaluate=0):
    # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    if args.local_rank not in [-1, 0] and not evaluate:
        dist.barrier()

    assert evaluate in [0, 1, 2]

    if evaluate == 0:
        input_file = args.train_file
    elif evaluate == 1:
        input_file = args.dev_file
    elif evaluate == 2:
        input_file = args.test_file
    else:
        raise ValueError('evaluate should be in {0, 1, 2}')

    # Load data features from cache or dataset file
    cached_features_file = f"{input_file}_{processor.opt['cache_suffix']}"

    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading tensors from cached file %s", cached_features_file)
        examples, tensors = torch.load(cached_features_file)
    else:
        logger.info("Creating tensors from dataset file %s", input_file)
        examples = processor.read(input_file)
        tensors = processor.convert_examples_to_tensors(examples, tokenizer)
        if args.local_rank in [-1, 0]:
            logger.info("Saving examples and tensors into cached file %s", cached_features_file)
            torch.save((examples, tensors), cached_features_file)

    # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    if args.local_rank == 0 and not evaluate:
        dist.barrier()

    dataset = TensorDataset(*tensors)
    return dataset, examples


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--train_file",
        default=None,
        type=str
    )
    parser.add_argument("--dev_file", default=None, type=str)
    parser.add_argument("--test_file", default=None, type=str)
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--reader_name",
        default=None,
        type=str,
        required=True,
        help="The name to initialize processors."
    )
    parser.add_argument(
        "--model_name",
        default=None,
        type=str,
        required=True,
        help="The name to initialize model."
    )
    parser.add_argument("--pretrain", default=None, type=str)

    # Other parameters
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default=None,
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
             "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument("--max_query_length", default=128, type=int)
    parser.add_argument("--num_workers", default=12, type=int)
    parser.add_argument("--if_parallel", default=False, action='store_true',
                        help="If use multiprocess to convert examples into features.")

    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the test set.")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Rul evaluation during training at each logging step."
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )
    parser.add_argument("--fp16_opt_level", default="O1", type=str)

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--adam_betas", default="(0.9, 0.999)", type=str)
    # parser.add_argument("--max_grad_norm", default=-1., type=float)
    parser.add_argument("--no_bias_correction", default=False, action='store_true')
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--warmup_proportion", default=0.0, type=float)
    parser.add_argument("--save_metric", type=str, default="acc", help="Metric for saving models.")
    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")

    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    parser.add_argument("--resume", default=False, action='store_true')

    parser.add_argument("--num_labels", type=int, default=2)

    # Bert SSP SG
    parser.add_argument("--graph_layers", type=int, default=3)
    parser.add_argument("--cls_n_head", type=int, default=12)
    parser.add_argument("--cls_d_head", type=int, default=64)

    # Bert SSP ES
    parser.add_argument("--es_layer_id", type=int, default=9)

    # Bert SSP KG
    parser.add_argument("--kg_layer_ids", type=str, default='6')
    parser.add_argument("--edge_type", type=str, default='knowledge')

    # Bert HA for MCRC
    parser.add_argument("--cls_type", default=0, type=int)
    parser.add_argument("--no_residual", default=False, action='store_true')
    parser.add_argument("--use_new_pro", default=False, action='store_true')
    parser.add_argument("--no_ff", default=False, action='store_true')
    parser.add_argument("--re_initialize", default=False, action='store_true')

    # Query based pre-training config
    parser.add_argument("--query_dropout", default=0.2, type=float)
    parser.add_argument("--sr_query_dropout", default=0.2, type=float)
    parser.add_argument("--lm_query_dropout", default=0.1, type=float)
    parser.add_argument("--word_dropout", default=0.0, type=float)
    parser.add_argument("--share_ssp_sum", default=False, action='store_true')

    parser.add_argument("--oss_pretrain", default=None, type=str)

    args = parser.parse_args()

    if (
            os.path.exists(args.output_dir)
            and os.listdir(args.output_dir)
            and args.do_train
            and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        dist.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    global logger
    logger = setting_logger(args.output_dir)
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: True",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
    )
    logger.info(args)

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    # Make sure only the first process in distributed training will download model & vocab
    if args.local_rank not in [-1, 0]:
        dist.barrier()

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        # do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir,
    )

    model_class = model_map[args.model_name]

    if args.pretrain is not None:
        logger.info(f"Loading pre-trained model state dict from {args.pretrain}")
        pretrain_state_dict = torch.load(args.pretrain, map_location='cpu')
    elif args.oss_pretrain is not None:
        logger.info(f"Loading pre-trained model state dict from oss: {args.oss_pretrain}")
        pretrain_state_dict = torch.load(load_buffer_from_oss(args.oss_pretrain), map_location='cpu')
        # pretrain_state_dict = clean_state_dict(model_class, pretrain_state_dict)
    else:
        pretrain_state_dict = None

    model = model_class.from_pretrained(args.model_name_or_path, **get_added_args(args, model_class),
                                        state_dict=pretrain_state_dict)
    args.base_model_type = model_class.config_class.model_type
    processor = get_processor(args)

    # Make sure only the first process in distributed training will download model & vocab
    if args.local_rank == 0:
        dist.barrier()

    model.to(args.device)

    # Training
    if args.do_train:
        train_dataset, _ = load_and_cache_examples(args, tokenizer, processor, evaluate=0)
        global_step, tr_loss = train(args, train_dataset, model, tokenizer, processor)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Test
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info(" the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""

            model = model_class.from_pretrained(checkpoint, **get_added_args(args, model_class))
            model.to(args.device)

            if args.test_file:
                prefix = 'test' + prefix

            result = evaluate(args, model, tokenizer, processor, prefix=prefix,
                              if_eval=(False if args.test_file else True))  # No test set
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            results.update(result)

    return results


if __name__ == "__main__":
    main()
