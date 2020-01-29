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
""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet, RoBERTa, Albert, XLM-RoBERTa)."""

# bad code: change class RobertaForSequenceClassification(BertPreTrainedModel) in modeling_roberta.py
# loss from:
 # loss_fct = CrossEntropyLoss()
# loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
# into:
# loss_fct = torch.nn.BCEWithLogitsLoss()
# loss = loss_fct(logits.view(-1, self.num_labels), labels.type_as(logits)) # change the label into the same type of the logits


# todo: no distribution training yet

import argparse
import glob
import json
import logging
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from tqdm import tqdm, trange
from seqeval import metrics

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    AlbertConfig,
    AlbertForSequenceClassification,
    AlbertTokenizer,
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer,
    DistilBertConfig,
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    RobertaConfig,
    RobertaForSequenceClassification,
    RobertaTokenizer,
    XLMConfig,
    XLMForSequenceClassification,
    XLMRobertaConfig,
    XLMRobertaTokenizer,
    XLMTokenizer,
    XLNetConfig,
    XLNetForSequenceClassification,
    XLNetTokenizer,
    get_linear_schedule_with_warmup,
)

from transformers_tools import *

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)

ALL_MODELS = sum(
    (
        tuple(conf.pretrained_config_archive_map.keys())
        for conf in (
            BertConfig,
            XLNetConfig,
            XLMConfig,
            RobertaConfig,
            DistilBertConfig,
            AlbertConfig,
            XLMRobertaConfig, )
    ),
    (),
)

MODEL_CLASSES = {
    "bert": (BertConfig, BertForSequenceClassification, BertTokenizer),
    "xlnet": (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    "xlm": (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    "roberta": (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    "distilbert": (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer),
    "albert": (AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer),
    "xlmroberta": (XLMRobertaConfig, XLMRobertaForSequenceClassificationMultiLabel, XLMRobertaTokenizer),
}

# def train(args, train_dataset, model, tokenizer):
#     """ Train the model """
#     tb_writer = SummaryWriter()
#
#     # args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
#     args.train_batch_size = 64
#     train_sampler = RandomSampler(train_dataset) #if args.local_rank == -1 else DistributedSampler(train_dataset)
#     train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
#
#     if args.max_steps > 0:
#         t_total = args.max_steps
#         args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
#     else:
#         t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
#
#     # Prepare optimizer and schedule (linear warmup and decay)
#     no_decay = ["bias", "LayerNorm.weight"]
#     optimizer_grouped_parameters = [
#         {
#             "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
#             "weight_decay": args.weight_decay,
#         },
#         {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
#     ]
#
#     optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
#     scheduler = get_linear_schedule_with_warmup(
#         optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
#     )
#
#     # multi-gpu training (should be after apex fp16 initialization)
#     if args.n_gpu > 1:
#         model = torch.nn.DataParallel(model)
#
#     # Train!
#     logger.info("***** Running training *****")
#     logger.info("  Num examples = %d", len(train_dataset))
#     logger.info("  Num Epochs = %d", args.num_train_epochs)
#     logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)
#     logger.info(
#         "  Total train batch size (w. parallel, distributed & accumulation) = %d",
#         args.train_batch_size
#         * args.gradient_accumulation_steps
#         * (1), #(torch.distributed.get_world_size() if args.local_rank != -1 else 1),
#     )
#     logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
#     logger.info("  Total optimization steps = %d", t_total)
#
#     global_step = 0
#     epochs_trained = 0
#     steps_trained_in_current_epoch = 0
#     # Check if continuing training from a checkpoint
#     output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
#     if os.path.exists(output_dir):
#         # set global_step to gobal_step of last saved checkpoint from model path
#         global_step = int(output_dir.split("-")[-1].split("/")[0])
#         epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
#         steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)
#
#         logger.info("  Continuing training from checkpoint, will skip to saved global_step")
#         logger.info("  Continuing training from epoch %d", epochs_trained)
#         logger.info("  Continuing training from global step %d", global_step)
#         logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
#
#     tr_loss, logging_loss = 0.0, 0.0
#     model.zero_grad()
#     train_iterator = trange(
#         epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=False #args.local_rank not in [-1, 0],
#     )
#     set_seed(args)  # Added here for reproductibility
#     for _ in train_iterator:
#         epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=False) #args.local_rank not in [-1, 0])
#         for step, batch in enumerate(epoch_iterator):
#
#             # Skip past any already trained steps if resuming training
#             if steps_trained_in_current_epoch > 0:
#                 steps_trained_in_current_epoch -= 1
#                 continue
#
#             model.train()
#             batch = tuple(t.to(args.device) for t in batch)
#             inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
#             if args.model_type != "distilbert":
#                 inputs["token_type_ids"] = (
#                     batch[2] if args.model_type in ["bert", "xlnet", "albert"] else None
#                 )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
#             # outputs = model(inputs['input_ids'], labels=inputs['labels'])
#             outputs = model(**inputs)
#             loss = outputs[0]  # model outputs are always tuple in transformers (see doc)
#
#             if args.n_gpu > 1:
#                 loss = loss.mean()  # mean() to average on multi-gpu parallel training
#             if args.gradient_accumulation_steps > 1:
#                 loss = loss / args.gradient_accumulation_steps
#
#             loss.backward()
#
#             tr_loss += loss.item()
#             if (step + 1) % args.gradient_accumulation_steps == 0:
#                 torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
#
#                 optimizer.step()
#                 scheduler.step()  # Update learning rate schedule
#                 model.zero_grad()
#                 global_step += 1
#
#                 # if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
#                 if args.logging_steps > 0 and global_step % args.logging_steps == 0:
#                     logs = {}
#
#                     loss_scalar = (tr_loss - logging_loss) / args.logging_steps
#                     learning_rate_scalar = scheduler.get_lr()[0]
#                     logs["learning_rate"] = learning_rate_scalar
#                     logs["loss"] = loss_scalar
#                     logging_loss = tr_loss
#
#                     for key, value in logs.items():
#                         tb_writer.add_scalar(key, value, global_step)
#                     print(json.dumps({**logs, **{"step": global_step}}))
#
#                 # if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
#                 if args.save_steps > 0 and global_step % args.save_steps == 0:
#                     # Save model checkpoint
#                     output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
#                     if not os.path.exists(output_dir):
#                         os.makedirs(output_dir)
#                     model_to_save = (
#                         model.module if hasattr(model, "module") else model
#                     )  # Take care of distributed/parallel training
#                     model_to_save.save_pretrained(output_dir)
#                     tokenizer.save_pretrained(output_dir)
#
#                     torch.save(args, os.path.join(output_dir, "training_args.bin"))
#                     logger.info("Saving model checkpoint to %s", output_dir)
#
#                     torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
#                     torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
#                     logger.info("Saving optimizer and scheduler states to %s", output_dir)
#
#             if args.max_steps > 0 and global_step > args.max_steps:
#                 epoch_iterator.close()
#                 break
#         if args.max_steps > 0 and global_step > args.max_steps:
#             train_iterator.close()
#             break
#
#     tb_writer.close()
#
#     return global_step, tr_loss / global_step

# def load_and_cache_examples(args, task, tokenizer, evaluate=False):
#     processor = DutchwordsProcessor()
#     output_mode = "classification"
#     # Load data features from cache or dataset file
#     cached_features_file = os.path.join(
#         args.data_dir,
#         "cached_{}_{}_{}_{}".format(
#             "dev" if evaluate else "train",
#             str(args.config_name),
#             str(args.max_seq_length),
#             str(task),
#         ),
#     )
#     if os.path.exists(cached_features_file) :
#         logger.info("Loading features from cached file %s", cached_features_file)
#         features = torch.load(cached_features_file)
#     else:
#         logger.info("Creating features from dataset file at %s", args.data_dir)
#         label_list = processor.get_labels()
#         examples = (
#             processor.get_dev_examples(args.data_dir) if evaluate else processor.get_train_examples(args.data_dir)
#         )
#
#         features = glue_convert_examples_to_featuresMultilabel(
#             examples,
#             tokenizer,
#             label_list=label_list,
#             max_length=args.max_seq_length,
#             pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
#             pad_token_segment_id=0,
#             task='dutchwords'
#         )
#
#         logger.info("Saving features into cached file %s", cached_features_file)
#         torch.save(features, cached_features_file)
#
#     # Convert to Tensors and build dataset
#     all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
#     all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
#     all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
#     if output_mode == "classification":
#         all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
#     elif output_mode == "regression":
#         all_labels = torch.tensor([f.label for f in features], dtype=torch.float)
#
#     dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
#     return dataset


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--data_dir", default='./data/', type=str, required=False,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
    parser.add_argument( "--model_type", default='xlmroberta', type=str, required=False,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument( "--output_dir", default='./model_saves/Transformer_xmlroberta/', type=str, required=False,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    # Other parameters
    parser.add_argument( "--config_name", default="xlm-roberta-base", type=str,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument("--task_type", default="classification", type=str, help="Generation or classification task")
    parser.add_argument( "--max_seq_length", default=128, type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument( "--evaluate_during_training", action="store_true", help="Rul evaluation during training at each logging step.",)
    parser.add_argument( "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model.",)

    parser.add_argument( "--gradient_accumulation_steps",type=int,default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument( "--num_train_epochs", default=1.0, type=float, help="Total number of training epochs to perform.",)
    parser.add_argument( "--max_steps",default=-1,type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--eval_batch_size", default= 32, type=int, help="batch size during evaluation")
    parser.add_argument("--train_batch_size", default= 32, type=int, help="batch size during training")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=int, default=50, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=50, help="Save checkpoint every X updates steps.")
    parser.add_argument( "--eval_all_checkpoints", action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument( "--fp16", action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    set_seed(args)

    processor = DutchwordsProcessor()
    args.output_mode = "classification"

    label_list = processor.get_labels()
    num_labels = len(label_list)

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(
        args.config_name, # args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels,
    )
    tokenizer = tokenizer_class.from_pretrained(
        args.config_name, # args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
    )
    model = model_class.from_pretrained(
        args.config_name,
        config=config,
    )

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, 'dutchwords', tokenizer, evaluate=False)
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    if args.do_train:
        # Create output directory if needed
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

        model = model_class.from_pretrained(args.output_dir)
        tokenizer = tokenizer_class.from_pretrained(args.output_dir)
        model.to(args.device)

    # Evaluation
    results = {}
    # if args.do_eval and args.local_rank in [-1, 0]:
    if args.do_eval:
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        # use the last saved checkpoints
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""
            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            result = evaluate(args, model, tokenizer, prefix=prefix)
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            results.update(result)

    return results


if __name__ == "__main__":
    main()

