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

from transformers import *
from transformers_tools import *

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)

ALL_MODELS = sum(
    (
        tuple(conf.pretrained_config_archive_map.keys())
        for conf in ( BertConfig, XLNetConfig, XLMConfig, RobertaConfig, DistilBertConfig, AlbertConfig, XLMRobertaConfig, )
    ),
    (),
)

MODEL_CLASSES = {
    "bert": (BertConfig, BertForSequenceClassificationMultiLabel, BertTokenizer),
    "xlm": (XLMConfig, XLMForSequenceClassificationMultiLabel, XLMTokenizer),
    "distilbert": (DistilBertConfig, DistilBertForSequenceClassificationMultiLabel, DistilBertTokenizer),
    "xlmroberta": (XLMRobertaConfig, XLMRobertaForSequenceClassificationMultiLabel, XLMRobertaTokenizer),
}

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
    parser.add_argument( "--config_name", default="xlm-roberta-base", type=str,
        help="Pretrained config name or path if not the same as model_name. Cross-lingual pretrained models are:"
        "xlm-mlm-17-1280;xlm-roberta-base;"
        "bert-base-multilingual-cased;distilbert-base-multilingual-cased;")

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
        num_labels=num_labels,)
    tokenizer = tokenizer_class.from_pretrained(
        args.config_name, # args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,)
    model = model_class.from_pretrained(
        args.config_name,
        config=config,)

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

