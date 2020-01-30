from __future__ import absolute_import, division, print_function, unicode_literals
import os

from transformers import XLMTokenizer, XLMWithLMHeadModel,XLMConfig
import argparse
import logging
import torch
from transformers_tools import *
from transformers import *

processor = DutchwordsProcessor()
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--data_dir",default='./data/', type=str, required=False,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
    parser.add_argument( "--output_dir", default='./model_saves/transformer_XLM_gen/', type=str, required=False,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    # Other parameters
    parser.add_argument("--config_name", default="xlm-mlm-17-1280", type=str,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument("--task_type", default="generation", type=str, help="Generation or classification task")
    parser.add_argument("--max_seq_length", default=128, type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument( "--fp16", action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action="store_true", help="Rul evaluation during training at each logging step.",)
    parser.add_argument("--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model.",)
    parser.add_argument("--gradient_accumulation_steps",type=int,default=1, help="Number of updates steps to accumulate before performing a backward/update pass.",)
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument( "--num_train_epochs", default=1.0, type=float, help="Total number of training epochs to perform.",)
    parser.add_argument( "--max_steps",default=-1,type=int, help="If > 0: set total number of training steps to perform. Override num_train_epochs.", )
    parser.add_argument("--eval_batch_size", default= 32, type=int, help="batch size during evaluation")
    parser.add_argument("--train_batch_size", default= 32, type=int, help="batch size during training")
    parser.add_argument("--warmup_steps", default=10, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=int, default=50, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=0, help="Save checkpoint every X updates steps.")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    args = parser.parse_args()

    # device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.device = device

    # args.device = device
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    set_seed(args)
    logging.info("%d gpus avaliable", args.n_gpu)

    # until now only XLM model can generate dutch
    config = XLMConfig().from_pretrained('xlm-mlm-17-1280')
    config.lang_id = config.lang2id['nl']
    config.early_stopping = True
    tokenizer = XLMTokenizer.from_pretrained('xlm-mlm-17-1280',config = config)
    model = XLMWithLMHeadModel.from_pretrained('xlm-mlm-17-1280', config = config)


    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)
    logger.info("The configuration information is %s",config)
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
        model_to_save = (model.module if hasattr(model, "module") else model)
        model_to_save.save_pretrained(args.output_dir)
        # tokenizer.save_pretrained(args.output_dir)

        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

if __name__ == "__main__":
    main()
