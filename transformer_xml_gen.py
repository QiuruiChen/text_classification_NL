from __future__ import absolute_import, division, print_function, unicode_literals
import os
# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'eci-workspace-rachel-ce818938681b.json'
import tensorflow as tf
"""Translates text into the target language.

Target must be an ISO 639-1 language code.
See https://g.co/cloud/translate/v2/translate-reference#supported_languages
"""
# from google.cloud import translate_v2 as translate
import pandas as pd
import six
import numpy as np
import re
# translate_client = translate.Client()

# raw_data = pd.read_pickle('../data/raw_data.pkl')
# raw_data['text_length'] = raw_data['comment_text'].apply(lambda x: len(x.split(" "))-1)
# raw_data = raw_data[raw_data['text_length'] > 4]
# data_len = raw_data.shape[0]
#
# import time
# eng_com = []
#
# text = raw_data['comment_text'].values.tolist()
# text = [comment.decode('utf-8') if isinstance(text, six.binary_type) else comment for comment in text]
#
# # since the api requires 100seconds maximum requirement 1000
# for idx, comment in enumerate(text):
#     try:
#         print("idx:",idx)
#         temp_result = translate_client.translate(comment, target_language='en',source_language='nl')['translatedText']
#         eng_com.append(temp_result)
#     except:
#         print('enter sleeping!')
#         time.sleep(100.0)
#         print("idx: ",idx)
#         temp_result = translate_client.translate(comment, target_language='en',source_language='nl')['translatedText']
#         eng_com.append(temp_result)
#
# # # Text can also be a sequence of strings, in which case this method
# # # will return a sequence of results for each text.
# raw_data['english_text'] = eng_com
# raw_data.to_csv('raw_data_test.csv', index=False)


# df1 = pd.read_csv('raw_data2.csv')
# df2 = pd.read_csv('raw_data3.csv')
# pd.concat([df1,df2]).to_csv('raw_data.csv',index=False)


## write data info tffiles
# from preproc import *
# #
# csv = pd.read_csv("raw_data.csv")
# print(csv.columns)
# csv = csv[['comment_text','Huid']]
# csv['Huid'] = csv['Huid'].astype(int)
# targets = csv['Huid'].values
# preproc = Preproc(base_dir ='../data/')
# input_tensor,inp_lang = preproc.load_dataset(tuple(csv['comment_text'].values.tolist()),pad=False)
# word_index = [k for k,_ in inp_lang.word_index.items()]
#
## write the vocabulary file
# with open('../data/dutch_text.vocab', 'w') as f:
#     for item in word_index:
#         f.write("%s\n" % item)

# print(input_tensor)
# with tf.python_io.TFRecordWriter("../data/csv.tfrecords") as writer:
#     for idx,_ in enumerate(targets):
#         features, label = input_tensor[idx], targets[idx]
#         example = tf.train.Example()
#         example.features.feature["features"].int64_list.value.extend(features)
#         example.features.feature["label"].int64_list.value.append(label)
#         writer.write(example.SerializeToString())

## read the tfrecord data
# import tensorflow as tf
# tf.enable_eager_execution()
# print(tf.executing_eagerly())
# filenames = ['../data/csv.tfrecords']
# raw_dataset = tf.data.TFRecordDataset(filenames)
#
# # Create a description of the features.
# feature_description = {
#     # 'inputs':tf.io.FixedLenFeature([], tf.int64, default_value=0),
#     # 'targets': tf.io.FixedLenFeature([], tf.int64, default_value=0),
#     'features':tf.io.FixedLenSequenceFeature([], tf.int64, default_value=0,allow_missing = True),
#     'label': tf.io.FixedLenSequenceFeature([], tf.int64, default_value=0,allow_missing = True)
# }
# def _parse_function(example_proto):
#   # Parse the input `tf.Example` proto using the dictionary above.
#   return tf.io.parse_single_example(example_proto, feature_description)
#
# for raw_record in raw_dataset.take(10):
#   print(repr(raw_record))
#
# parsed_dataset = raw_dataset.map(_parse_function)
# print(parsed_dataset)
# for parsed_record in parsed_dataset.take(10):
#   print(repr(parsed_record))


# import gpt_2_simple as gpt2
# import os
# import requests
#
# model_name = "124M"
# if not os.path.isdir(os.path.join("models", model_name)):
# 	print(f"Downloading {model_name} model...")
# 	gpt2.download_gpt2(model_name=model_name)   # model is saved into current directory under /models/124M/
#
#
# file_name = "shakespeare.txt"
# if not os.path.isfile(file_name):
# 	url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
# 	data = requests.get(url)
#
# 	with open(file_name, 'w') as f:
# 		f.write(data.text)
#
# sess = gpt2.start_tf_sess()
# gpt2.finetune(sess,
#               file_name,
#               model_name=model_name,
#               steps=1000)   # steps is max number of training steps
#
# gpt2.generate(sess)

from transformers import XLMTokenizer, XLMWithLMHeadModel,XLMConfig
import argparse
import logging
import torch
from transformers_tools import *

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
    parser.add_argument( "--evaluate_during_training", action="store_true", help="Rul evaluation during training at each logging step.",)
    parser.add_argument( "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model.",)
    parser.add_argument( "--gradient_accumulation_steps",type=int,default=1, help="Number of updates steps to accumulate before performing a backward/update pass.",)
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument( "--num_train_epochs", default=1.0, type=float, help="Total number of training epochs to perform.",)
    parser.add_argument( "--max_steps",default=-1,type=int, help="If > 0: set total number of training steps to perform. Override num_train_epochs.", )
    parser.add_argument("--eval_batch_size", default= 32, type=int, help="batch size during evaluation")
    parser.add_argument("--train_batch_size", default= 32, type=int, help="batch size during training")
    parser.add_argument("--warmup_steps", default=30000, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=int, default=50, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=50, help="Save checkpoint every X updates steps.")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    args = parser.parse_args()

    # device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    print("how many gpu is available?", args.n_gpu)

    args.device = device
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    set_seed(args)

    config = XLMConfig().from_pretrained('xlm-mlm-17-1280')
    config.lang_id = config.lang2id['nl']
    print(config)
    tokenizer = XLMTokenizer.from_pretrained('xlm-mlm-17-1280',config = config)
    model = XLMWithLMHeadModel.from_pretrained('xlm-mlm-17-1280', config = config)
    logger.info("Training/evaluation parameters %s", args)
    train_dataset = load_and_cache_examples(args, 'dutchwords', tokenizer, evaluate=False)
    global_step, tr_loss = train(args, train_dataset, model, tokenizer)
    logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

if __name__ == "__main__":
    main()



