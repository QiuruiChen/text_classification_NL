# The bare DistilBERT encoder/transformer outputing raw hidden-states without any specific head on top. DistilBERT is a small, fast, cheap and light Transformer model trained by distilling Bert base. It has 40% less parameters than bert-base-uncased, runs 60% faster while preserving over 95% of Bert’s performances as measured on the GLUE language understanding benchmark.
import tensorflow as tf
from transformers import DistilBertTokenizer, TFDistilBertModel

# tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')
# model = TFDistilBertModel.from_pretrained('distilbert-base-multilingual-cased')
# input_ids = tf.constant(tokenizer.encode("Hello, my dog is cute"))[None, :]  # Batch size 1
# outputs[0].shape ==> (1,10,768)
# input_ids ==> (1,10)


# distilbert-base-multilingual-cased
# 16-layer, 1280-hidden, 16-heads
# XLM model trained with MLM (Masked Language Modeling) on 17 languages.
# en-fr-es-de-it-pt-nl-sv-pl-ru-ar-tr-zh-ja-ko-hi-vi

# Todo: figure out the configurations
import tensorflow as tf
from transformers import *
from transformers_tools import *
import tensorflow_addons as tfa
# Bert classfiication, applying "multilingual- uncased-model" (without any preprocessing)
# source code in Transform library was modified:
# glue_convert_examples_to_featuresMultilabel: creatd by glue_convert_examples_to_features
# TFBertForSequenceMultiLabelClassification: created by TFBertForSequenceClassification
import os
import logging
logger = tf.get_logger()
logger.setLevel(logging.INFO)
print(tf.__version__)

import datetime
import pandas as pd
import numpy as np

from sklearn.metrics import hamming_loss,accuracy_score
from seqeval import metrics

try:
    from fastprogress import master_bar, progress_bar
except ImportError:
    from fastprogress.fastprogress import master_bar, progress_bar
import math
from seqeval import metrics

BATCH_SIZE = 32
NUM_TRAIN_EPOCHS = 1
LOGGING_STEPS = 2
EVALUATE_DURING_TRAINING = True
SAVE_STEPS = 2 # save checkpoint at every 2 epoches
MODEL_TYPE = "Bert"
if MODEL_TYPE == "distilBert":
    OUTPUT_DIR = './model_saves/distillBERT/'
elif MODEL_TYPE == "Bert":
    OUTPUT_DIR = './model_saves/Transformer_bert/'

N_DEVICE = 1
SEQ_LEN = 128


def dataset_get(batch_size,model_type,training):

    train_size = 19067
    val_size = 4764

    # Load dataset, tokenizer, model from pretrained model/vocabulary
    # see what is possible: https://huggingface.co/transformers/pretrained_models.html
    if model_type == "Bert":
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased',add_special_tokens=True)
    elif model_type =="distilBert":
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased',add_special_tokens=True) #bert-base-multilingual-cased
    # train dataset
    df_train = pd.read_csv('./data/transformer_bert_train.csv')
    prob_list_names = ['Persoonlijke_zorg', 'Medicatie', 'Huid', 'Circulatie', 'Voeding','Urinewegfunctie', 'Neuro_musculaire_skeletfunctie', 'Cognitie', 'Pijn',
    'Darmfunctie', 'Geestelijke_gezondheid', 'Ademhaling', 'Mantelzorg_zorg_voor_kind_huisgenoot', 'Fysieke_activiteit', 'Zicht','other']

    df_train['label'] = [np.asarray(val) for val in df_train.reset_index()[prob_list_names].values.tolist()]
    dataset = tf.data.Dataset.from_tensor_slices({'sentence':df_train['comment_text'].values,
                                                  'label':df_train[prob_list_names].values})
    # validatiojn dataset
    df_val = pd.read_csv('./data/transformer_bert_val.csv')
    df_val['label'] = [np.asarray(val) for val in df_val.reset_index()[prob_list_names].values.tolist()]
    dataset_val = tf.data.Dataset.from_tensor_slices({'sentence':df_val['comment_text'].values,
                                                  'label':df_val[prob_list_names].values})
    # test dataset
    df_test = pd.read_csv('./data/transformer_bert_test.csv')
    prob_list_names = ['Persoonlijke_zorg', 'Medicatie', 'Huid', 'Circulatie', 'Voeding','Urinewegfunctie', 'Neuro_musculaire_skeletfunctie', 'Cognitie', 'Pijn',
    'Darmfunctie', 'Geestelijke_gezondheid', 'Ademhaling', 'Mantelzorg_zorg_voor_kind_huisgenoot', 'Fysieke_activiteit', 'Zicht','other']

    df_test['label'] = [np.asarray(val) for val in df_test.reset_index()[prob_list_names].values.tolist()]
    dataset_test = tf.data.Dataset.from_tensor_slices({'sentence':df_test['comment_text'].values,
                                                       'label':df_test[prob_list_names].values})

    for val in dataset.take(5):
      print ('Features: {}, Target: {}'.format(val['sentence'], val['label']))

    train_dataset = glue_convert_examples_to_featuresMultilabel(dataset, tokenizer, max_length=SEQ_LEN, task='dutchwords')
    val_dataset = glue_convert_examples_to_featuresMultilabel(dataset_val, tokenizer, max_length=SEQ_LEN, task='dutchwords')
    test_dataset = glue_convert_examples_to_featuresMultilabel(dataset_test, tokenizer, max_length=SEQ_LEN, task='dutchwords')

    for val in train_dataset.take(1):
      # print ('Features: {}, Target: {}'.format(val['sentence'], val['label']))
      print(val)

    train_dataset= train_dataset.map(lambda inputs,label: (inputs['input_ids'], label))
    # train_x, trian_y = train_dataset
    train_dat_batched = train_dataset.shuffle(100).batch(batch_size).repeat(2)

    val_dataset = val_dataset.map(lambda inputs,label: (inputs['input_ids'], label))
    valid_dat_batched = val_dataset.batch(batch_size)
    test_dataset = test_dataset.map(lambda inputs,label: (inputs['input_ids'], label))
    test_dat_batched = test_dataset.batch(batch_size)
    if training == True:
        return valid_dat_batched,test_dat_batched,train_dat_batched,train_size,val_size,tokenizer
    elif training == False:
        return test_dataset, df_test.shape[0]


## =======
##  train
## =======
def configuration_get(model_type):
    if model_type == "Bert":
        configuration = BertConfig.from_pretrained('bert-base-multilingual-cased')
        configuration.output_hidden_states=True
    elif model_type =="distilBert":
        configuration = DistilBertConfig.from_pretrained('distilbert-base-multilingual-cased')
    configuration.early_stopping = True
    configuration.num_labels= 16
    print(configuration)
    return configuration

def model_build(configuration,model_type):
    '''
    :param configuration:
    :return:
    '''
    input_layer = tf.keras.layers.Input(shape=(SEQ_LEN,), dtype = 'int64')
    if model_type == "Bert":
        model_base = TFBertModel.from_pretrained('bert-base-multilingual-cased', config= configuration)(input_layer) # distilbert-base-multilingual-cased
        hidden_state = model_base[2][0] # shape (batch_size, sequence_length, hidden_size)
        x = tf.keras.layers.GlobalAveragePooling1D()(hidden_state) # (batch_size, hidden_size)
    elif model_type =="distilBert":
        model_base = TFDistilBertModel.from_pretrained('distilbert-base-multilingual-cased', config= configuration)(input_layer) # distilbert-base-multilingual-cased
        # x = model_base[0]
        summary_type = 'mean'
        if summary_type == 'last':
            x = model_base[0][:, -1]
        elif summary_type == 'first':
            x = model_base[0][:, 0]
        elif summary_type == 'mean':
            x = tf.reduce_mean(model_base[0], axis=1)

    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(16 ,kernel_initializer= tf.keras.initializers.TruncatedNormal(stddev=0.02),
                              name='summary', activation = 'sigmoid')(x)

    model = tf.keras.models.Model(input_layer, x)
    return model

## train on the high-level
def train_high_level(model_type):

    class_config = configuration_get(model_type)
    model = model_build(configuration=class_config,model_type=model_type)

    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    metric = tf.keras.metrics.BinaryAccuracy('accuracy')
    #   "optimizer": "adam_inverse_sqrt,lr=0.00005,warmup_updates=30000,beta1=0.9,beta2=0.999,weight_decay=0.01,eps=0.000001",
    # optimizer = tfa.optimizers.AdamW(weight_decay = 0.01,learning_rate=0.00005, beta_1=0.9, beta_2=0.999, epsilon=0.000001, amsgrad=False, name='AdamW',)
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

    valid_dat_batched,test_dat_batched,train_dat_batched,train_size,val_size,tokenizer = dataset_get(batch_size=BATCH_SIZE,model_type=model_type,training=True)

    model.fit(train_dat_batched, epochs=NUM_TRAIN_EPOCHS,
                        steps_per_epoch= int(train_size // BATCH_SIZE),
                        validation_data=valid_dat_batched,
                        validation_steps=int(val_size // BATCH_SIZE)) #115,7

    model.save_weights(OUTPUT_DIR+'weights/')
    model.layers[1].save_pretrained(OUTPUT_DIR+'pretrained/')

    # model.load_weights(OUTPUT_DIR+'base_model/weights/)

# train_high_level(MODEL_TYPE)

## ===========
##  evaluate
## ===========

def evaluate_high_level(model_type):
    class_config = configuration_get(model_type)
    model = model_build(configuration=class_config,model_type=model_type)

    model.load_weights(OUTPUT_DIR+'weights/')
    model.layers[1].from_pretrained(OUTPUT_DIR+'pretrained/')

    model.summary()
    # inputs, targets = next(iter(val_dataset.batch(val_size)))
    test_dataset, test_num = dataset_get(batch_size=BATCH_SIZE,model_type=model_type,training=False)

    inputs, targets = next(iter(test_dataset.batch(2000))) #test_num
    predictions = model(inputs)
    # Load the TensorFlow model in PyTorch for inspection
    prediction = tf.cast(predictions>0.5, tf.int64)

    y_true = [[str(ele) for ele in label]for label in targets.numpy()]
    y_pred = [[str(ele) for ele in pred]for pred in prediction.numpy()]
    report = metrics.classification_report(y_true, y_pred, digits=4)

    print("Eval at step " + str(metrics.precision_score(y_true, y_pred)))
    print("eval_loss: " + str(metrics.recall_score(y_true, y_pred)))
    print("f1 score:" +str(metrics.f1_score(y_true, y_pred)))
    print("accuracy is:" + str(metrics.accuracy_score(y_true,y_pred)))
    print("report is", report)

    print("the exact accuracy is: ",accuracy_score(targets.numpy(), prediction.numpy()))
    print("the hamming loss is: ", hamming_loss(targets.numpy(), prediction.numpy()))

    prob_list_names = ['Persoonlijke_zorg', 'Medicatie', 'Huid', 'Circulatie', 'Voeding','Urinewegfunctie', 'Neuro_musculaire_skeletfunctie', 'Cognitie', 'Pijn',
    'Darmfunctie', 'Geestelijke_gezondheid', 'Ademhaling', 'Mantelzorg_zorg_voor_kind_huisgenoot', 'Fysieke_activiteit', 'Zicht','other']

    for i in range(0,16):
        print("the accuracy for the problem",prob_list_names[i])
        print("acc is:", str((targets.numpy()[:,i]== prediction.numpy()[:,i]).sum() / targets.numpy().shape[0]))

evaluate_high_level(model_type=MODEL_TYPE)

## =============
## visualization
## =============
# max_length = SEQ_LEN
# # sentence is: Evt vanmiddag doen zat nauwelijks ontlasting in het zakje
# # actual label is: Zicht,,,,,,Huid,,,,,,,,,,Voeding,,,,,,,Medicatie,,,Darmfunctie
# inputs = tokenizer.encode_plus(
#     'Evt vanmiddag doen zat nauwelijks ontlasting in het zakje',
#     None,
#     add_special_tokens=True,
#     max_length=max_length,
# )
# input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
# pad_token = 0
# padding_length = max_length - len(input_ids)
# input_ids = input_ids + ([pad_token] * padding_length)
# input_ids_tensor = tf.convert_to_tensor(input_ids)
# input_ids_tensor = tf.expand_dims(input_ids_tensor,0) # 1 batch
#
# atten_config = BertConfig.from_pretrained('bert-base-multilingual-cased')
# atten_config.num_labels = 16
# atten_config.output_hidden_states=True
# atten_config.output_attentions=True
#
# # model = TFBertForSequenceMultiLabelClassification.from_pretrained('bert-base-multilingual-cased', config= atten_config) #bert-base-cased
#
# # the shape of attention is (batch_size, num_heads, sequence_length, sequence_length)
# atten_weights = model(input_ids_tensor)[1]
# prediction =  model(input_ids_tensor)[0]
# # prob_list_names
# thresholded_pred = tf.cast(prediction[0]>0.5, tf.int64).numpy()
# predict_problems = [prob_list_names[idx] for idx,val in enumerate(thresholded_pred) if val ==1]
# print("the predicted problems are:", predict_problems)
# print("the actual problems are:", ['Zicht','Huid','Voeding','Medicatie','Darmfunctie'])
#
# def plot_attention_weights(attention, layer_num):
#     fig = plt.figure(figsize=(8, 16))
#     attention = tf.squeeze(attention[layer_num], axis=0) #since there is only one batch
#     attention = attention[:,:input_ids.index(0),:input_ids.index(0)]
#     input_ids_no_pad = input_ids[:input_ids.index(0)]
#     for head in range(attention.shape[0]):
#         ax = fig.add_subplot(atten_config.num_attention_heads,1, head+1)
#
#         # plot the attention weights
#         ax.matshow(tf.expand_dims(tf.math.reduce_mean(attention[head],0),0), cmap='viridis')
#         # ax.matshow(attention[head][:-1, :], cmap='viridis')
#
#         fontdict = {'fontsize': 10}
#
#         ax.set_xticks(range(len(input_ids_no_pad)))
#         ax.set_yticks(range(1))
#
#         # ax.set_ylim(len(sentence)-1.5, -0.5)
#         ax.set_xticklabels(
#             [tokenizer._convert_id_to_token(input_id) for input_id in input_ids_no_pad],
#             fontdict=fontdict, rotation=90)
#
#         ax.set_yticklabels(['weights'],fontdict=fontdict)
#
#         ax.set_xlabel('Head {}'.format(head+1))
#
#     plt.tight_layout()
#     plt.show()
#     fig.savefig('./result/transformer_bert_layer'+str(layer_num)+'.png')
#
# print("plotting the attention weights of the first layer....")
# plot_attention_weights(atten_weights, atten_config.num_hidden_layers-1)
# print("plotting the attention weights of the last layer....")
# plot_attention_weights(atten_weights, 0)
