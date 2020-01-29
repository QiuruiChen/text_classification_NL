# xlm-mlm-17-1280
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

try:
    from fastprogress import master_bar, progress_bar
except ImportError:
    from fastprogress.fastprogress import master_bar, progress_bar
import math
from seqeval import metrics

BATCH_SIZE = 512
NUM_TRAIN_EPOCHS = 1
LOGGING_STEPS = 2
EVALUATE_DURING_TRAINING = True
SAVE_STEPS = 2 # save checkpoint at every 2 epoches
OUTPUT_DIR = './model_saves/transformer_XLM/'
N_DEVICE = 1
SEQ_LEN = 128
    
def dataset_get(batch_size):
    train_size = 19067
    val_size = 4764
    # Load dataset, tokenizer, model from pretrained model/vocabulary
    # see what is possible: https://huggingface.co/transformers/pretrained_models.html
    tokenizer = XLMTokenizer.from_pretrained('xlm-mlm-17-1280',add_special_tokens=True) #bert-base-multilingual-cased
    # train dataset
    df_train = pd.read_csv('transformer_bert_train.csv')
    prob_list_names = ['Persoonlijke_zorg', 'Medicatie', 'Huid', 'Circulatie', 'Voeding','Urinewegfunctie', 'Neuro_musculaire_skeletfunctie', 'Cognitie', 'Pijn',
    'Darmfunctie', 'Geestelijke_gezondheid', 'Ademhaling', 'Mantelzorg_zorg_voor_kind_huisgenoot', 'Fysieke_activiteit', 'Zicht','other']

    df_train['label'] = [np.asarray(val) for val in df_train.reset_index()[prob_list_names].values.tolist()]
    dataset = tf.data.Dataset.from_tensor_slices({'sentence':df_train['comment_text'].values,
                                                  'label':df_train[prob_list_names].values})
    # validatiojn dataset
    df_val = pd.read_csv('transformer_bert_val.csv')
    df_val['label'] = [np.asarray(val) for val in df_val.reset_index()[prob_list_names].values.tolist()]
    dataset_val = tf.data.Dataset.from_tensor_slices({'sentence':df_val['comment_text'].values,
                                                  'label':df_val[prob_list_names].values})
    # test dataset
    df_test = pd.read_csv('transformer_bert_test.csv')
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

    return valid_dat_batched,test_dat_batched,train_dat_batched,train_size,val_size,tokenizer


## =========
##  train
## ==========
def configuration_get(config_type):
    configuration = XLMConfig.from_pretrained('xlm-mlm-17-1280')
    configuration.early_stopping = True
    configuration.num_labels= 16
    configuration.n_langs = 1
    configuration.lang_id = 9
    if config_type == 'classification':
        configuration.summary_activation='sigma'
        configuration.summary_type = 'mean'
    return configuration

def model_build(model_type,configuration):
    '''
    :param model_type: classification: TFXLMForSequenceClassification model
                        raw: TFXLMModel
    :param configuration:
    :return:
    '''
    if model_type =="classification":
        model = TFXLMForSequenceClassification.from_pretrained('xlm-mlm-17-1280', config= configuration)
    elif model_type == "raw":

        summary_type = 'mean'
        input_layer = tf.keras.layers.Input(shape=(SEQ_LEN,), dtype = 'int64')
        model_base = TFXLMModel.from_pretrained('xlm-mlm-17-1280', config= configuration)(input_layer) # xlm-mlm-17-1280
        # x = model_base[0]
        if summary_type == 'last':
            x = model_base[0][:, -1]
        elif summary_type == 'first':
            x = model_base[0][:, 0]
        elif summary_type == 'mean':
            x = tf.reduce_mean(model_base[0], axis=1)

        x = tf.keras.layers.Dropout(0.1)(x)
        x = tf.keras.layers.Dense(16 ,
                                  kernel_initializer= tf.keras.initializers.TruncatedNormal(stddev=0.02),
                                  name='summary',
                                  activation = 'sigmoid')(x)

        model = tf.keras.models.Model(input_layer, x)
    return model

def train(configuration,NUM_TRAIN_EPOCHS,LOGGING_STEPS,SAVE_STEPS,strategy,N_DEVICE,
          train_dataset,val_dataset,model,OUTPUT_DIR,batch_size,train_size):
    EVALUATE_DURING_TRAINING = True
    num_train_steps = (
            math.ceil(train_size / batch_size)
            // configuration.accumulate_gradients #args["gradient_accumulation_steps"]
            * NUM_TRAIN_EPOCHS
        )

    writer = tf.summary.create_file_writer("/tmp/mylogs") #open tensorboard in: tensorboard --logdir /tmp/mylogs

    with strategy.scope():
        loss_fct = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE,from_logits=True)
        optimizer = create_optimizer(0.00005, num_train_steps, 30000)
        if configuration.fp16:
            optimizer = tf.keras.mixed_precision.experimental.LossScaleOptimizer(optimizer, "dynamic")

        loss_metric = tf.keras.metrics.Mean(name="loss", dtype=tf.float32)
        gradient_accumulator = GradientAccumulator()

    logging.info("***** Running training *****")
    logging.info("  Num examples = %d", train_size)
    logging.info("  Num Epochs = %d", NUM_TRAIN_EPOCHS)
    logging.info("  Instantaneous batch size per device = %d", batch_size)
    logging.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        batch_size * configuration.accumulate_gradients,
    )
    logging.info("  Gradient Accumulation steps = %d", configuration.accumulate_gradients)
    logging.info("  Total training steps = %d", num_train_steps)

    model.summary()

    @tf.function
    def apply_gradients():
        grads_and_vars = []

        for gradient, variable in zip(gradient_accumulator.gradients, model.trainable_variables):
            if gradient is not None:
                scaled_gradient = gradient / (1 * configuration.accumulate_gradients)
                grads_and_vars.append((scaled_gradient, variable))
            else:
                grads_and_vars.append((gradient, variable))

        optimizer.apply_gradients(grads_and_vars, configuration.clip_grad_norm)
        gradient_accumulator.reset()

    @tf.function
    def train_step(train_features, train_labels):
        def step_fn(train_features, train_labels):

            with tf.GradientTape() as tape:
                logits = model(train_features)[0]
                cross_entropy = loss_fct(train_labels, logits)
                loss = tf.reduce_sum(cross_entropy) * (1.0 / batch_size)
                grads = tape.gradient(loss, model.trainable_variables)

                gradient_accumulator(grads)

            return cross_entropy

        per_example_losses = strategy.experimental_run_v2(step_fn, args=(train_features, train_labels))
        mean_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_example_losses, axis=0)

        return mean_loss

    train_iterator = master_bar(range(NUM_TRAIN_EPOCHS))
    current_time = datetime.datetime.now()
    global_step = 0
    logging_loss = 0.0

    for epoch in train_iterator:
        epoch_iterator = progress_bar(train_dataset, total=num_train_steps, parent=train_iterator, display= N_DEVICE > 1)
        step = 1

        with strategy.scope():
            for train_features, train_labels in epoch_iterator:
                loss = train_step(train_features, train_labels)

                if step % configuration.accumulate_gradients == 0:
                    strategy.experimental_run_v2(apply_gradients)

                    loss_metric(loss)

                    global_step += 1

                    if LOGGING_STEPS > 0 and global_step % LOGGING_STEPS == 0:
                        # Log metrics
                        if (N_DEVICE == 1 and EVALUATE_DURING_TRAINING):  # Only evaluate when single GPU otherwise metrics may not average well
                            y_true, y_pred, eval_loss  = evaluate(strategy, model,N_DEVICE,batch_size,val_dataset)

                            report = metrics.classification_report(y_true, y_pred, digits=4)

                            logging.info("Eval at step " + str(global_step) + "\n" + report)
                            logging.info("eval_loss: " + str(eval_loss))

                            precision = metrics.precision_score(y_true, y_pred)
                            recall = metrics.recall_score(y_true, y_pred)
                            f1 = metrics.f1_score(y_true, y_pred)
                            acc = metrics.accuracy_score(y_true,y_pred)
                            with writer.as_default():
                                tf.summary.scalar("eval_loss", eval_loss, global_step)
                                tf.summary.scalar("precision", precision, global_step)
                                tf.summary.scalar("recall", recall, global_step)
                                tf.summary.scalar("f1", f1, global_step)
                                tf.summary.scalar("accuracy", acc, global_step)

                        lr = optimizer.learning_rate
                        learning_rate = lr(step)

                        with writer.as_default():
                            tf.summary.scalar("lr", learning_rate, global_step)
                            tf.summary.scalar("loss", (loss_metric.result() - logging_loss) / LOGGING_STEPS, global_step)

                        logging_loss = loss_metric.result()

                    with writer.as_default():
                        tf.summary.scalar("loss", loss_metric.result(), step=step)

                    if SAVE_STEPS > 0 and global_step % SAVE_STEPS == 0:
                        OUTPUT_DIR = os.path.join(OUTPUT_DIR, "checkpoint-{}".format(global_step))

                        if not os.path.exists(OUTPUT_DIR):
                            os.makedirs(OUTPUT_DIR)

                        model.save_pretrained(OUTPUT_DIR)
                        logging.info("Saving model checkpoint to %s", OUTPUT_DIR)

                train_iterator.child.comment = f"loss : {loss_metric.result()}"
                step += 1

        train_iterator.write(f"loss epoch {epoch + 1}: {loss_metric.result()}")
        loss_metric.reset_states()

        logging.info("  Training took time = {}".format(datetime.datetime.now() - current_time))

def evaluate(strategy, model,N_DEVICE,batch_size,eval_dataset,val_size):

    eval_batch_size = batch_size * N_DEVICE

    preds = None

    num_eval_steps = math.ceil(val_size / eval_batch_size)
    master = master_bar(range(1))
    eval_iterator = progress_bar(eval_dataset, total=num_eval_steps, parent=master, display=N_DEVICE > 1)
    loss_fct = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
    loss = 0.0

    logging.info("***** Running evaluation *****")
    logging.info("  Num examples = %d", val_size)
    logging.info("  Batch size = %d", eval_batch_size)

    for eval_features, eval_labels in eval_iterator:

        with strategy.scope():
            logits = model(eval_features)[0]
            cross_entropy = loss_fct(eval_labels,logits)
            loss += tf.reduce_sum(cross_entropy) * (1.0 / eval_batch_size)

        if preds is None:
            preds = logits.numpy()
            label_ids = eval_labels.numpy()
        else:
            preds = np.append(preds, logits.numpy(), axis=0)
            label_ids = np.append(label_ids, eval_labels.numpy(), axis=0)

    loss = loss / num_eval_steps

    preds = np.where(preds > 0.5, 1, 0)
    # convert elements in list into string to fit metrics toolbox
    preds = [[str(element) for element in elist] for elist in preds]
    label_ids = [[str(element) for element in elist] for elist in label_ids]

    return label_ids, preds, loss.numpy()


def train_eager_model(configuration,train_dat_batched,valid_dat_batched,tokenizer):
    configuration.fp16 = True
    # if tpu:
    #     resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=tpu)
    #     tf.config.experimental_connect_to_cluster(resolver)
    #     tf.tpu.experimental.initialize_tpu_system(resolver)
    #     strategy = tf.distribute.experimental.TPUStrategy(resolver)
    #     n_device = num_tpu_cores
    # elif len(gpus.split(',')) > 1:
    #     n_device = len([f"/gpu:{gpu}" for gpu in gpus.split(',')])
    #     strategy = tf.distribute.MirroredStrategy(devices=[f"/gpu:{gpu}" for gpu in gpus.split(',')])
    # elif no_cuda:
    #     n_device = 1
    #     strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
    # else:
    #     n_device = len(gpus.split(','))
    #     strategy = tf.distribute.OneDeviceStrategy(device="/gpu:" + gpus.split(',')[0])
    #
    # logging.warning("n_device: %s, distributed training: %s, 16-bits training: %s",
    #                n_device, bool(n_device > 1), configuration.fp16)
    
    if configuration.fp16:
        tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})

    # mac terminal: sysctl -n hw.ncpu
    strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
    with strategy.scope():
        model = TFXLMForSequenceClassification.from_pretrained('xlm-mlm-17-1280', config= configuration)
        # model.layers[-1].activation = tf.keras.activations.sigmoid

    train_dataset = strategy.experimental_distribute_dataset(train_dat_batched)
    val_dataset = strategy.experimental_distribute_dataset(valid_dat_batched)
    train(NUM_TRAIN_EPOCHS,LOGGING_STEPS,SAVE_STEPS,strategy,N_DEVICE,train_dataset,val_dataset,model,OUTPUT_DIR)

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    logging.info("Saving model to %s",OUTPUT_DIR)

    model.save_pretrained(OUTPUT_DIR+'/models/')
    tokenizer.save_pretrained(OUTPUT_DIR+'/tokenizers/')

## train on the high-level
def train_high_level(train_type):
    if train_type == "classification":
        
        class_config = configuration_get(config_type="classification")
        
        model = model_build(model_type="classification",configuration=class_config)

        loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        metric = tf.keras.metrics.BinaryAccuracy('accuracy')
        #   "optimizer": "adam_inverse_sqrt,lr=0.00005,warmup_updates=30000,beta1=0.9,beta2=0.999,weight_decay=0.01,eps=0.000001",
        optimizer = tfa.optimizers.AdamW(weight_decay = 0.01,learning_rate=0.00005, beta_1=0.9, beta_2=0.999, epsilon=0.000001, amsgrad=False, name='AdamW',)
        model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
        valid_dat_batched,test_dat_batched,train_dat_batched,train_size,val_size,tokenizer = dataset_get(batch_size=BATCH_SIZE)
        # 297 steps ==> 1 step ~ 2min; ~10hs
        history = model.fit(train_dat_batched, epochs=NUM_TRAIN_EPOCHS,
                            steps_per_epoch= int(train_size // BATCH_SIZE),
                            validation_data=valid_dat_batched,
                            validation_steps=int(val_size // BATCH_SIZE)) #115,7
        model.save_pretrained('./model_saves/transformer_XLM/classifier_model/')
        
    elif train_type == "raw":
        class_config = configuration_get(config_type="raw")
        model = model_build(model_type="raw",configuration=class_config)

        
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        metric = tf.keras.metrics.BinaryAccuracy('accuracy')
        #   "optimizer": "adam_inverse_sqrt,lr=0.00005,warmup_updates=30000,beta1=0.9,beta2=0.999,weight_decay=0.01,eps=0.000001",
        optimizer = tfa.optimizers.AdamW(weight_decay = 0.01,learning_rate=0.00005, beta_1=0.9, beta_2=0.999, epsilon=0.000001, amsgrad=False, name='AdamW',)
        model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
        valid_dat_batched,test_dat_batched,train_dat_batched,train_size,val_size,tokenizer = dataset_get(batch_size=BATCH_SIZE)
        history = model.fit(train_dat_batched, epochs=NUM_TRAIN_EPOCHS,
                            steps_per_epoch= int(train_size // BATCH_SIZE),
                            validation_data=valid_dat_batched,
                            validation_steps=int(val_size // BATCH_SIZE)) #115,7
        model.save_weights(OUTPUT_DIR+'base_model/weights/')
        model.layers[1].save_pretrained(OUTPUT_DIR+'base_model/pretrained/')

        # model.load_weights(OUTPUT_DIR+'base_model/weights/)


## trian on the low-level (gradient tape), slow since train only on 1CPU
# valid_dat_batched,test_dat_batched,train_dat_batched,train_size,val_size,tokenizer = dataset_get(batch_size=BATCH_SIZE)
# configuration = configuration_get(config_type="classification")
# train_eager_model(configuration,train_dat_batched=train_dat_batched,valid_dat_batched= valid_dat_batched,tokenizer=tokenizer)
## or train on high level
train_high_level(train_type="raw")

# # Load the TensorFlow model in PyTorch for inspection
#
# sample = next(iter(train_dataset.map(lambda inputs,label: (inputs['input_ids'], label)).batch(2)))
# pred = model(sample[0])
# print("true label is:", sample[1])
# logits = pred[0]
# print("prediction is:", logits)

## ===========
##  evaluate
## ===========
# model = TFBertForSequenceMultiLabelClassification.from_pretrained('./model_saves/transformer_bert/', config= configuration) #bert-base-cased
# # Prepare training: Compile tf.keras model with optimizer, loss and learning rate schedule
# # optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0)
# # loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
# # metric = tf.keras.metrics.BinaryAccuracy('accuracy')
# # model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
# model.summary()
#
# # inputs, targets = next(iter(val_dataset.batch(val_size)))
# inputs, targets = next(iter(test_dataset.batch(2000)))
# predictions = model(inputs)
# # Load the TensorFlow model in PyTorch for inspection
# prediction = tf.cast(predictions[0]>0.5, tf.int64)
#
# from sklearn.metrics import hamming_loss,accuracy_score
#
# print("the exact accuracy is: ",accuracy_score(targets.numpy(), prediction.numpy()))
# print("the hamming loss is: ", hamming_loss(targets.numpy(), prediction.numpy()))
# for i in range(0,16):
#     print("the accuracy for the problem",prob_list_names[i])
#     print("acc is:", str((targets.numpy()[:,i]== prediction.numpy()[:,i]).sum() / targets.numpy().shape[0]))


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
