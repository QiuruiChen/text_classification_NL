# Created by CHEN at 8/22/2019
# Email: rachelchen0831@gmail.com
'''
Pretrained word embedding is from github: https://github.com/coosto/dutch-word-embeddings

args['train_type']: use different architecture for training models.
'simple':  embedding + globalAverage1D
'conv1d': embedding+ conv1d
'conv2d': embedding + conv2d
'bidireLSTM': bidirectional LSTM

args['fasttext'] = True: run FastText (n-gram based on letters)

Train on the first 3000 dataset:
result:

'''
from __future__ import print_function

import numpy as np

from sklearn.model_selection import train_test_split
import pandas as pd
import tensorflow as tf
print(tf.__version__)
import os.path
from absl import logging
from absl import flags
from absl import app

from sklearn.metrics import accuracy_score
from preproc import *
import gensim
import io
import re

flags.DEFINE_string(
    "train_type", 'conv1d',
    "chose from 'conv1d','conv2d','bidireLSTM','simple'")

flags.DEFINE_string(
    "class_name", 'Persoonlijke_zorg',
    "class name that needed to be classified")

flags.DEFINE_string(
    "data_dir", "../data/",
    "data location")

flags.DEFINE_integer(
    "max_seq_len", 512,
    "The maximum total input sentence length after tokenization. "
    "Sequences longer than this will be truncated, sequences shorter "
    "will be padded.")

flags.DEFINE_string(
    "tpu", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Total number of TPU cores to use.")

flags.DEFINE_integer(
    "num_train_epochs", 300,
    "training epochs")

flags.DEFINE_integer(
    "batch_size", 64,
    "Batch size per GPU/CPU/TPU.")

flags.DEFINE_float(
    "learning_rate", 5e-5,
    "The initial learning rate for Adam.")

flags.DEFINE_float(
    "valid_split", 0.2,
    "dataset split ratio.")

flags.DEFINE_float(
    "test_split", 0.2,
    "dataset split ratio.")

flags.DEFINE_integer(
    "max_len", 300,
    "maximum text length of the text")

flags.DEFINE_integer(
    "emb_dim", 300,
    "embedding dimensions")

flags.DEFINE_boolean(
    "fasttext", False,
    "using fast text training?")

flags.DEFINE_integer(
    "ngram_range", 5,
    "n-gram in fasttext")

flags.DEFINE_boolean(
    "no_cuda", True,
    "Avoid using CUDA when available")

flags.DEFINE_integer(
    "seed", 42,
    "random seed for initialization")

flags.DEFINE_boolean(
    "fp16", False,
    "Whether to use 16-bit (mixed) precision instead of 32-bit")

flags.DEFINE_string(
    "gpus", "0",
    "Comma separated list of gpus devices. If only one, switch to single "
    "gpu strategy, if None takes all the gpus available.")

def main(_):
    logging.set_verbosity(logging.INFO)
    args = flags.FLAGS.flag_values_dict()
    SAVED_FILE_NAME = args['train_type'] +'fasttext_prob' if args['fasttext'] else args['train_type']+'_prob'

    # first, build index mapping words in the embeddings set
    # to their embedding vector

    print('Indexing word vectors.')
    wd_file = 'glove.42B.300d.txt'#roularta-320.txt',sonar-320.txt,wikipedia-320.txt,combined-320.txt
    embeddings_index = {}
    with open(os.path.join(args['data_dir'], wd_file),encoding='latin-1') as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, 'f', sep=' ')
            embeddings_index[word] = coefs

    # inputFile = args['data_dir'] +'model.bin'
    # model = gensim.models.KeyedVectors.load_word2vec_format(inputFile, binary=True)
    # word_vectors = model.wv
    # print("Number of word vectors: {}".format(len(word_vectors.vocab)))
    # MAX_NB_WORDS = len(word_vectors.vocab)

    # second, prepare text samples and their labels
    print('Processing text dataset')
    preproc = Preproc(base_dir = args['data_dir'])

    file_path = args['data_dir']+'raw_data_preprocessed.pkl'
    if os.path.exists(file_path):
        raw_data = pd.read_pickle(args['data_dir']+'raw_data_preprocessed.pkl')[:100000]
    else:
        raw_data = preproc.chunkesize_data()

    raw_data['comment_text'] = raw_data['comment_text'].apply(lambda x:" ".join([x for x in re.split("[^a-zA-Z]*",x) if len(x) > 1]))
    raw_data['text_length'] = raw_data['comment_text'].apply(lambda x: len(x.split(" "))-1)
    raw_data = raw_data[raw_data['text_length'] > 4]
    raw_data = raw_data[raw_data['text_length'] < args['max_len']]
    text_list = raw_data['comment_text'].tolist()
    print("preview 5 samples: \n",raw_data['comment_text'].head(5))
    # Try experimenting with the size of that dataset
    input_tensor,inp_lang = preproc.load_dataset(tuple(text_list),pad=False)
    # Calculate max_length of the target tensors
    max_length_inp = preproc.max_length(input_tensor)

    # remove zero-variance columns
    raw_data = raw_data.loc[:, (raw_data != raw_data.iloc[0]).any()]
    # raw_data['comment_text'].to_csv('../data/comment_text.csv',index=False)
    # classes_num = raw_data.drop(columns=['comment_text','comment_text','id', 'PatientID','text_length']).shape[1]
    # output_tensor = raw_data.drop(columns=['comment_text','comment_text','id', 'PatientID','text_length']).values.tolist()
    # classes_num  = raw_data[['Medicatie']].shape[1]
    # output_tensor = raw_data[['Medicatie']].values.tolist()

    classes_num = raw_data[['Voeding']].shape[1]
    output_tensor = raw_data[['Voeding']].values.tolist()

    # Creating training and validation sets using an 80-20 split
    input_tensor_train,input_tensor_test,target_tensor_train,target_tensor_test = train_test_split(
        input_tensor,
        output_tensor,
        test_size=args['test_split'],
        random_state = args['seed'],
        shuffle = True)
    input_tensor_train,input_tensor_val,target_tensor_train,target_tensor_val = train_test_split(
        input_tensor_train,
        target_tensor_train,
        test_size=args['valid_split'],
        random_state = args['seed'],
        shuffle = True)

    # Show length
    # print(len(input_tensor_train), len(input_tensor_val))
    def convert(lang, tensor):
        index_word = {v: k for k, v in lang.word_index.items()}
        for t in tensor:
            if t!=0:
                print ("%d ----> %s" % (t, index_word[t]))
    print ("Input Language; index to word mapping")
    convert(inp_lang, input_tensor[0])

    # build vocabulary
    int_word_dict = {}
    index_word = {v: k for k, v in inp_lang.word_index.items()}
    word_index = {k: v for k, v in inp_lang.word_index.items()}
    print("how many word in vocabulary?",len(index_word))
    max_val = max(index_word, key=int)

    ###
    print('Loading data...')
    x_train = input_tensor_train
    y_train = np.asarray(target_tensor_train)
    x_val = input_tensor_val
    y_val =  np.asarray(target_tensor_val)
    x_test = input_tensor_test
    y_test =  np.asarray(target_tensor_test)


    '''
    Below demonstrates the use of fasttext for text classification
    Based on Joulin et al's paper:
    Bag of Tricks for Efficient Text Classification
    (https://arxiv.org/abs/1607.01759)
    Results on IMDB datasets with uni and bi-gram embeddings:
    Embedding|Accuracy, 5 epochs|Speed (s/epoch)|Hardware
    :--------|-----------------:|----:|:-------
    Uni-gram |            0.8813|    8|i7 CPU
    Bi-gram  |            0.9056|    2|GTx 980M GPU
    
    
    Each word is represented as a bag of character n-grams in addition to the word itself
    The model is considered to be a bag of words model because aside of the sliding window of n-gram selection,
    there is no internal structure of a word that is taken into account for featurization,
    i.e as long as the characters fall under the window, the order of the character n-grams does not matter.
    
    '''
    if args['fasttext'] == True:
        # Set parameters:
        # args['ngram_range'] = 2 will add bi-grams features
        MAX_FEATURES = 0 # do not change it!
        print(len(x_train), 'train sequences')
        print(len(x_val), 'test sequences')
        print('Average train sequence length: {}'.format(np.mean(list(map(len, x_train)), dtype=int)))
        print('Average test sequence length: {}'.format(np.mean(list(map(len, x_val)), dtype=int)))

        if args['ngram_range'] > 1:
            print('Adding {}-gram features'.format(args['ngram_range']))
            # Create set of unique n-gram from the training set.
            ngram_set = set()
            for input_list in x_train:
                for i in range(2, args['ngram_range'] + 1):
                    set_of_ngram = create_ngram_set(input_list, ngram_value=i)
                    ngram_set.update(set_of_ngram)

            # Dictionary mapping n-gram token to a unique integer.
            # Integer values are greater than max_features in order to avoid collision with existing features.
            start_index = MAX_FEATURES + 1
            token_indice = {v: k + start_index for k, v in enumerate(ngram_set)}
            indice_token = {token_indice[k]: k for k in token_indice}
            # max_features is the highest integer that could be found in the dataset.
            max_features = np.max(list(indice_token.keys())) + 1
            # Augmenting x_train and x_val with n-grams features
            x_train = add_ngram(x_train, token_indice, args['ngram_range'])
            x_val = add_ngram(x_val, token_indice, args['ngram_range'])
            print('Average train sequence length: {}'.format(np.mean(list(map(len, x_train)), dtype=int)))
            print('Average test sequence length: {}'.format(np.mean(list(map(len, x_val)), dtype=int)))

    print('Pad sequences (samples x time)')
    x_train =  tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=args['max_seq_len'])
    x_val =  tf.keras.preprocessing.sequence.pad_sequences(x_val, maxlen=args['max_seq_len'])
    x_test =  tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=args['max_seq_len'])
    print('x_train shape:', x_train.shape)
    print('x_val shape:', x_val.shape)
    print('x_test shape:',x_test.shape)

    print('Preparing embedding matrix.')
    # prepare embedding matrix
    MAX_NUM_WORDS = len(index_word) + 1
    embedding_matrix = np.zeros((MAX_NUM_WORDS, args['emb_dim']))
    for word, i in word_index.items():
        if i >= MAX_NUM_WORDS:
            continue
        if word in embeddings_index.keys(): #word_vectors.vocab:
            embedding_vector = embeddings_index[word] #word_vectors[word]
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        else:
             # words not found in embedding index will be all-zeros.
             embedding_matrix[i] = np.zeros(args['emb_dim'])

    if args['fp16']:
            tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})
    if args['tpu']:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=args['tpu'])
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        strategy = tf.distribute.experimental.TPUStrategy(resolver)
        args['n_device'] = args['num_tpu_cores']
    elif len(args['gpus'].split(',')) > 1:
        args['n_device'] = len([f"/gpu:{gpu}" for gpu in args['gpus'].split(',')])
        strategy = tf.distribute.MirroredStrategy(devices=[f"/gpu:{gpu}" for gpu in args['gpus'].split(',')])
    elif args['no_cuda']:
        args['n_device'] = 1
        strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
    else:
        args['n_device'] = len(args['gpus'].split(','))
        strategy = tf.distribute.OneDeviceStrategy(device="/gpu:" + args['gpus'].split(',')[0])

    logging.warning("n_device: %s, distributed training: %s, 16-bits training: %s",
                   args['n_device'], bool(args['n_device'] > 1), args['fp16'])

    logging.info("***** Running training *****")
    logging.info("  Num examples = %d", int(x_train.shape[0]))
    logging.info("  Num Epochs = %d", args['num_train_epochs'])
    logging.info("  Instantaneous batch size per device = %d", args['batch_size'])

    # with strategy.scope():
    if args['fasttext'] == True:
        print("fastText cannot trained based on pretrained word embeddings!")
        embedding_layer = tf.keras.layers.Embedding(max_features,
                                                    args['emb_dim'],
                                                    input_length=args['max_seq_len'])
    else:
        print("load pretrained word embeddings!")
        # load pre-trained word embeddings into an Embedding layer
        # note that we set trainable = False so as to keep the embeddings fixed
        embedding_layer = tf.keras.layers.Embedding(MAX_NUM_WORDS,
                                                    args['emb_dim'],
                                                    embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
                                                    input_length=args['max_seq_len']) #trainable=False

    print('Training model.')
    # train a 1D convnet with global maxpooling
    sequence_input = tf.keras.layers.Input(shape=(args['max_seq_len'],), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)

    if args['train_type'] == 'conv1d': # chose from 'conv1d','conv2d','bidireLSTM'
        x = tf.keras.layers.Conv1D(128, 5, activation='relu')(embedded_sequences)
        x = tf.keras.layers.MaxPooling1D(5)(x)
        x = tf.keras.layers.Conv1D(128, 5, activation='relu')(x)
        x = tf.keras.layers.MaxPooling1D(5)(x)
        x = tf.keras.layers.Conv1D(128, 5, activation='relu')(x)
        x = tf.keras.layers.GlobalMaxPooling1D()(x)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        preds = tf.keras.layers.Dense(units=classes_num, activation='sigmoid')(x)

    elif args['train_type'] == 'conv2d':
        ## or with conv2D
        FILTER_NUMS = 64
        filter_sizes = [3,5,7]
        print("embedding.shap",embedded_sequences.shape)
        reshape = tf.expand_dims(embedded_sequences,-1)
        print("reshape.shape",reshape.shape)

        conv_0 = tf.keras.layers.Conv2D(FILTER_NUMS,
                                        kernel_size=(filter_sizes[0], args['emb_dim']),
                                        padding='valid',
                                        kernel_initializer='normal',
                                        activation='relu')(reshape)
        conv_1 = tf.keras.layers.Conv2D(FILTER_NUMS,
                                        kernel_size=(filter_sizes[1], args['emb_dim']),
                                        padding='valid',
                                        kernel_initializer='normal',
                                        activation='relu')(reshape)
        conv_2 = tf.keras.layers.Conv2D(FILTER_NUMS,
                                        kernel_size=(filter_sizes[2], args['emb_dim']),
                                        padding='valid',
                                        kernel_initializer='normal',
                                        activation='relu')(reshape)

        maxpool_0 = tf.keras.layers.MaxPool2D(pool_size=(args['max_seq_len'] - filter_sizes[0] + 1, 1), strides=(1,1), padding='valid')(conv_0)
        maxpool_1 = tf.keras.layers.MaxPool2D(pool_size=(args['max_seq_len'] - filter_sizes[1] + 1, 1), strides=(1,1), padding='valid')(conv_1)
        maxpool_2 = tf.keras.layers.MaxPool2D(pool_size=(args['max_seq_len'] - filter_sizes[2] + 1, 1), strides=(1,1), padding='valid')(conv_2)

        concatenated_tensor = tf.keras.layers.Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])
        flatten = tf.keras.layers.Flatten()(concatenated_tensor)
        print(flatten.shape)
        dropout = tf.keras.layers.Dropout(0.8)(flatten)
        preds = tf.keras.layers.Dense(units=classes_num, activation='sigmoid')(dropout)

    elif args['train_type'] == 'bidireLSTM': # chose from 'conv1d','conv2d','bidireLSTM'
        # or with bidirectional LSTM
        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,return_sequences = True))(embedded_sequences)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64))(x)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        preds = tf.keras.layers.Dense(classes_num, activation='sigmoid')(x)

    elif args['train_type'] == 'simple':
        print('Build model...')
        # we add a GlobalAveragePooling1D, which will average the embeddings of all words in the document
        x = tf.keras.layers.GlobalAveragePooling1D()(embedded_sequences)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        # We project onto a single unit output layer, and squash it with a sigmoid:
        preds = tf.keras.layers.Dense(classes_num, activation='sigmoid')(x)
    else:
        raise Exception("Please chose traning type from chose from 'conv1d','conv2d','bidireLSTM','simple")


    model = tf.keras.models.Model(sequence_input, preds)
    model.summary()
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', # tf.keras.optimizers.Adam(learning_rate=args['learning_rate']),
                  metrics=['accuracy']) #
    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir='log_dir/'+SAVED_FILE_NAME+'/'),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=False,verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0),
        tf.keras.callbacks.ModelCheckpoint('model_saves/'+SAVED_FILE_NAME+'_best.h5',  monitor='val_accuracy', save_best_only=True) #period=2,
    ]
    model.fit(x_train, y_train,
              batch_size=args['batch_size'],
              epochs=args['num_train_epochs'],
              callbacks=callbacks,
              validation_data=(x_val, y_val),
              shuffle =True,verbose =1)

    model.save('model/saves/'+SAVED_FILE_NAME)
    # model = tf.keras.models.load_model('model/saves/'+SAVED_FILE_NAME)

    prediction = model.predict(x_test)
    prediction[prediction>0.5] = 1
    prediction[prediction<0.5] = 0
    score = accuracy_score(y_test, prediction)
    logging.info('the score is {}'.format(score))
    # ================================
    # Saving embeddings
    # ================================
    # save the embeddings
    e = model.layers[1]
    weights = e.get_weights()[0]
    logging.info('the weight shape is {}'.format(weights.shape)) # shape: (vocab_size, embedding_dim)

    # encoder = info.features['text'].encoder
    out_v = io.open('vecs.tsv', 'w', encoding='utf-8')
    out_m = io.open('meta.tsv', 'w', encoding='utf-8')

    # for num, word in enumerate(encoder.subwords):
    for num, word in int_word_dict.items():
        vec = weights[num] # skip 0, it's padding.
        out_m.write(word + "\n")
        out_v.write('\t'.join([str(x) for x in vec]) + "\n")
    out_v.close()
    out_m.close()

    # ================================
    # model evluate
    # ================================
    test_loss, test_acc = model.evaluate(x_val, y_val)
    logging.info('Test Loss: {}'.format(test_loss))
    logging.info('Test Accuracy: {}'.format(test_acc))

    # ================================
    # model predict (for bidirectional lstm)
    # ================================
    # def pad_to_size(vec, size):
    #   zeros = [0] * (size - len(vec))
    #   vec.extend(zeros)
    #   return vec
    #
    # def sample_predict(sentence, pad):
    #   sentence = preproc.preprocess_sentence(sentence)
    #   sentence = preproc.lematize_dutch([sentence])[0]
    #   sentence = preproc.remove_stop([sentence])[0]
    #   encoded_sample_pred_text = [word_index[word] for word in sentence.split()]
    #
    #   if pad:
    #     encoded_sample_pred_text = pad_to_size(encoded_sample_pred_text, 64)
    #   encoded_sample_pred_text = tf.cast(encoded_sample_pred_text, tf.float32)
    #   predictions = model.predict(tf.expand_dims(encoded_sample_pred_text, 0))
    #
    #   return (predictions)
    #
    # # predict on a sample text with padding
    # sample_pred_text = ('mar voelt wel goed. bloeddruk gemet dez maand gebeurd ste woensdag maand hog kant. dhr moeit plass prostaatklachten. ker geholp hieran. hiervor keertje nar ha gan.')
    # predictions = tf.round(sample_predict(sample_pred_text, pad=True))
    # print ("prediction is: ",predictions)

if __name__ == "__main__":
    # flags.mark_flag_as_required("data_dir")
    # flags.mark_flag_as_required("output_dir")
    # flags.mark_flag_as_required("model_name_or_path")
    # flags.mark_flag_as_required("model_type")
    app.run(main)
