from __future__ import print_function

import os
import numpy as np

import unicodedata
import pandas as pd
import re
import tensorflow as tf
print(tf.__version__)
import os.path

class Preproc:
    def __init__(self,base_dir):
        self.BASE_DIR = base_dir
    # Converts the unicode file to ascii
    def _unicode_to_ascii(self,s):
      return ''.join(c for c in unicodedata.normalize('NFD', s)
          if unicodedata.category(c) != 'Mn')

    def preprocess_sentence(self,w):
      w = self._unicode_to_ascii(w.lower().strip())
      # creating a space between a word and the punctuation following it
      # eg: "he is a boy." => "he is a boy ."
      # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
      w = re.sub(r"([?.!,¿])", r" \1 ", w)
      w = re.sub(r'[" "]+', " ", w)
      return w


    def lematize_dutch(self,text_list,is_frog=False):
        if is_frog == True:
            from frog import Frog, FrogOptions
            frog = Frog(FrogOptions(parser=False,ner=False,morph=False,chunking=False,mwu=False))
            w = frog.process(';'.join(text_list)) #process
            lemma_w = [val['lemma'] for val in w]
            lemma_w = ' '.join(lemma_w)
            text_list = lemma_w.split(";")
        elif is_frog == False:
            voc = [line.rstrip('\n') for line in open(self.BASE_DIR+'voc.txt')]
            stems = [line.rstrip('\n') for line in open(self.BASE_DIR+'stem.txt')]
            adict = dict(zip(voc,stems))
            text_list = [' '.join([adict[text] if text in adict else text for text in text_string.split(' ')]) for text_string in text_list]
        return text_list

    def remove_stop(self,text_list):
        stopVoc = [combines.split('|')[0].replace(" ", "") for combines in [line.rstrip('\n') for line in open(self.BASE_DIR+'stopWords.txt')]]
        text_list = [' '.join([text for text in text_string.split(' ') if text not in stopVoc]) for text_string in text_list]
        return text_list

    def max_length(self,tensor):
        return max(len(t) for t in tensor)

    def _tokenize(self,lang,pad):
        lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
        lang_tokenizer.fit_on_texts(lang)
        tensor = lang_tokenizer.texts_to_sequences(lang)
        if pad == True:
            tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,padding='post')
        return tensor,lang_tokenizer

    def load_dataset(self,lang,pad=True):
        # creating cleaned input, output pairs
        input_tensor,inp_lang_tokenizer = self._tokenize(lang,pad)
        return input_tensor,inp_lang_tokenizer

    def chunkesize_data(self,chunk_num=10):
        raw_data = pd.read_pickle(self.BASE_DIR+'raw_data.pkl')
        unit_len = int(np.floor(raw_data.shape[0]/chunk_num))
        print("chunk size is: ",unit_len)
        for i in range(chunk_num):
            if i == chunk_num-1:
                temp = raw_data[i*unit_len:]
            else:
                temp = raw_data[i*unit_len:(i+1)*unit_len-1]
            temp.loc[:,'comment_text'] = temp['comment_text'].apply(lambda x: self.preprocess_sentence(x))
            text_list = self.lematize_dutch(temp['comment_text'].tolist())
            text_list = self.remove_stop(text_list)
            print("procesing chunck"+str(i))
            temp.loc[:,'comment_text'] = text_list
            temp.to_pickle(self.BASE_DIR+'raw_data'+str(i)+'.pkl')

        pd_list = []
        for i in range(chunk_num):
            file_path = 'raw_data'+str(i)+'.pkl'
            if os.path.exists(file_path):
                pd_list.append(pd.read_pickle(file_path))
                os.remove(file_path)
            else:
                raise Exception('file does NOT exist!')

        raw_data = pd.concat(pd_list)
        raw_data.to_pickle(self.BASE_DIR+'raw_data_preprocessed.pkl')
        return raw_data

# raw_data = pd.read_pickle('data/raw_data.pkl')
# raw_data.drop(columns = {'id', 'PatientID'})
# raw_data['comment_text'] = raw_data['comment_text'].apply(lambda x: preprocess_sentence(x))
# text_list = lematize_dutch(raw_data['comment_text'].tolist())
# text_list = remove_stop(text_list)

def create_ngram_set(input_list, ngram_value=2):
    """
    Extract a set of n-grams from a list of integers.
    # >>> create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=2)
    # {(4, 9), (4, 1), (1, 4), (9, 4)}
    # >>> create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=3)
    [(1, 4, 9), (4, 9, 4), (9, 4, 1), (4, 1, 4)]
    """
    return set(zip(*[input_list[i:] for i in range(ngram_value)]))


def add_ngram(sequences, token_indice, ngram_range=2):
    """
    Augment the input list of list (sequences) by appending n-grams values.
    Example: adding bi-gram
    # >>> sequences = [[1, 3, 4, 5], [1, 3, 7, 9, 2]]
    # >>> token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017}
    # >>> add_ngram(sequences, token_indice, ngram_range=2)
    # [[1, 3, 4, 5, 1337, 2017], [1, 3, 7, 9, 2, 1337, 42]]
    # Example: adding tri-gram
    # >>> sequences = [[1, 3, 4, 5], [1, 3, 7, 9, 2]]
    # >>> token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017, (7, 9, 2): 2018}
    # >>> add_ngram(sequences, token_indice, ngram_range=3)
    [[1, 3, 4, 5, 1337, 2017], [1, 3, 7, 9, 2, 1337, 42, 2018]]
    """
    new_sequences = []
    for input_list in sequences:
        new_list = input_list[:]
        for ngram_value in range(2, ngram_range + 1):
            for i in range(len(new_list) - ngram_value + 1):
                ngram = tuple(new_list[i:i + ngram_value])
                if ngram in token_indice:
                    new_list = np.append(new_list,token_indice[ngram])
        new_sequences.append(new_list)

    return new_sequences
