# Created by CHEN at 8/22/2019
# Email: rachelchen0831@gmail.com
'''
Inspect the data
the problem and its sorted appeared amount:
{'Persoonlijke_zorg': 1537953, 'Medicatie': 1358607, 'Huid': 891845, 'Circulatie': 882950, 'Voeding': 482491,
'Urinewegfunctie': 450005, 'Neuro_musculaire_skeletfunctie': 415844, 'Cognitie': 404994, 'Pijn': 364136,
'Darmfunctie': 321148, 'Geestelijke_gezondheid': 233844, 'Ademhaling': 202626, 'Mantelzorg_zorg_voor_kind_huisgenoot': 181914,
'Fysieke_activiteit': 171765, 'Zicht': 123522,

'Sociaal_contact': 95316, 'Gehoor': 86467, 'Spijsvertering_vochthuishouding': 86387,
'Communicatie_met_maatschappelijke_voorzieningen': 70110, 'Besmettelijke_infectueuze_conditie': 66571,
'Slaap_en_rust_patronen': 63848, 'Woning': 62200, 'Mondgezondheid': 50958, 'Interpersoonlijke_relaties': 50027,
'Spraak_en_taal': 48271, 'Rouw': 43038, 'Gezondheidszorg_supervisie': 26175, 'Buurt_werkplek_veiligheid': 19647,
'Omgevings_hygiene': 16975, 'Rolverandering': 15381, 'Bewustzijn': 13418, 'Gebruik_van_verslavende_middelen': 13416,

'Geslachtsorganen': 8730, 'Verwaarlozing': 7830, 'Inkomen_financien': 3975, 'Spiritualiteit': 1956,
'Sexualiteit': 1838, 'Mishandeling_misbruik': 1241, 'Groei_en_ontwikkeling': 522, 'Gezinsplanning': 133, 'Postnataal': 100, 'Zwangerschap': 18}

 the co-occurent problems with the problem Persoonlijke_zorgis: {'Persoonlijke_zorg': 1537953, 'Medicatie': 713282, 'Huid': 531480, 'Circulatie': 478618, 'Urinewegfunctie': 316787, 'Neuro_musculaire_skeletfunctie': 280696, 'Voeding': 275587, 'Pijn': 244784, 'Cognitie': 232983, 'Darmfunctie': 218123, 'Ademhaling': 141272, 'Geestelijke_gezondheid': 135301, 'Mantelzorg_zorg_voor_kind_huisgenoot': 127858, 'Fysieke_activiteit': 126124, 'Zicht': 65010, 'Sociaal_contact': 58755, 'Gehoor': 57274, 'Spijsvertering_vochthuishouding': 49616, 'Woning': 42434, 'Mondgezondheid': 41565, 'Slaap_en_rust_patronen': 40089, 'Communicatie_met_maatschappelijke_voorzieningen': 36055, 'Spraak_en_taal': 35434, 'Besmettelijke_infectueuze_conditie': 32938, 'Interpersoonlijke_relaties': 30614, 'Rouw': 24340, 'Gezondheidszorg_supervisie': 12880, 'Buurt_werkplek_veiligheid': 11862, 'Omgevings_hygiene': 10579, 'Rolverandering': 10442, 'Bewustzijn': 9103, 'Gebruik_van_verslavende_middelen': 7473, 'Geslachtsorganen': 6580, 'Verwaarlozing': 4429, 'Inkomen_financien': 2666, 'Spiritualiteit': 1491, 'Sexualiteit': 1417, 'Mishandeling_misbruik': 896, 'Groei_en_ontwikkeling': 195, 'Gezinsplanning': 131, 'Postnataal': 86, 'Zwangerschap': 7}
 the co-occurent problems with the problem Medicatieis: {'Medicatie': 1358607, 'Persoonlijke_zorg': 713282, 'Circulatie': 379991, 'Huid': 357902, 'Voeding': 341710, 'Cognitie': 275367, 'Urinewegfunctie': 224658, 'Neuro_musculaire_skeletfunctie': 178076, 'Pijn': 174227, 'Darmfunctie': 165159, 'Geestelijke_gezondheid': 141371, 'Ademhaling': 104566, 'Fysieke_activiteit': 83580, 'Mantelzorg_zorg_voor_kind_huisgenoot': 82124, 'Zicht': 66089, 'Sociaal_contact': 61544, 'Gehoor': 54410, 'Spijsvertering_vochthuishouding': 50218, 'Woning': 40549, 'Slaap_en_rust_patronen': 38073, 'Communicatie_met_maatschappelijke_voorzieningen': 37741, 'Mondgezondheid': 32670, 'Interpersoonlijke_relaties': 29543, 'Besmettelijke_infectueuze_conditie': 27168, 'Rouw': 24605, 'Spraak_en_taal': 21709, 'Gezondheidszorg_supervisie': 14541, 'Buurt_werkplek_veiligheid': 13393, 'Omgevings_hygiene': 11255, 'Gebruik_van_verslavende_middelen': 9784, 'Bewustzijn': 8333, 'Rolverandering': 7808, 'Verwaarlozing': 5555, 'Geslachtsorganen': 3399, 'Inkomen_financien': 2675, 'Sexualiteit': 1513, 'Spiritualiteit': 1205, 'Mishandeling_misbruik': 908, 'Groei_en_ontwikkeling': 420, 'Gezinsplanning': 67, 'Postnataal': 40, 'Zwangerschap': 0}
 the co-occurent problems with the problem Huidis: {'Huid': 891845, 'Persoonlijke_zorg': 531480, 'Medicatie': 357902, 'Circulatie': 354465, 'Urinewegfunctie': 176562, 'Neuro_musculaire_skeletfunctie': 162444, 'Pijn': 151495, 'Darmfunctie': 139060, 'Voeding': 127623, 'Cognitie': 99252, 'Ademhaling': 82140, 'Geestelijke_gezondheid': 73213, 'Fysieke_activiteit': 71531, 'Mantelzorg_zorg_voor_kind_huisgenoot': 64390, 'Zicht': 40322, 'Besmettelijke_infectueuze_conditie': 35874, 'Gehoor': 34285, 'Spijsvertering_vochthuishouding': 32287, 'Sociaal_contact': 25763, 'Mondgezondheid': 25235, 'Woning': 22381, 'Slaap_en_rust_patronen': 21641, 'Communicatie_met_maatschappelijke_voorzieningen': 20348, 'Spraak_en_taal': 18668, 'Interpersoonlijke_relaties': 14230, 'Rouw': 13995, 'Gezondheidszorg_supervisie': 9359, 'Buurt_werkplek_veiligheid': 7038, 'Omgevings_hygiene': 6778, 'Rolverandering': 5267, 'Bewustzijn': 5052, 'Geslachtsorganen': 4659, 'Gebruik_van_verslavende_middelen': 3583, 'Verwaarlozing': 2078, 'Inkomen_financien': 1229, 'Spiritualiteit': 864, 'Sexualiteit': 577, 'Mishandeling_misbruik': 132, 'Groei_en_ontwikkeling': 126, 'Gezinsplanning': 115, 'Postnataal': 42, 'Zwangerschap': 0}
 the co-occurent problems with the problem Circulatieis: {'Circulatie': 882950, 'Persoonlijke_zorg': 478618, 'Medicatie': 379991, 'Huid': 354465, 'Urinewegfunctie': 143828, 'Neuro_musculaire_skeletfunctie': 143519, 'Pijn': 118581, 'Voeding': 107647, 'Cognitie': 101996, 'Darmfunctie': 92031, 'Ademhaling': 83405, 'Fysieke_activiteit': 64008, 'Geestelijke_gezondheid': 61748, 'Mantelzorg_zorg_voor_kind_huisgenoot': 52661, 'Zicht': 42993, 'Gehoor': 33544, 'Sociaal_contact': 26785, 'Spijsvertering_vochthuishouding': 22266, 'Woning': 21801, 'Slaap_en_rust_patronen': 20599, 'Communicatie_met_maatschappelijke_voorzieningen': 20177, 'Besmettelijke_infectueuze_conditie': 19740, 'Rouw': 13783, 'Interpersoonlijke_relaties': 13765, 'Spraak_en_taal': 13061, 'Mondgezondheid': 12480, 'Gezondheidszorg_supervisie': 8548, 'Buurt_werkplek_veiligheid': 6837, 'Rolverandering': 5180, 'Omgevings_hygiene': 3938, 'Gebruik_van_verslavende_middelen': 3399, 'Bewustzijn': 3154, 'Geslachtsorganen': 2851, 'Verwaarlozing': 1458, 'Inkomen_financien': 891, 'Spiritualiteit': 711, 'Sexualiteit': 569, 'Mishandeling_misbruik': 333, 'Groei_en_ontwikkeling': 79, 'Postnataal': 45, 'Gezinsplanning': 13, 'Zwangerschap': 0}
 the co-occurent problems with the problem Voedingis: {'Voeding': 482491, 'Medicatie': 341710, 'Persoonlijke_zorg': 275587, 'Cognitie': 129322, 'Huid': 127623, 'Circulatie': 107647, 'Urinewegfunctie': 88961, 'Darmfunctie': 74604, 'Neuro_musculaire_skeletfunctie': 69166, 'Pijn': 66797, 'Geestelijke_gezondheid': 60760, 'Ademhaling': 42949, 'Fysieke_activiteit': 38888, 'Sociaal_contact': 32216, 'Mantelzorg_zorg_voor_kind_huisgenoot': 31205, 'Woning': 23749, 'Zicht': 22444, 'Gehoor': 21975, 'Mondgezondheid': 19341, 'Slaap_en_rust_patronen': 18775, 'Spijsvertering_vochthuishouding': 18605, 'Communicatie_met_maatschappelijke_voorzieningen': 17170, 'Interpersoonlijke_relaties': 12747, 'Besmettelijke_infectueuze_conditie': 10517, 'Rouw': 10035, 'Spraak_en_taal': 9496, 'Omgevings_hygiene': 7739, 'Buurt_werkplek_veiligheid': 7125, 'Gezondheidszorg_supervisie': 6720, 'Gebruik_van_verslavende_middelen': 5951, 'Bewustzijn': 3306, 'Rolverandering': 3055, 'Verwaarlozing': 2981, 'Inkomen_financien': 1673, 'Geslachtsorganen': 1666, 'Sexualiteit': 1105, 'Spiritualiteit': 775, 'Mishandeling_misbruik': 447, 'Groei_en_ontwikkeling': 77, 'Gezinsplanning': 12, 'Zwangerschap': 2, 'Postnataal': 0}

The first 3 top problems: Persoonlijke_zorg Medicatie Huid Circulatie
'''
from __future__ import print_function

import numpy as np

from sklearn.model_selection import train_test_split
import pandas as pd
import tensorflow as tf
print(tf.__version__)
import os.path
import matplotlib.pyplot as plt
from preproc import *

TRAINING_TYPE = 'simple' # chose from 'conv1d','conv2d','bidireLSTM'
FASTTEXT = False
SAVED_FILE_NAME = TRAINING_TYPE+'_FASTTEXT_prob' if FASTTEXT else TRAINING_TYPE+'_prob'
BASE_DIR = '../data/'
TEXT_DATA_DIR = os.path.join(BASE_DIR, '')
MAX_SEQUENCE_LENGTH = 1000
MAX_NUM_WORDS = 20000
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.2

import gensim
inputFile = BASE_DIR +'model.bin'
model = gensim.models.KeyedVectors.load_word2vec_format(inputFile, binary=True)
word_vectors = model.wv
print("Number of word vectors: {}".format(len(word_vectors.vocab)))
MAX_NB_WORDS = len(word_vectors.vocab)

# second, prepare text samples and their labels
print('Processing text dataset')
texts = []  # list of text samples
labels = []  # list of label ids
preproc = Preproc(base_dir = '../data/')

file_path = '../data/raw_data_preprocessed.pkl'
if os.path.exists(file_path):
    raw_data = pd.read_pickle('../data/raw_data_preprocessed.pkl')
else:
    raw_data = preproc.chunkesize_data()

print(raw_data.drop(columns = {'comment_text','id', 'PatientID'}).columns)

raw_data = raw_data.drop(columns ={'comment_text','id', 'PatientID'})
problems_list = list(raw_data.columns)
prob_freq = {}
for problem in problems_list:
    prob_freq[problem] = raw_data[raw_data[problem] == 1].shape[0]

prob_freq = {k: v for k, v in sorted(prob_freq.items(), key=lambda item: item[1],reverse=True)}
print("the problem and its sorted appeared amount: ", prob_freq)
problems_list = list(prob_freq.keys())
prob_freq = list(prob_freq.values())

fig, ax = plt.subplots()
plt.bar(problems_list, prob_freq)
plt.title("Amount of different problems")
plt.xticks(rotation=90)
fig.subplots_adjust(bottom=0.56)
plt.show()

comm_problems = problems_list[:5]
comm_prob_freq = prob_freq[:5]

# find the most co-occurent probelms with the 5 most frequent problems
for prob in comm_problems:
    temp = raw_data[raw_data[prob] == 1]
    prob_freq_sub = {}
    for problem in problems_list:
        prob_freq_sub[problem] = temp[temp[problem] == 1].shape[0]
        prob_freq_sub = {k: v for k, v in sorted(prob_freq_sub.items(), key=lambda item: item[1],reverse=True)}

    print(" the co-occurent problems with the problem "+ prob+ "is:", prob_freq_sub)
    fig, ax = plt.subplots()
    plt.bar(range(len(prob_freq_sub)), list(prob_freq_sub.values()), align='center')
    plt.xticks(range(len(prob_freq_sub)), list(prob_freq_sub.keys()))
    plt.title("Amount of problems that coocure with the "+ prob)
    plt.xticks(rotation=90)
    fig.subplots_adjust(bottom=0.56)
    plt.show()

# how many 1 in each row : the mean, median, max() and min()
prob_list = raw_data.reset_index().values.tolist()
prob_appear = [val.count(1) for val in  prob_list]
print("mean: ", np.mean(prob_appear))
print("median: ", np.median(prob_appear))
print("max: ", np.max(prob_appear))
print("mean: ", np.min(prob_list))
