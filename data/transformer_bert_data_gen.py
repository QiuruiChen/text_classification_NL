from __future__ import absolute_import, division, print_function, unicode_literals
import os
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'your google credential json file path-ce818938681b.json'
# from google.cloud import translate_v2 as translate
import pandas as pd

def save_data(start_pos,end_pos,data_type):
    raw_data = pd.read_pickle('../data/raw_data.pkl')[start_pos:end_pos]
    raw_data['text_length'] = raw_data['comment_text'].apply(lambda x: len(x.split(" "))-1)
    raw_data = raw_data[raw_data['text_length'] > 4]

    print("training data size:", raw_data.shape[0])

    # remove the problem that contain less than 1w appearance
    other_list = raw_data.reset_index()[['Sociaal_contact', 'Gehoor', 'Spijsvertering_vochthuishouding',
    'Communicatie_met_maatschappelijke_voorzieningen', 'Besmettelijke_infectueuze_conditie',
    'Slaap_en_rust_patronen', 'Woning', 'Mondgezondheid', 'Interpersoonlijke_relaties',
    'Spraak_en_taal', 'Rouw', 'Gezondheidszorg_supervisie', 'Buurt_werkplek_veiligheid',
    'Omgevings_hygiene', 'Rolverandering', 'Bewustzijn', 'Gebruik_van_verslavende_middelen','Geslachtsorganen', 'Verwaarlozing', 'Inkomen_financien', 'Spiritualiteit',
    'Sexualiteit', 'Mishandeling_misbruik', 'Groei_en_ontwikkeling', 'Gezinsplanning', 'Postnataal', 'Zwangerschap']].values.tolist()

    other_val = [1 if 1 in val else 0 for val in other_list ]
    raw_data['other'] = other_val

    prob_list_names = ['Persoonlijke_zorg', 'Medicatie', 'Huid', 'Circulatie', 'Voeding','Urinewegfunctie',
                       'Neuro_musculaire_skeletfunctie', 'Cognitie', 'Pijn','Darmfunctie', 'Geestelijke_gezondheid',
                       'Ademhaling', 'Mantelzorg_zorg_voor_kind_huisgenoot', 'Fysieke_activiteit', 'Zicht','other']

    raw_data[prob_list_names] = raw_data[prob_list_names].astype(int)
    raw_data[['comment_text']+prob_list_names].to_csv('transformer_bert_'+data_type+'.csv',index=False)


if __name__ == "__main__":
    # generate 20000 training data
    save_data(0,20000,'train')
    # generating 5000 validation data
    save_data(20000,25000,'val')
    # generating 5k test data
    save_data(25000,30000,'test')
