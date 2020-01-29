# transformer for text classification
# Just average the states you get from the encoder;

from __future__ import absolute_import, division, print_function, unicode_literals
from sklearn.model_selection import train_test_split
# try:
#   !pip install -q tf-nightly
# except Exception:
#   pass
import tensorflow_datasets as tfds
import tensorflow as tf

import time
import numpy as np
import matplotlib.pyplot as plt

from preproc import *
print('Processing text dataset')
preproc = Preproc(base_dir = '../data/')

file_path = '../data/raw_data_preprocessed.pkl'
if os.path.exists(file_path):
    raw_data = pd.read_pickle('../data/raw_data.pkl')[:30000]
    # raw_data = pd.read_csv('../classify/raw_data.csv')[:1500]
else:
    raw_data = preproc.chunkesize_data()

# raw_data = raw_data[raw_data.prob_relate ==1]
print("data shape is:", raw_data.shape)

# remove the problem that contain less than 1w appearance
other_list = raw_data.reset_index()[['Sociaal_contact', 'Gehoor', 'Spijsvertering_vochthuishouding',
'Communicatie_met_maatschappelijke_voorzieningen', 'Besmettelijke_infectueuze_conditie',
'Slaap_en_rust_patronen', 'Woning', 'Mondgezondheid', 'Interpersoonlijke_relaties',
'Spraak_en_taal', 'Rouw', 'Gezondheidszorg_supervisie', 'Buurt_werkplek_veiligheid',
'Omgevings_hygiene', 'Rolverandering', 'Bewustzijn', 'Gebruik_van_verslavende_middelen','Geslachtsorganen', 'Verwaarlozing', 'Inkomen_financien', 'Spiritualiteit',
'Sexualiteit', 'Mishandeling_misbruik', 'Groei_en_ontwikkeling', 'Gezinsplanning', 'Postnataal', 'Zwangerschap']].values.tolist()

other_val = [1 if 1 in val else 0 for val in other_list ]
raw_data['other'] = other_val

prob_list_names = ['Persoonlijke_zorg', 'Medicatie', 'Huid', 'Circulatie', 'Voeding','Urinewegfunctie', 'Neuro_musculaire_skeletfunctie', 'Cognitie', 'Pijn',
'Darmfunctie', 'Geestelijke_gezondheid', 'Ademhaling', 'Mantelzorg_zorg_voor_kind_huisgenoot', 'Fysieke_activiteit', 'Zicht','other']

# remove rows that doesnt contain any problems
other_list = raw_data.reset_index()[prob_list_names].values.tolist()
contain_one = [True if 1 in val else False for val in other_list]
raw_data['contain_one'] = contain_one

print("data shape before filtering: ", raw_data.shape)
raw_data = raw_data[raw_data['contain_one'] == True]
print("data shape after filtering: ",raw_data.shape)


# check how many probems for each row
other_list = raw_data.reset_index()[prob_list_names].values.tolist()
one_list = [val.count(1) for val in other_list]
print("the mean amount of the problems is:",  np.mean(one_list))
print("the median amount of the problem is:", np.median(one_list))
print("the max amount of the problem is:", np.max(one_list))
print("the min amount of the problems is:", np.min(one_list))
# remove zero-variance columns
# raw_data = raw_data.loc[:, (raw_data != raw_data.iloc[0]).any()]

raw_data['comment_text'] = raw_data['comment_text'].apply(lambda x:" ".join([x for x in re.split("[^a-zA-Z]*",x) if len(x) > 1]))
raw_data['text_length'] = raw_data['comment_text'].apply(lambda x: len(x.split(" "))-1)
raw_data = raw_data[raw_data['text_length'] > 4]
raw_data = raw_data[raw_data['text_length'] < 300]
text_list = raw_data['comment_text'].tolist()

classes_num = raw_data[prob_list_names].shape[1]
output_tensor = raw_data[prob_list_names].values.tolist()

print("how many elements in text_list:", len(text_list))
text_list_test = []
output_tensor_test = []
for i in range(0,2000):
  text_list_test.append(text_list.pop())
  output_tensor_test.append(output_tensor.pop())
print(len(text_list_test))
print(len(text_list))

print(len(output_tensor_test))
print(len(output_tensor))
print("preview 5 samples: \n",raw_data['comment_text'].head(5))

# Try experimenting with the size of that dataset
input_tensor,inp_lang = preproc.load_dataset(tuple(text_list),pad=False)
# Calculate max_length of the target tensors
max_length_inp = preproc.max_length(input_tensor)

# raw_data['comment_text'].to_csv('../data/comment_text.csv',index=False)
# classes_num = raw_data.drop(columns=['comment_text','comment_text','id', 'PatientID','text_length']).shape[1]
# output_tensor = raw_data.drop(columns=['comment_text','comment_text','id', 'PatientID','text_length']).values.tolist()
# classes_num   = raw_data[['Medicatie']].shape[1]
# output_tensor = raw_data[['Medicatie']].values.tolist()

tokenizer_nl = tfds.features.text.SubwordTextEncoder.build_from_corpus(
    (nl for nl in text_list), target_vocab_size=2**9)

# test the tokenizer_nl
sample_string = 'wij zin hier'
tokenized_string = tokenizer_nl.encode(sample_string)
print ('Tokenized string is {}'.format(tokenized_string))

original_string = tokenizer_nl.decode(tokenized_string)
print ('The original string: {}'.format(original_string))

assert original_string == sample_string


# The tokenizer encodes the string by breaking it into subwords if the word is not in its dictionary.
for ts in tokenized_string:
  print ('{} ----> {}'.format(ts, tokenizer_nl.decode([ts])))

BUFFER_SIZE = 20000
BATCH_SIZE = 64
# Add a start and end token to the input and target.
def encode(lang1,label):
  lang1 = [tokenizer_nl.vocab_size] + tokenizer_nl.encode(lang1.numpy()) + [tokenizer_nl.vocab_size+1]
  return lang1, label

def tf_encode(en, label):
  result_en, label = tf.py_function(encode, [en, label], [tf.int64, tf.int64])
  result_en.set_shape([None])
  label.set_shape([classes_num])
  return result_en, label

# to keep this example small
MAX_LENGTH = 40
def filter_max_length(x,y, max_length=MAX_LENGTH):
  return tf.size(x) <= max_length

input_tensor_train,input_tensor_val,target_tensor_train,target_tensor_val = train_test_split(
    text_list,
    output_tensor,
    test_size=0.2,
    random_state = 42,
    shuffle = True)

print("how many training data:", len(input_tensor_train))
print("how many test data:", len(input_tensor_val))

input_tensor_train = tf.convert_to_tensor(input_tensor_train)
input_tensor_val = tf.convert_to_tensor(input_tensor_val)
input_tensor_test = tf.convert_to_tensor(text_list_test)
target_tensor_train = [[int(label) for label in element] for element in target_tensor_train]
target_tensor_val = [[int(label) for label in element] for element in target_tensor_val]
target_tensor_test = [[int(label) for label in element] for element in output_tensor_test]

target_tensor_train = tf.convert_to_tensor(target_tensor_train,dtype=np.int64)
target_tensor_val = tf.convert_to_tensor(target_tensor_val,dtype=np.int64)
target_tensor_test = tf.convert_to_tensor(target_tensor_test,dtype=np.int64)

train_examples = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train))
val_examples = tf.data.Dataset.from_tensor_slices((input_tensor_val, target_tensor_val))
test_examples = tf.data.Dataset.from_tensor_slices((input_tensor_test, target_tensor_test))

train_dataset = train_examples.map(tf_encode)
train_dataset = train_dataset.filter(filter_max_length)

# cache the dataset to memory to get a speedup while reading from it.
train_dataset = train_dataset.cache()
train_dataset = train_dataset.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE,padded_shapes=([None],[classes_num]))
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

# validation only one batch
val_dataset = val_examples.map(tf_encode)
val_dataset = val_dataset.filter(filter_max_length).padded_batch(len(input_tensor_val),padded_shapes=([None],[classes_num]))

# test only one batch
test_dataset = test_examples.map(tf_encode)
test_dataset = test_dataset.filter(filter_max_length).padded_batch(len(input_tensor_test),padded_shapes=([None],[classes_num]))

en_batch,label = next(iter(val_dataset))
print("the next batch of the dataset is:", en_batch)
print("the next batch of the label is:", label)

## positionl encoding
def get_angles(pos, i, d_model):
  angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
  return pos * angle_rates

def positional_encoding(position, d_model):
  angle_rads = get_angles(np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)

  # apply sin to even indices in the array; 2i
  angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
  # apply cos to odd indices in the array; 2i+1
  angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
  pos_encoding = angle_rads[np.newaxis, ...]

  return tf.cast(pos_encoding, dtype=tf.float32)

# test
pos_encoding = positional_encoding(50, 512)
print ('pos_encoding = positional_encoding(50, 512) :\n',pos_encoding.shape)
# plt.pcolormesh(pos_encoding[0], cmap='RdBu')
# plt.xlabel('Depth')
# plt.xlim((0, 512))
# plt.ylabel('Position')
# plt.colorbar()
# plt.show()


## Masing
def create_padding_mask(seq):
  seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
  # add extra dimensions to add the padding
  # to the attention logits.
  return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


x = tf.constant([[7, 6, 0, 0, 1], [1, 2, 3, 0, 0], [0, 0, 0, 4, 5]])
print("created padding mask from x = tf.constant([[7, 6, 0, 0, 1], [1, 2, 3, 0, 0], [0, 0, 0, 4, 5]]):\n",
      create_padding_mask(x))

def create_look_ahead_mask(size):
  mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
  return mask  # (seq_len, seq_len)

x = tf.random.uniform((1, 3))
temp = create_look_ahead_mask(x.shape[1])
print(temp)


def scaled_dot_product_attention(q, k, v, mask):
  """Calculate the attention weights.
  q, k, v must have matching leading dimensions.
  k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
  The mask has different shapes depending on its type(padding or look ahead)
  but it must be broadcastable for addition.

  Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable
          to (..., seq_len_q, seq_len_k). Defaults to None.

  Returns:
    output, attention_weights
  """

  matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

  # scale matmul_qk
  dk = tf.cast(tf.shape(k)[-1], tf.float32)
  scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

  # add the mask to the scaled tensor.
  if mask is not None:
    scaled_attention_logits += (mask * -1e9)

  # softmax is normalized on the last axis (seq_len_k) so that the scores
  # add up to 1.
  attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

  output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

  return output, attention_weights


def scaled_dot_product_attention(q, k, v, mask):
  """Calculate the attention weights.
  q, k, v must have matching leading dimensions.
  k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
  The mask has different shapes depending on its type(padding or look ahead)
  but it must be broadcastable for addition.

  Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable
          to (..., seq_len_q, seq_len_k). Defaults to None.

  Returns:
    output, attention_weights
  """

  matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

  # scale matmul_qk
  dk = tf.cast(tf.shape(k)[-1], tf.float32)
  scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

  # add the mask to the scaled tensor.
  if mask is not None:
    scaled_attention_logits += (mask * -1e9)

  # softmax is normalized on the last axis (seq_len_k) so that the scores
  # add up to 1.
  attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

  output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

  return output, attention_weights

def print_out(q, k, v):
  temp_out, temp_attn = scaled_dot_product_attention(
      q, k, v, None)
  print ('Attention weights are:')
  print (temp_attn)
  print ('Output is:')
  print (temp_out)

np.set_printoptions(suppress=True)

temp_k = tf.constant([[10,0,0],
                      [0,10,0],
                      [0,0,10],
                      [0,0,10]], dtype=tf.float32)  # (4, 3)

temp_v = tf.constant([[   1,0],
                      [  10,0],
                      [ 100,5],
                      [1000,6]], dtype=tf.float32)  # (4, 2)

# This `query` aligns with the second `key`,
# so the second `value` is returned.
temp_q = tf.constant([[0, 10, 0]], dtype=tf.float32)  # (1, 3)
print_out(temp_q, temp_k, temp_v)

# This query aligns with a repeated key (third and fourth),
# so all associated values get averaged.
temp_q = tf.constant([[0, 0, 10]], dtype=tf.float32)  # (1, 3)
print_out(temp_q, temp_k, temp_v)

# This query aligns equally with the first and second key,
# so their values get averaged.
temp_q = tf.constant([[10, 10, 0]], dtype=tf.float32)  # (1, 3)
print_out(temp_q, temp_k, temp_v)
# put all the queries together
temp_q = tf.constant([[0, 0, 10], [0, 10, 0], [10, 10, 0]], dtype=tf.float32)  # (3, 3)
print_out(temp_q, temp_k, temp_v)

# Multi-head attention
class MultiHeadAttention(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads):
    super(MultiHeadAttention, self).__init__()
    self.num_heads = num_heads
    self.d_model = d_model

    assert d_model % self.num_heads == 0

    self.depth = d_model // self.num_heads

    self.wq = tf.keras.layers.Dense(d_model)
    self.wk = tf.keras.layers.Dense(d_model)
    self.wv = tf.keras.layers.Dense(d_model)

    self.dense = tf.keras.layers.Dense(d_model)

  def split_heads(self, x, batch_size):
    """Split the last dimension into (num_heads, depth).
    Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
    """
    x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(x, perm=[0, 2, 1, 3])

  def call(self, v, k, q, mask):
    batch_size = tf.shape(q)[0]

    q = self.wq(q)  # (batch_size, seq_len, d_model)
    k = self.wk(k)  # (batch_size, seq_len, d_model)
    v = self.wv(v)  # (batch_size, seq_len, d_model)

    q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
    k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
    v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

    # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
    # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
    scaled_attention, attention_weights = scaled_dot_product_attention(
        q, k, v, mask)

    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

    concat_attention = tf.reshape(scaled_attention,
                                  (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

    output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

    return output, attention_weights

temp_mha = MultiHeadAttention(d_model=512, num_heads=8)
y = tf.random.uniform((1, 60, 512))  # (batch_size, encoder_sequence, d_model)
out, attn = temp_mha(y, k=y, q=y, mask=None)
print(out.shape)
print(attn.shape)

# Point wise feed forward network
def point_wise_feed_forward_network(d_model, dff):
  return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
      tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
  ])
sample_ffn = point_wise_feed_forward_network(512, 2048)
print(sample_ffn(tf.random.uniform((64, 50, 512))).shape)


class EncoderLayer(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads, dff, rate=0.1):
    super(EncoderLayer, self).__init__()

    self.mha = MultiHeadAttention(d_model, num_heads)
    self.ffn = point_wise_feed_forward_network(d_model, dff)

    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    self.dropout1 = tf.keras.layers.Dropout(rate)
    self.dropout2 = tf.keras.layers.Dropout(rate)

  def call(self, x, training, mask):

    attn_output, attenweights = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
    attn_output = self.dropout1(attn_output, training=training)
    out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

    ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
    ffn_output = self.dropout2(ffn_output, training=training)
    out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

    return out2,attenweights


sample_encoder_layer= EncoderLayer(512, 8, 2048)
sample_encoder_layer_output,attenweights = sample_encoder_layer(tf.random.uniform((64, 43, 512)), False, None)
print(sample_encoder_layer_output.shape)  # (batch_size, input_seq_len, d_model)
print("attention weights in the encoder:", attenweights.shape)

class Encoder(tf.keras.layers.Layer):
  def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, maximum_position_encoding, rate=0.1):
    super(Encoder, self).__init__()

    self.d_model = d_model
    self.num_layers = num_layers

    self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
    self.pos_encoding = positional_encoding(maximum_position_encoding,
                                            self.d_model)
    self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                       for _ in range(num_layers)]

    self.dropout = tf.keras.layers.Dropout(rate)

  def call(self, x, training, mask):

    seq_len = tf.shape(x)[1]
    attention_weights = {}
    # adding embedding and position encoding.
    x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x += self.pos_encoding[:, :seq_len, :]

    x = self.dropout(x, training=training)

    for i in range(self.num_layers):
      x, atten = self.enc_layers[i](x, training, mask)
      attention_weights['encoder_layer{}'.format(i+1)] = atten

    return x,attention_weights # (batch_size, input_seq_len, d_model)

sample_encoder = Encoder(num_layers=2, d_model=512, num_heads=8,
                         dff=2048, input_vocab_size=8500,
                         maximum_position_encoding=10000)
temp_input = tf.random.uniform((64, 62), dtype=tf.int64, minval=0, maxval=200)
sample_encoder_output,atten = sample_encoder(temp_input, training=False, mask=None)
print(sample_encoder_output.shape)  # (batch_size, input_seq_len, d_model)
print(atten)

class Transformer(tf.keras.Model):
  def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, class_num, pe_input, rate=0.1):
    super(Transformer, self).__init__()

    self.encoder = Encoder(num_layers, d_model, num_heads, dff, input_vocab_size, pe_input, rate)
    self.final_layer = tf.keras.layers.Dense(class_num,activation='sigmoid')

  def call(self, inp, training, enc_padding_mask):

    enc_output,atten = self.encoder(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)
    enc_output = tf.reduce_mean(enc_output,1) # sigmoid((batch_size, d_model) * W (d_model, class_num)) = (batch_size,class_num)
    final_output = self.final_layer(enc_output)  # (batch_size, tar_seq_len, target_vocab_size)

    return final_output,atten


sample_transformer = Transformer(
    num_layers=2, d_model=512, num_heads=8, dff=2048,
    input_vocab_size=8500, class_num = classes_num,
    pe_input=10000)
temp_input = tf.random.uniform((64, 38), dtype=tf.int64, minval=0, maxval=200)
fn_out,atten= sample_transformer(temp_input,training=False,enc_padding_mask=None)
print(fn_out.shape)  # (batch_size, tar_seq_len, target_vocab_size)
print('attention weights are:', atten)

# set hyperparameters
num_layers = 6
d_model = 512
dff = 2048
num_heads = 8

input_vocab_size = tokenizer_nl.vocab_size + 2
dropout_rate = 0.1

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=4000):
    super(CustomSchedule, self).__init__()

    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps

  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


learning_rate = CustomSchedule(d_model)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
temp_learning_rate_schedule = CustomSchedule(d_model)
plt.plot(temp_learning_rate_schedule(tf.range(40000, dtype=tf.float32)))
plt.ylabel("Learning Rate")
plt.xlabel("Train Step")


def loss_function(real, pred):
  loss_ = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction='none')(real, pred)
  return tf.reduce_mean(loss_)


train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')

transformer = Transformer(num_layers, d_model, num_heads, dff,
                          input_vocab_size, class_num = classes_num,
                          pe_input=input_vocab_size,
                          rate=dropout_rate)

checkpoint_path = "./model_saves/transformer_en"
ckpt = tf.train.Checkpoint(transformer=transformer,optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
  ckpt.restore(ckpt_manager.latest_checkpoint)
  print ('Latest checkpoint restored!!')

EPOCHS = 5
# The @tf.function trace-compiles train_step into a TF graph for faster
# execution. The function specializes to the precise shape of the argument
# tensors. To avoid re-tracing due to the variable sequence lengths or variable
# batch sizes (the last batch is smaller), use input_signature to specify
# more generic shapes.
train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, classes_num), dtype=tf.int64),
]

@tf.function(input_signature=train_step_signature)
def train_step(inp, tar):

  enc_padding_mask = create_padding_mask(inp)
  # print("inp is:", inp)
  # print('enc_padding_mask',enc_padding_mask)
  # print('tar:',tar)
  with tf.GradientTape() as tape:
    predictions,_ = transformer(inp,training=True, enc_padding_mask = enc_padding_mask)
    # print("prediction is:",predictions)
    loss = loss_function(tar, predictions)

  gradients = tape.gradient(loss, transformer.trainable_variables)
  optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

  train_loss(loss)
  train_accuracy(tar, tf.cast(predictions>0.5, tf.int64))


### Trianing
for epoch in range(EPOCHS):
  start = time.time()

  train_loss.reset_states()
  train_accuracy.reset_states()

  # inp -> portuguese, tar -> english
  for (batch, (inp, label)) in enumerate(train_dataset):
    train_step(inp, label)

    if batch % 50 == 0:
      print ('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
          epoch + 1, batch, train_loss.result(), train_accuracy.result()))

  if (epoch + 1) % 5 == 0:
    ckpt_save_path = ckpt_manager.save()
    print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,ckpt_save_path))

  print ('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, train_loss.result(), train_accuracy.result()))
  print ('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))



###  evalutaion
# @tf.function(input_signature=train_step_signature)
def eval_step(inp, tar):
  enc_padding_mask = create_padding_mask(inp)
  predictions,_ = transformer(inp,training=False, enc_padding_mask = enc_padding_mask)
  loss = loss_function(tar, predictions)

  train_loss(loss)
  prediction = tf.cast(predictions>0.5, tf.int64)

  train_accuracy(tar, prediction)
  for i in range(0,classes_num):
    print("the accuracy for the problem",prob_list_names[i])
    print("acc is:", str((tar.numpy()[:,i]== prediction.numpy()[:,i]).sum() / tar.numpy().shape[0]))

train_loss.reset_states()
train_accuracy.reset_states()
# inp -> portuguese, tar -> english
for (batch, (inp, label)) in enumerate(test_dataset):
    eval_step(inp, label)
    print ('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(1, batch, train_loss.result(), train_accuracy.result()))

# if (epoch + 1) % 5 == 0:
#     ckpt_save_path = ckpt_manager.save()

print ('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(1,train_loss.result(),train_accuracy.result()))


## visualization
def plot_attention_weights(attention, sentence, plot):
  fig = plt.figure(figsize=(16, 8))
  sentence = tokenizer_nl.encode(sentence)
  attention = tf.squeeze(attention[plot], axis=0)

  for head in range(attention.shape[0]):
    ax = fig.add_subplot(8,1, head+1)

    # plot the attention weights
    ax.matshow(tf.expand_dims(tf.math.reduce_mean(attention[head],0),0), cmap='viridis')
    # ax.matshow(attention[head][:-1, :], cmap='viridis')

    fontdict = {'fontsize': 10}

    ax.set_xticks(range(len(sentence)+2))
    ax.set_yticks(range(1))

    # ax.set_ylim(len(sentence)-1.5, -0.5)
    ax.set_xticklabels(
        ['<start>']+[tokenizer_nl.decode([i]) for i in sentence]+['<end>'],
        fontdict=fontdict, rotation=90)

    ax.set_yticklabels(['weights'],fontdict=fontdict)

    ax.set_xlabel('Head {}'.format(head+1))

  plt.tight_layout()
  plt.show()

def evaluate(inp_sentence):
  inp_sentence = [tokenizer_nl.vocab_size] + tokenizer_nl.encode(inp_sentence) + [tokenizer_nl.vocab_size+1]
  inp_sentence = tf.expand_dims(inp_sentence, 0)

  # for i in range(MAX_LENGTH):
  enc_padding_mask= create_padding_mask(inp_sentence)
  # predictions.shape == (batch_size, seq_len, vocab_size)
  predictions, attention_weights =  transformer(inp_sentence,training=False, enc_padding_mask = enc_padding_mask)

  return predictions,attention_weights

def translate(sentence, plot=''):
  result, attention_weights = evaluate(sentence)
  print('Input: {}'.format(sentence))
  print('Predicted translation: {}'.format(result))

  if plot:
    plot_attention_weights(attention_weights, sentence, plot)

translate("dochter is zaterdag met hr naar de hapost geweest voor zijn dikke voet waarschijnlijk is dit jicht hr heeft "
          "een prednison kuur gekregen gisteravond de tweede tablet ingenomen vanavond derde tablet geven ligt in het kastje",
          plot='encoder_layer6')
# print ("Real translation: this is a problem we have to solve .")

