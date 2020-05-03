# Text classification

Classifying patient illness name by nurse notes (Dutch). \

Two type methods are applied: \
1. Using non-transformer methods:
   - preprocessing text
   - applying:
     - word embedding + globalAveraged1D
     - Or word embedding + convolutional 1D
     - Or word embedding + convolutional 2D
     - Or bidirectional LSTM
     - Or FastText (n-gram based on character-level)
2. Transformer-based methods:
   - BERT
   - XLMRoberta

## File structure:
- data:
  - [transformer_bert_data_gen.py](data/transformer_bert_data_gen.py): generating train/dev/test text for BERT
- result:
  - [conv1d](result/conv1d): reult for `embedding + convolutional 1D` method
  - [result_XLMRoberta](result/result_XLMRoberta): result for XLMRoberta
  - transformer_bert_layer0/11: attention plot for first and last layer
  - [transformer_distilledBert](result/transformer_distilledBert): esult for distilledBert

- [preproc.py](preproc.py): preprocessing text
- [transformer_classifier_tf.py](transformer_classifier_tf.py):
  - transfromer for tensorflow
- [transformer_classifier_torch.py](transformer_classifier_torch.py):
  - transformer for PyTorch
- [transformers_tools.py](transformers_tools.py):
  - help functions for transformers
- [word_embedding_methods.py](word_embedding_methods.py):
  -  non-transformer methods:
     - word embedding + globalAveraged1D
       ```bash
      python --train_type simple
      ```
     - Or word embedding + convolutional 1D:
      ```bash
      python --train_type conv1d
      ```
     - Or word embedding + convolutional 2D
      ```bash
      python --train_type conv2d
      ```
     - Or bidirectional LSTM:
      ```bash
      python --train_type bidireLSTM
      ```
     - Or FastText (n-gram based on character-level): `python --fasttext True`

