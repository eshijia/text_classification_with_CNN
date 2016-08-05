# -*- coding: utf-8 -*-

from __future__ import print_function
from gensim.models import Word2Vec
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Flatten, Dropout, merge, Input
from keras.engine import Model
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.constraints import maxnorm
from keras.regularizers import l2

import os
import numpy as np
np.random.seed(1337)


MAX_SEQUENCE_LENGTH = 60
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.1

nb_filter = 250
filter_lengths = [2, 3, 5, 7]

print('Indexing word vectors.')

pre_trained_embeddings = Word2Vec.load_word2vec_format('data/GoogleNews-vectors-negative300.bin', binary=True)

weights = pre_trained_embeddings.syn0
embeddings_index = dict([(k.encode('utf-8'), v.index) for k, v in pre_trained_embeddings.vocab.items()])

print('Found %s word vectors.' % len(embeddings_index))

print('Processing text dataset')

DATA_DIR = 'data/MR'
texts = []
labels_index = {}
labels = []

for name in sorted(os.listdir(DATA_DIR)):
    labels_id = len(labels_index)
    labels_index[name.split('.')[0]] = labels_id
    f = open(os.path.join(DATA_DIR, name))
    for line in f.readlines():
        texts.append(line)
        labels.append(labels_id)

print('Found %s texts.' % len(texts))

tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]

print('Preparing embedding matrix.')

nb_words = min(MAX_NB_WORDS, len(word_index))
embedding_matrix = np.zeros((nb_words + 1, EMBEDDING_DIM))

for word, i in word_index.items():
    if i > MAX_NB_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = weights[embeddings_index[word], :]

embedding_layer = Embedding(nb_words + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=True)

print('Training model.')

sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)

cnns = []

for filter_length in filter_lengths:
    x = Conv1D(nb_filter=nb_filter,
               filter_length=filter_length,
               border_mode='valid',
               activation='relu',
               W_constraint=maxnorm(3),
               W_regularizer=l2(0.0001),
               subsample_length=1)(embedded_sequences)
    x = MaxPooling1D(pool_length=MAX_SEQUENCE_LENGTH - filter_length + 1)(x)
    x = Flatten()(x)
    cnns.append(x)

x = merge(cnns, mode='concat')
x = Dropout(0.2)(x)
x = Dense(128, activation='relu')(x)
preds = Dense(len(labels_index), activation='softmax')(x)

model = Model(sequence_input, preds)

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# happy learning!
model.fit(x_train, y_train, validation_data=(x_val, y_val),
          nb_epoch=5, batch_size=128)
