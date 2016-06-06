# -*- coding: utf-8 -*-

from __future__ import print_function
from keras.preprocessing import sequence
from keras.regularizers import l2
from keras.constraints import maxnorm
from keras.layers import Dense, Dropout, Lambda, Input, merge
from keras.layers import Embedding
from keras.layers import Convolution1D
from keras.models import Model
from keras.utils import np_utils
from keras import backend as K

import numpy as np
import cPickle as pickle

np.random.seed(1337)  # for reproducibility

vocabulary = pickle.load(open('data/mr-id2word.pkl'))

# set parameters:
max_features = len(vocabulary) + 1
maxlen = 56
batch_size = 50
embedding_dims = 300
nb_filter = 30
# filter_length = 3
hidden_dims = 100
nb_epoch = 20

print('Loading data...')

X_train_pos = pickle.load(open('data/mr-train-pos.pkl'))
X_train_neg = pickle.load(open('data/mr-train-neg.pkl'))
y_train = list(np.ones((len(X_train_pos,)), dtype='int8')) + list(np.zeros((len(X_train_neg,)), dtype='int8'))
X_train = X_train_pos + X_train_neg
y_train = np.asarray(y_train)
y_train = np_utils.to_categorical(y_train, 2)

X_test_pos = pickle.load(open('data/mr-test-pos.pkl'))
X_test_neg = pickle.load(open('data/mr-test-neg.pkl'))
y_test = list(np.ones((len(X_test_pos,)), dtype='int8')) + list(np.zeros((len(X_test_neg,)), dtype='int8'))
X_test = X_test_pos + X_test_neg
y_test = np.asarray(y_test)
y_test = np_utils.to_categorical(y_test, 2)

print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')

print('Pad sequences (samples x time)')
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

print('Build model...')
input_sentence = Input(shape=(maxlen,), dtype='int32')

# we start off with an efficient embedding layer which maps
# our vocab indices into embedding_dims dimensions
input_embedding = Embedding(max_features, embedding_dims,
                            weights=[np.load('mr_word2vec_300_dim_google.embeddings')])(input_sentence)

cnns = [Convolution1D(filter_length=filter_length,
                      nb_filter=nb_filter,
                      W_regularizer=l2(0.0001),
                      W_constraint=maxnorm(3),
                      activation='relu',
                      border_mode='same') for filter_length in [2, 3, 5, 7]]

cnn_feature = merge([cnn(input_embedding) for cnn in cnns], mode='concat')

dropout = Dropout(0.25)
sentence_dropout = dropout(cnn_feature)


maxpool = Lambda(lambda x: K.max(x, axis=1, keepdims=False), output_shape=lambda x: (x[0], x[2]))
sentence_pool = maxpool(sentence_dropout)

predict_sentiment = Dense(2, activation='softmax')(sentence_pool)

model = Model(input=[input_sentence], output=[predict_sentiment])
model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit([X_train], [y_train], nb_epoch=nb_epoch, verbose=1, validation_data=(X_test, y_test))
