# -*- coding: utf-8 -*-
from __future__ import print_function

import os
import sys
import random
import numpy as np

import pickle

from gensim.models import Word2Vec

random.seed(42)
os.environ['MR'] = 'data'


def load(path, name):
    return pickle.load(open(os.path.join(path, name), 'rb'))


def revert(vocab, indices):
    return [vocab.get(i, 'X') for i in indices]

if __name__ == '__main__':
    try:
        data_path = os.environ['MR']
    except KeyError:
        print("MR is not set.  Set it to your path of MR")
        sys.exit(1)

    vocab = load(data_path, 'mr-id2word.pkl')

    sentences = list()

    for sent in load(data_path, 'mr-train-pos.pkl'):
        sentences.append(revert(vocab, sent))

    for sent in load(data_path, 'mr-train-neg.pkl'):
        sentences.append(revert(vocab, sent))

    for sent in load(data_path, 'mr-test-pos.pkl'):
        sentences.append(revert(vocab, sent))

    for sent in load(data_path, 'mr-test-neg.pkl'):
        sentences.append(revert(vocab, sent))

    model = Word2Vec(sentences, size=300, min_count=2, window=5, sg=1, iter=25)
    weights = model.syn0
    d = dict([(k, v.index) for k, v in model.vocab.items()])

    # this is the stored weights of an equivalent embedding layer
    emb = np.zeros(shape=(len(vocab)+1, 300))

    # swap the word2vec weights with the embedded weights
    for i, w in vocab.items():
        if w not in d:
            continue
        emb[i, :] = weights[d[w], :]

    np.save(open('mr_word2vec_300_dim.embeddings', 'wb'), emb)
