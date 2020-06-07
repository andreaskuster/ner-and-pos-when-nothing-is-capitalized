#!/usr/bin/env python3
# encoding: utf-8

"""
    Copyright (C) 2020  Andreas Kuster

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

__author__ = "Andreas Kuster"
__copyright__ = "Copyright 2020"
__license__ = "GPL"


from keras.models import Sequential
from keras.layers import Activation
from keras.optimizers import Adam
from keras.layers import LSTM, Dense, Bidirectional, TimeDistributed
from keras_contrib.layers import CRF
from keras.utils import to_categorical
from keras.models import load_model
from keras_contrib.losses import crf_loss
from keras_contrib.metrics.crf_accuracies import crf_viterbi_accuracy

from numpy import save, load
import numpy as np
from os.path import isfile

from datasets.ptb.penn_treebank import PTB
from embeddings.word2vec.word2vec import Word2Vec
from embeddings.elmo.elmov2 import ELMo
from embeddings.glove.glove import GloVe
from helper.multi_gpu import to_multi_gpu

"""
    This is the flat/initial implementation of the pos framework. Have a look at this if you want to learn something 
    respectively learn how the internals work. For actual tests, please use the fully structured and functional improved
    pos.py.

    Improvement ideas:
        - k-fold cross validation: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html

    File placement:
        - add penn treebank wsj files into ~/nltk_data/corpora/treebank/combined (extracted from 00-24 structure)

    Package Install:
        - keras_contrib: pip install git+https://www.github.com/keras-team/keras-contrib.git

    Credits:
        - https://nlpforhackers.io/lstm-pos-tagger-keras/
        - https://machinelearningmastery.com/save-load-keras-deep-learning-models/
        - CRF:
        - https://createmomo.github.io/2017/09/12/CRF_Layer_on_the_Top_of_BiLSTM_1/
        - https://github.com/keras-team/keras-contrib/blob/master/keras_contrib/layers/crf.py
        - https://github.com/Hironsan/keras-crf-layer
        - https://www.youtube.com/watch?v=rc3YDj5GiVM
        - http://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/18/image-segmentation-with-tensorflow-using-cnns-and-conditional-random-fields/
"""

# show only relevant cuda gpu devices (i.e. mask some for parallel jobs), comment out to show all
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# define relevant model parameters
_DATASET = "dummy"
_EMBEDDINGS = "elmo"
_DATA_SOURCE_WORD2VEC = "word2vec-google-news-300"
_DATA_SOURCE_GLOVE = "common_crawl_840b_cased"
_DATA_SET_GLOVE = "glove.840B.300d.txt"
_NUM_GPUS = 1  # number of gpus used for the model
_BATCH_SIZE_EMBEDDINGS = 4096
_SAVE_EMBEDDING_ELMO = False  # store embedded dataset to disk to save the memory/compute intensive re-computation
_LOAD_EMBEDDING_ELMO = True  # load embedded dataset from disk
_ADD_CRF = False  # add a crf layer on top of the BiLSTM model


#################################
# Import data
#################################
print("Importing Penn Treebank data...")

# instantiate Penn TreeBank data loader
ptb = PTB()

if _DATASET == "dummy":  # load small dummy data set for testing
    # load two sentence dummy dataset
    x, y = ptb.load_data([0])
    trainX, trainY = x[0:10], y[0:10]
    devX, devY = x[0:4], y[0:4]
    testX, testY = x[4:10], y[4:10]

else:  # load actual dataset
    # train: 0..18 -> range(0, 19)
    trainX, trainY = ptb.load_data(range(0, 19))
    # dev: 19..21 -> range(19, 22)
    devX, devY = ptb.load_data(range(19, 22))
    # test: 22..24 -> range(22, 25)
    testX, testY = ptb.load_data(range(22, 25))

#################################
# Pad sequences
#################################
print("Padding sequence...")

# compute maximum sentence length
token_train_len = [len(x) for x in trainX]
token_dev_len = [len(x) for x in devX]
token_test_len = [len(x) for x in testX]
max_len = max(token_train_len + token_dev_len + token_test_len)

# pad all sequences to max_len
for i in range(len(trainX)):
    no_pad = max_len - len(trainX[i])
    trainX[i] = list(trainX[i]) + ([""] * no_pad)
for i in range(len(devX)):
    no_pad = max_len - len(devX[i])
    devX[i] = list(devX[i]) + ([""] * no_pad)
for i in range(len(testX)):
    no_pad = max_len - len(testX[i])
    testX[i] = list(testX[i]) + ([""] * no_pad)

for i in range(len(trainY)):
    no_pad = max_len - len(trainY[i])
    trainY[i] = list(trainY[i]) + ([""] * no_pad)
for i in range(len(devY)):
    no_pad = max_len - len(devY[i])
    devY[i] = list(devY[i]) + ([""] * no_pad)
for i in range(len(testY)):
    no_pad = max_len - len(testY[i])
    testY[i] = list(testY[i]) + ([""] * no_pad)

#################################
# Map to embeddings
#################################

if _EMBEDDINGS == "word2vec":
    # instantiate preprocessor
    preprocessor = Word2Vec()
    # download pre-trained data
    preprocessor.import_pre_trained_data(_DATA_SOURCE_WORD2VEC)

elif _EMBEDDINGS == "elmo":
    # instantiate preprocessor
    preprocessor = ELMo()

elif _EMBEDDINGS == "glove":
    # instantiate preprocessor
    preprocessor = GloVe()
    # prepare pre-trained data
    preprocessor.import_pre_trained_data(_DATA_SOURCE_GLOVE, _DATA_SET_GLOVE)

else:
    raise RuntimeError("No embeddings preprocessor selected.")

if _EMBEDDINGS == "elmo":

    if _SAVE_EMBEDDING_ELMO:

        print("start elmo embedding...")
        # compute the number of required batches
        num_batches = int(len(trainX) / _BATCH_SIZE_EMBEDDINGS)
        # create embedding per batch
        trainX_embeddings = list()
        for i in range(num_batches):
            start = i * _BATCH_SIZE_EMBEDDINGS
            end = (i + 1) * _BATCH_SIZE_EMBEDDINGS
            trainX_embeddings.append(preprocessor.embedding(trainX[start:end]))
            print("elmo training embedding  round {}...".format(i))
        trainX_embeddings = np.array(trainX_embeddings)
        trainX_embeddings = trainX_embeddings.reshape(-1, trainX_embeddings[0].shape[-2], trainX_embeddings[0].shape[-1])
        # process remainder
        if end < len(trainX):
            trainX_embeddings = np.concatenate((trainX_embeddings, preprocessor.embedding(trainX[end:])), axis=0)
            print("elmo training embedding  round remainder...")
        # compute the number of required batches
        num_batches = int(len(testX) / _BATCH_SIZE_EMBEDDINGS)
        # create embedding per batch
        testX_embeddings = list()
        for i in range(num_batches):
            start = i * _BATCH_SIZE_EMBEDDINGS
            end = (i + 1) * _BATCH_SIZE_EMBEDDINGS
            testX_embeddings.append(preprocessor.embedding(testX[start:end]))
            print("elmo test embedding  round {}...".format(i))

        testX_embeddings = np.array(testX_embeddings)
        testX_embeddings = testX_embeddings.reshape(-1, testX_embeddings.shape[-2], testX_embeddings.shape[-1])
        # process remainder
        if end < len(testX):
            testX_embeddings = np.concatenate((testX_embeddings, preprocessor.embedding(testX[end:])), axis=0)
            print("elmo test embedding  round remainder...")
        # store the result
        save('trainX_embeddings_elmo.npy', trainX_embeddings)
        save('testX_embeddings_elmo.npy', testX_embeddings)
    # load embedding from file
    if _LOAD_EMBEDDING_ELMO:
        trainX_embeddings = load('trainX_embeddings_elmo.npy')
        testX_embeddings = load('testX_embeddings_elmo.npy')
else:
    # use word2vec instead
    trainX_embeddings = list()
    for sentence in trainX:
        trainX_embeddings.append([preprocessor.word2vec(word) for word in sentence])
    trainX_embeddings = np.array(trainX_embeddings)
    testX_embeddings = list()
    for sentence in testX:
        testX_embeddings.append([preprocessor.word2vec(word) for word in sentence])
    testX_embeddings = np.array(testX_embeddings)

dim_embedding_vec = preprocessor.dim

#################################
# Map y
#################################

# convert to categorical (assign each label a distinct number, and convert all labels to the corresponding numbers)
labels = set()
for item in trainY + devY + testY:
    for label in item:
        labels.add(label)

num_categories = len(labels)  # number of pos tags
int2word = list(labels)
word2int = {label: i for i, label in enumerate(int2word)}


trainY_int = list()
for sentence in trainY:
    trainY_int.append([word2int[word] for word in sentence])
testY_int = list()
for sentence in testY:
    testY_int.append([word2int[word] for word in sentence])


#################################
# Define BiLSTM/BiLSTM-CRF model
#################################

# try to load the model
if isfile("model.h5"):
    model = load_model("model.h5",
                       custom_objects={"CRF": CRF,
                                       "crf_loss": crf_loss,
                                       "crf_viterbi_accuracy": crf_viterbi_accuracy})
else:  # create new model isntead

    if _ADD_CRF:  # BiLSTM-CRF
        # note: no multi gpu support yet: https://github.com/keras-team/keras-contrib/issues/453
        model = Sequential()
        model.add(Bidirectional(layer=LSTM(units=1024, return_sequences=True),
                                input_shape=(max_len, dim_embedding_vec)))
        model.add(TimeDistributed(Dense(50, activation="relu")))  # a dense layer as suggested by neuralNer
        crf = CRF(num_categories)
        model.add(crf)
        model.compile(optimizer="rmsprop",
                      loss=crf_loss,
                      metrics=[crf.accuracy])
        model.summary()
    else: # BiLSTM
        model = Sequential()
        model.add(Bidirectional(layer=LSTM(units=1024, return_sequences=True), input_shape=(max_len, dim_embedding_vec)))
        model.add(Dense(num_categories))
        model.add(Activation('softmax'))
        model = to_multi_gpu(model, n_gpus=_NUM_GPUS)
        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(0.001),
                      metrics=['accuracy'])
        model.summary()

    # get y integer labels
    y = to_categorical(trainY_int, num_classes=num_categories)
    # train model
    model.fit(trainX_embeddings, y, batch_size=1024, epochs=10)
    # save the model to disk
    model.save("model.h5")

# predict
y_pred = model.predict(testX_embeddings)

# compute model accuracy
total = 0
count = 0
pred_itr = y_pred.__iter__()
for sentence in testY_int:
    total += len(sentence)
    pred_sentence = pred_itr.__next__()
    pred_sentence = pred_sentence.__iter__()
    for word in [int2word[x] for x in sentence]:
        pred_vec = pred_sentence.__next__()
        pred_word = int2word[np.argmax(pred_vec)]
        if word == "":
            total -= 1  # padding
        else:
            count += 1 if word == pred_word else 0
# report the model accuracy
print("accuracy: {}".format(count / total))
