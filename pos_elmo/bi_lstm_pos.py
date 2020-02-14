"""


    Improvement ideas:
        - k-fold cross validation: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html

    File placement:
        - add penn treebank wsj files into ~/nltk_data/corpora/treebank/combined (extracted from 00-24 structure)

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

import nltk
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = ""
from keras.models import Sequential
from keras.layers import Dense, LSTM, Bidirectional, Activation
from keras.optimizers import Adam
import numpy as np
from datasets.ptb.penn_treebank import PTB

#################################
# Import data
#################################

# define train/dev/test split
# train: 0..18 -> range(0, 19)
train_files = [item for i in range(0, 19) for item in os.listdir(os.path.join("data/wsj/{:02}".format(i)))]
# dev: 19..21 -> range(19, 22)
dev_files = [item for i in range(19, 22) for item in os.listdir(os.path.join("data/wsj/{:02}".format(i)))]
# test: 22..24 -> range(22, 25)
test_files = [item for i in range(22, 25) for item in os.listdir(os.path.join("data/wsj/{:02}".format(i)))]

# get train/dev/test data and split it into X and y
# train
trainX, trainY = [], []
for tagged_sentence in nltk.corpus.treebank.tagged_sents(train_files):
    sentence, tags = zip(*tagged_sentence)
    trainX.append(np.array(sentence))
    trainY.append(np.array(tags))
# dev
devX, devY = [], []
for tagged_sentence in nltk.corpus.treebank.tagged_sents(dev_files):
    sentence, tags = zip(*tagged_sentence)
    devX.append(np.array(sentence))
    devY.append(np.array(tags))
# test
testX, testY = [], []
for tagged_sentence in nltk.corpus.treebank.tagged_sents(test_files):
    sentence, tags = zip(*tagged_sentence)
    testX.append(np.array(sentence))
    testY.append(np.array(tags))


#################################
# Pad sequences
#################################

token_train_len = [len(x) for x in trainX]
token_dev_len = [len(x) for x in devX]
token_test_len = [len(x) for x in testX]

max_len = max(token_train_len + token_dev_len + token_test_len)

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

"""
dim_embedding_vec = 1024

elmo = hub.Module("https://tfhub.dev/google/elmo/3", trainable=True)

trainX_embeddings = elmo(
    inputs={
        "tokens": trainX,
        "sequence_len": [max_len]*len(trainX)  # token_train_len
    },
    signature="tokens",
    as_dict=True)["elmo"]

devX_embeddings = elmo(
    inputs={
        "tokens": devX,
        "sequence_len": [max_len]*len(devX)  # token_dev_len
    },
    signature="tokens",
    as_dict=True)["elmo"]

testX_embeddings = elmo(
    inputs={
        "tokens": testX,
        "sequence_len": [max_len]*len(testX)  # token_test_len
    },
    signature="tokens",
    as_dict=True)["elmo"]
"""
"""
from allennlp.modules.elmo import Elmo, batch_to_ids
import numpy as np


options_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

# Compute two different representation for each token.
# Each representation is a linear weighted combination for the
# 3 layers in ELMo (i.e., charcnn, the outputs of the two BiLSTM))
elmo = Elmo(options_file, weight_file, 2, dropout=0)

# use batch_to_ids to convert sentences to character ids
sentences = [['First', 'sentence', '.'], ['Another', '.']]
character_ids = batch_to_ids(sentences)

embeddings = elmo(character_ids)["elmo_representations"]
"""

from embeddings.glove import GloVe
_DATA_SOURCE = "common_crawl_840b_cased"
_DATA_SET = "glove.840B.300d.txt"
# instantiate preprocessor
preprocessor = GloVe()
# prepare pre-trained data
preprocessor.import_pre_trained_data(_DATA_SOURCE, _DATA_SET)


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

labels = set()
for item in trainY + devY + testY:
    for label in item:
        labels.add(label)

num_categories = len(labels)  # number of pos tags

word2int = {label: i for i, label in enumerate(list(labels))}

# trainY = [word2int[word] for word in sentence for sentence in trainY]
trainY_int = list()
for sentence in trainY:
    trainY_int.append([word2int[word] for word in sentence])
testY_int = list()
for sentence in testY:
    testY_int.append([word2int[word] for word in sentence])

#################################
# Define BiLSTM model
#################################

model = Sequential()
model.add(Bidirectional(layer=LSTM(units=1024, return_sequences=True), input_shape=(max_len, dim_embedding_vec)))
model.add(Dense(num_categories))
model.add(Activation('softmax'))

from multi_gpu import to_multi_gpu
model = to_multi_gpu(model, n_gpus=2)

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(0.001),
              metrics=['accuracy'])
model.summary()

from keras.utils import to_categorical
y = to_categorical(trainY_int, num_classes=num_categories)

model.fit(trainX_embeddings, y, batch_size=1024, epochs=40)  # batch_size=2 steps_per_epoch=128, epochs=40


scores = model.evaluate(testX_embeddings, to_categorical(testY_int, num_classes=num_categories))
print(f"{model.metrics_names[1]}: {scores[1] * 100}")  # acc: 99.09751977804825
print()



