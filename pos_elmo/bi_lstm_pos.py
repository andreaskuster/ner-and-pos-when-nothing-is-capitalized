"""


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

from keras.models import Sequential
from keras.layers import Dense, LSTM, Bidirectional, Activation
from keras.optimizers import Adam
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from keras_contrib.layers import CRF
import numpy as np
from datasets.ptb.penn_treebank import PTB
from embeddings.word2vec import Word2Vec
from embeddings.elmo import ELMoV2
from embeddings.glove import GloVe
from helper.multi_gpu import to_multi_gpu

# show only relevant cuda gpu devices (i.e. mask some for parallel jobs)
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

_DATASET = "dummy"
_EMBEDDINGS = "elmo"
_DATA_SOURCE_WORD2VEC = "word2vec-google-news-300"
_DATA_SOURCE_GLOVE = "common_crawl_840b_cased"
_DATA_SET_GLOVE = "glove.840B.300d.txt"
_NUM_GPUS = 4


def create_joined_crf_loss(crf):
    def loss(y_true, y_pred):
        offset = 0
        
        X = crf.input
        mask = crf.input_mask
        nloglik = crf.get_negative_log_likelihood(y_true, X, mask)
        return nloglik

    return loss


from keras_contrib.metrics.crf_accuracies import _get_accuracy
def create_joined_crf_accuracy(crf):
    def accuracy(y_true, y_pred):        
        X = crf.input
        mask = crf.input_mask
        y_pred = crf.viterbi_decoding(X, mask)
        return _get_accuracy(y_true, y_pred, mask, crf.sparse_target)

    return accuracy

#################################
# Import data
#################################
print("Importing Penn Treebank data...")

# instantiate Penn TreeBank data loader
ptb = PTB()

if _DATASET == "dummy":
    # load two sentence dummy dataset
    x, y = ptb.load_data([0])
    trainX, trainY = x[0:2], y[0:2]
    devX, devY = x[2:4], y[2:4]
    testX, testY = x[4:6], y[4:6]

else:
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
    preprocessor = ELMoV2()

elif _EMBEDDINGS == "glove":
    # instantiate preprocessor
    preprocessor = GloVe()
    # prepare pre-trained data
    preprocessor.import_pre_trained_data(_DATA_SOURCE_GLOVE, _DATA_SET_GLOVE)

else:
    raise RuntimeError("No embeddings preprocessor selected.")


if _EMBEDDINGS == "elmo":

    trainX_embeddings = preprocessor.embedding(trainX)

    testX_embeddings = preprocessor.embedding(testX).eval()
else:

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
# Define BiLSTM-CRF model
#################################
"""
# BiLSTM model:

model = Sequential()
model.add(Bidirectional(layer=LSTM(units=1024, return_sequences=True), input_shape=(max_len, dim_embedding_vec)))
model.add(Dense(num_categories))
model.add(Activation('softmax'))
model = to_multi_gpu(model, n_gpus=_NUM_GPUS)
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(0.001),
              metrics=['accuracy'])
model.summary()
"""

# multi gpu: https://github.com/keras-team/keras-contrib/issues/453
model = Sequential()
model.add(Bidirectional(layer=LSTM(units=1024, return_sequences=True), input_shape=(max_len, dim_embedding_vec)))
model.add(TimeDistributed(Dense(50, activation="relu")))  # a dense layer as suggested by neuralNer

crf = CRF(10, sparse_target=True, learn_mode="join")
model.add(crf)

#model = to_multi_gpu(model, n_gpus=_NUM_GPUS)

from keras_contrib.metrics import crf_viterbi_accuracy
model.compile(optimizer="rmsprop", loss=crf.loss_function, metrics=[crf.accuracy])

model.summary()


from keras.utils import to_categorical
y = to_categorical(trainY_int, num_classes=num_categories)

model.fit(trainX_embeddings, y, batch_size=1024, epochs=10)  # batch_size=2 steps_per_epoch=128, epochs=40


scores = model.evaluate(testX_embeddings, to_categorical(testY_int, num_classes=num_categories))
print(f"{model.metrics_names[1]}: {scores[1] * 100}")  # acc: 99.09751977804825
print()



