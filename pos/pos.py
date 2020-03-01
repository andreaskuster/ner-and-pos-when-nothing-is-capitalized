import os
import sys

import numpy as np

from enum import Enum, auto
from argparse import ArgumentParser
from typing import Tuple
from os.path import isfile

from sklearn.model_selection import train_test_split
from keras import Sequential
from keras.layers import LSTM, Dense, Activation, Bidirectional, TimeDistributed
from keras.optimizers import Adam
from keras.models import model_from_json
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical
from keras_contrib.layers import CRF

from datasets.ptb.penn_treebank import PTB
from datasets.brown.brown import Brown
from datasets.conll2000.conll2000 import CoNLL2000

from embeddings.word2vec import Word2Vec
from embeddings.glove import GloVe
from embeddings.elmov2 import ELMo

from helper.multi_gpu import to_multi_gpu


class LogLevel(Enum):
    NO: int = 0
    LIMITED: int = 1
    FULL: int = 2


class Dataset(Enum):
    PTB = auto()
    PTB_DUMMY = auto()
    PTB_REDUCED = auto()
    BROWN = auto()
    CONLL2000 = auto()


class CaseType(Enum):
    CASED = auto()
    UNCASED = auto()
    TRUECASE = auto()
    CASED_UNCASED = auto()
    HALF_MIXED = auto()


class Embedding(Enum):
    GLOVE = auto()
    WORD2VEC = auto()
    ELMO = auto()


class DataSourceWord2Vec(Enum):
    GOOGLE_NEWS_300: str = "word2vec-google-news-300"


class DataSourceGlove(Enum):
    COMMON_CRAWL_840B_CASED_300D: Tuple[str, str] = ("common_crawl_840b_cased", "glove.840B.300d.txt")


class Model(Enum):
    BILSTM = "bilstm"
    BILSTM_CRF = "bilstmcrf"


class POS:

    def __init__(self,
                 log_level: LogLevel = LogLevel.LIMITED,
                 data_source_word2vec: DataSourceWord2Vec = DataSourceWord2Vec.GOOGLE_NEWS_300,
                 data_source_glove: DataSourceGlove = DataSourceGlove.COMMON_CRAWL_840B_CASED_300D,
                 batch_size_embedding: int = 4096):
        """
        TODO: add description
        :param log_level:
        :param data_source_word2vec:
        :param data_source_glove:
        :param batch_size_embedding:
        """
        # log level
        self.log_level: LogLevel = log_level  # 0: none, 1: limited, 2: full
        # dataset
        self.dataset = None
        # casetype
        self.train_casetype = None
        self.test_casetype = None
        self.dev_casetype = None
        # dataset
        self.train_x, self.train_y, self.test_x, self.test_y, self.dev_x, self.dev_y = None, None, None, None, None, None
        self.data_x: dict = None
        self.data_y: dict = None
        # embedded dataset
        self.train_x_embedded, self.test_x_embedded, self.dev_x_embedded = None, None, None
        self.dataset_x_embedded: dict = dict()
        # tag to index mapping
        self.train_y_int, self.test_y_int, self.dev_y_int = None, None, None
        self.dataset_y_int: dict = dict()
        self.labels: set = None
        self.num_categories: int = -1
        self.int2word: list = None
        self.word2int: dict = None
        # mapping
        self.dataset_map: dict = {
            "train_x": "train_x_embedded",
            "test_x": "test_x_embedded",
            "dev_x": "dev_x_embedded",
            "train_y": "train_y_int",
            "test_y": "test_y_int",
            "dev_y": "dev_y_int"
        }
        # padding
        self.max_sentence_length: int = -1
        # embedding
        self.batch_size_embedding: int = batch_size_embedding
        self.dim_embedding_vec: int = -1
        self.data_source_word2vec: str = data_source_word2vec.value
        self.data_source_glove, self.data_set_glove = data_source_glove.value
        # model
        self.model = None
        self.model_details: dict = None

    def set_cuda_visible_devices(self,
                                 devices: str = None):
        """
        Masks visible cuda devices (i.e. useful for parallel job execution).
        :param devices: string with comma separated device indices
        :return None
        """
        # report action
        if self.log_level.value >= LogLevel.LIMITED.value:
            print("Set visible cuda devices...")
        # skip upon None val
        if devices is not None:
            # set environment variable
            os.environ["CUDA_VISIBLE_DEVICES"] = devices

    def import_data(self,
                    dataset: Dataset,
                    train_casetype: CaseType,
                    test_casetype: CaseType,
                    dev_casetype: CaseType = None,
                    test_size: float = 0.1,
                    dev_size: float = 0.1):
        """
        TODO: add description
        :param dataset:
        :param train_casetype:
        :param test_casetype:
        :param test_size:
        :param dev_size:
        :return:
        """
        # report action
        if self.log_level.value >= LogLevel.LIMITED.value:
            print("Importing data...")
        if dev_casetype is None:
            dev_casetype = test_casetype
        # store parameters
        self.dataset = dataset
        self.train_casetype = train_casetype
        self.test_casetype = test_casetype
        self.dev_casetype = dev_casetype
        # load data
        train_x, train_y, test_x, test_y, dev_x, dev_y = None, None, None, None, None, None
        if dataset == Dataset.PTB_DUMMY:
            # instantiate data loader
            ptb = PTB()
            # split data: train: 4 sentences, dev & test: 2 sentences each with 1 overlapping sentences
            data_x, data_y = ptb.load_data([0])
            # training data
            if train_casetype == CaseType.CASED:
                data_x, data_y = ptb.load_data([0])
            elif train_casetype == CaseType.UNCASED:
                data_x, data_y = ptb.load_data_lowercase([0])
            elif train_casetype == CaseType.TRUECASE:
                data_x, data_y = ptb.load_data_truecase([0])
            elif train_casetype == CaseType.CASED_UNCASED:
                data_x, data_y = ptb.load_data_cased_and_uncased([0])
            elif train_casetype == CaseType.HALF_MIXED:
                data_x, data_y = ptb.load_data_half_mixed([0])
            train_x, train_y = data_x[1:5], data_y[1:5]
            # test data
            if test_casetype == CaseType.CASED:
                data_x, data_y = ptb.load_data([0])
            elif test_casetype == CaseType.UNCASED:
                data_x, data_y = ptb.load_data_lowercase([0])
            elif test_casetype == CaseType.TRUECASE:
                data_x, data_y = ptb.load_data_truecase([0])
            elif test_casetype == CaseType.CASED_UNCASED:
                data_x, data_y = ptb.load_data_cased_and_uncased([0])
            elif test_casetype == CaseType.HALF_MIXED:
                data_x, data_y = ptb.load_data_half_mixed([0])
            test_x, test_y = data_x[4:6], data_y[4:6]
            # dev data
            if dev_casetype == CaseType.CASED:
                data_x, data_y = ptb.load_data([0])
            elif dev_casetype == CaseType.UNCASED:
                data_x, data_y = ptb.load_data_lowercase([0])
            elif dev_casetype == CaseType.TRUECASE:
                data_x, data_y = ptb.load_data_truecase([0])
            elif dev_casetype == CaseType.CASED_UNCASED:
                data_x, data_y = ptb.load_data_cased_and_uncased([0])
            elif dev_casetype == CaseType.HALF_MIXED:
                data_x, data_y = ptb.load_data_half_mixed([0])
            dev_x, dev_y = data_x[0:2], data_y[0:2]  # always use cased

        elif dataset == Dataset.PTB or dataset == Dataset.PTB_REDUCED:
            # instantiate data loader
            ptb = PTB()
            # train sections: 0..18
            train_sections = range(0, 19) if dataset == Dataset.PTB else range(0, 5)
            # dev sections: 19..21
            dev_sections = range(19, 22) if dataset == Dataset.PTB else range(5, 7)
            # test sections: 22..24
            test_sections = range(22, 25) if dataset == Dataset.PTB else range(7, 9)
            # load data

            # training data
            if train_casetype == CaseType.CASED:
                train_x, train_y = ptb.load_data(train_sections)
            elif train_casetype == CaseType.UNCASED:
                train_x, train_y = ptb.load_data_lowercase(train_sections)
            elif train_casetype == CaseType.TRUECASE:
                train_x, train_y = ptb.load_data_truecase(train_sections)
            elif train_casetype == CaseType.CASED_UNCASED:
                train_x, train_y = ptb.load_data_cased_and_uncased(train_sections)
            elif train_casetype == CaseType.HALF_MIXED:
                train_x, train_y = ptb.load_data_half_mixed(train_sections)

            # test data
            if test_casetype == CaseType.CASED:
                test_x, test_y = ptb.load_data(test_sections)
            elif test_casetype == CaseType.UNCASED:
                test_x, test_y = ptb.load_data_lowercase(test_sections)
            elif test_casetype == CaseType.TRUECASE:
                test_x, test_y = ptb.load_data_truecase(test_sections)
            elif test_casetype == CaseType.CASED_UNCASED:
                test_x, test_y = ptb.load_data_cased_and_uncased(test_sections)
            elif test_casetype == CaseType.HALF_MIXED:
                test_x, test_y = ptb.load_data_half_mixed(test_sections)

            # dev data
            if dev_casetype == CaseType.CASED:
                dev_x, dev_y = ptb.load_data(dev_sections)
            elif dev_casetype == CaseType.UNCASED:
                dev_x, dev_y = ptb.load_data_lowercase(dev_sections)
            elif dev_casetype == CaseType.TRUECASE:
                dev_x, dev_y = ptb.load_data_truecase(dev_sections)
            elif dev_casetype == CaseType.CASED_UNCASED:
                dev_x, dev_y = ptb.load_data_cased_and_uncased(dev_sections)
            elif dev_casetype == CaseType.HALF_MIXED:
                dev_x, dev_y = ptb.load_data_half_mixed(dev_sections)

        elif dataset == Dataset.BROWN or dataset == Dataset.CONLL2000:
            # instantiate data loader
            data_loader = Brown() if dataset == Dataset.BROWN else CoNLL2000()

            # training data
            if train_casetype == CaseType.CASED:
                train_x, train_y = data_loader.load_data()
            elif train_casetype == CaseType.UNCASED:
                train_x, train_y = data_loader.load_data_lowercase()
            elif train_casetype == CaseType.TRUECASE:
                train_x, train_y = data_loader.load_data_truecase()
            elif train_casetype == CaseType.CASED_UNCASED:
                train_x, train_y = data_loader.load_data_cased_and_uncased()
            elif train_casetype == CaseType.HALF_MIXED:
                train_x, train_y = data_loader.load_data_half_mixed()
            # test data
            if test_casetype == CaseType.CASED:
                test_x, test_y = data_loader.load_data()
            elif test_casetype == CaseType.UNCASED:
                test_x, test_y = data_loader.load_data_lowercase()
            elif test_casetype == CaseType.TRUECASE:
                test_x, test_y = data_loader.load_data_truecase()
            elif test_casetype == CaseType.CASED_UNCASED:
                test_x, test_y = data_loader.load_data_cased_and_uncased()
            elif test_casetype == CaseType.HALF_MIXED:
                test_x, test_y = data_loader.load_data_half_mixed()
            # dev data
            if dev_casetype == CaseType.CASED:
                dev_x, dev_y = data_loader.load_data()
            elif dev_casetype == CaseType.UNCASED:
                dev_x, dev_y = data_loader.load_data_lowercase()
            elif dev_casetype == CaseType.TRUECASE:
                dev_x, dev_y = data_loader.load_data_truecase()
            elif dev_casetype == CaseType.CASED_UNCASED:
                dev_x, dev_y = data_loader.load_data_cased_and_uncased()
            elif dev_casetype == CaseType.HALF_MIXED:
                dev_x, dev_y = data_loader.load_data_half_mixed()
            # compute train/test/dev dataset size
            total_size = len(train_x)
            start, middle0, middle1, end = 0, int(test_size*total_size),  int((test_size + dev_size)*total_size), total_size
            # split data
            train_x, train_y = train_x[middle1:end], train_y[middle1:end]
            test_x, test_y = test_x[start:middle0], test_y[start:middle0]
            dev_x, dev_y = dev_x[middle0:middle1], dev_y[middle0:middle1]
        else:
            raise RuntimeError("Unknown dataset.")
        # store dataset internally
        self.train_x, self.train_y, self.test_x, self.test_y, self.dev_x, self.dev_y = train_x, train_y, test_x, test_y, dev_x, dev_y
        self.data_x = {
            "train_x": self.train_x,
            "test_x": self.test_x,
            "dev_x": self.dev_x
        }
        self.data_y = {
            "train_y": self.train_y,
            "test_y": self.test_y,
            "dev_y": self.dev_y
        }
        # return dataset
        return train_x, train_y, test_x, test_y, dev_x, dev_y

    def pad_sequence(self):
        """
        TODO: add description
        :return:
        """
        # report action
        if self.log_level.value >= LogLevel.LIMITED.value:
            print("Padding sequence...")
        # update maximum sentence length
        for dataset in self.data_x:
            self.max_sentence_length = max(self.max_sentence_length,  max([len(x) for x in self.data_x[dataset]]))
        # pad x
        for dataset in self.data_x:
            for i in range(len(self.data_x[dataset])):
                no_pad = self.max_sentence_length - len(self.data_x[dataset][i])
                self.data_x[dataset][i] = list(self.data_x[dataset][i]) + ([""] * no_pad)
        # pad y
        for dataset in self.data_y:
            for i in range(len(self.data_y[dataset])):
                no_pad = self.max_sentence_length - len(self.data_y[dataset][i])
                self.data_y[dataset][i] = list(self.data_y[dataset][i]) + ([""] * no_pad)

    def embedding(self, embedding: Embedding):
        """
        TODO: add description
        :param embedding:
        :return:
        """
        # report action
        if self.log_level.value >= LogLevel.LIMITED.value:
            print("Embedding data...")
        if embedding == Embedding.WORD2VEC:
            # instantiate preprocessor
            preprocessor = Word2Vec()
            # download pre-trained data
            preprocessor.import_pre_trained_data(self.data_source_word2vec)
        elif embedding == Embedding.ELMO:
            # instantiate preprocessor
            preprocessor = ELMo()
        elif embedding == Embedding.GLOVE:
            # instantiate preprocessor
            preprocessor = GloVe()
            # prepare pre-trained data
            preprocessor.import_pre_trained_data(self.data_source_glove, self.data_set_glove)
        else:
            raise RuntimeError("Unknown embedding.")

        # set embedding vector length
        self.dim_embedding_vec = preprocessor.dim

        if embedding == Embedding.ELMO:
            # try to load embeddings (elmo uses a lot of CPU/RAM resources, therefore, we save them to disk for re-use)
            path_data_x = "elmo_embedding_{}_train_{}_test_{}_dev_{}_data_x.npz".format(self.dataset.name,
                                                                                        self.train_casetype.name,
                                                                                        self.test_casetype.name,
                                                                                        self.dev_casetype.name)
            if isfile(path_data_x):
                with np.load(path_data_x) as data:
                    self.dataset_x_embedded["train_x_embedded"], self.train_x_embedded = data["train_x_embedded"], data[
                        "train_x_embedded"]
                    self.dataset_x_embedded["test_x_embedded"], self.test_x_embedded = data["test_x_embedded"], data[
                        "test_x_embedded"]
                    self.dataset_x_embedded["dev_x_embedded"], self.dev_x_embedded = data["dev_x_embedded"], data[
                        "dev_x_embedded"]
            else:
                # run batch-wise embedding (it uses too much memory otherwise)
                if self.log_level.value >= LogLevel.LIMITED.value:
                    print("Start elmo embedding...")

                for dataset in self.data_x:
                    num_batches = int(len(self.data_x[dataset]) / self.batch_size_embedding)

                    if num_batches > 0:
                        start, end = 0, 0
                        data_x_embedding = list()
                        for i in range(num_batches):
                            # compute indices
                            start = i * self.batch_size_embedding
                            end = (i + 1) * self.batch_size_embedding
                            # do batch embedding
                            data_x_embedding.append(preprocessor.embedding(self.data_x[dataset][start:end]))
                            # report action
                            if self.log_level.value >= LogLevel.FULL.value:
                                print("Elmo embedding round {}...".format(i))
                        data_x_embedding = np.array(data_x_embedding)
                        data_x_embedding = data_x_embedding.reshape(-1, data_x_embedding[0].shape[-2],
                                                                    data_x_embedding[0].shape[-1])
                        if end < len(self.data_x[dataset]):
                            data_x_embedding = np.concatenate(
                                (data_x_embedding, preprocessor.embedding(self.data_x[dataset][end:])), axis=0)
                            # report action
                            if self.log_level.value >= LogLevel.FULL.value:
                                print("Elmo embedding round remainder...")
                    else:
                        data_x_embedding = preprocessor.embedding(self.data_x[dataset])
                    # store data internally
                    self.dataset_x_embedded[self.dataset_map[dataset]] = data_x_embedding
                self.train_x_embedded, self.test_x_embedded, self.dev_x_embedded = self.dataset_x_embedded["train_x_embedded"], self.dataset_x_embedded["test_x_embedded"], self.dataset_x_embedded["dev_x_embedded"]

                # store data to file
                path_data_x = "elmo_embedding_{}_train_{}_test_{}_dev_{}_data_x.npz".format(self.dataset.name,
                                                                                     self.train_casetype.name,
                                                                                     self.test_casetype.name,
                                                                                    self.dev_casetype.name)
                np.savez(path_data_x,
                         train_x_embedded=self.dataset_x_embedded["train_x_embedded"],
                         test_x_embedded=self.dataset_x_embedded["test_x_embedded"],
                         dev_x_embedded=self.dataset_x_embedded["dev_x_embedded"])

        else:  # glove and word2vec are sequence independent -> process one-by-one
            for dataset in self.data_x:
                data_x_embedded = list()
                for sentence in self.data_x[dataset]:
                    data_x_embedded.append([preprocessor.word2vec(word) for word in sentence])
                self.dataset_x_embedded[self.dataset_map[dataset]] = np.array(data_x_embedded)
            self.train_x_embedded, self.test_x_embedded, self.dev_x_embedded = self.dataset_x_embedded["train_x_embedded"], self.dataset_x_embedded["test_x_embedded"], self.dataset_x_embedded["dev_x_embedded"]

    def map_y(self):
        """
        TODO: add description
        :return:
        """
        # report action
        if self.log_level.value >= LogLevel.LIMITED.value:
            print("Add label to int mapping...")
        self.labels: set = set()
        for item in self.train_y + self.test_y + self.dev_y:
            for label in item:
                self.labels.add(label)

        self.num_categories = len(self.labels)  # number of pos tags
        self.int2word = list(self.labels)
        self.word2int = {label: i for i, label in enumerate(self.int2word)}

        for dataset in self.data_y:
            data_y_int = list()
            for sentence in self.data_y[dataset]:
                data_y_int.append([self.word2int[word] for word in sentence])
            # store data internally
            self.dataset_y_int[self.dataset_map[dataset]] = data_y_int
        self.train_y_int, self.test_y_int, self.dev_y_int = self.dataset_y_int["train_y_int"], self.dataset_y_int["test_y_int"], self.dataset_y_int["dev_y_int"]

    def model_name(self) -> str:
        """
        TODO: add description
        :return:
        """
        if self.model_details is None:
            return ""
        return "{}_{}_units_{}_dropout_{}_recdropout_{}_lr_{}_dataset_{}_train_{}_test_{}_dev".format(self.model_details["model"].name,
                                                               self.model_details["lstm_hidden_units"],
                                                               self.model_details["lstm_dropout"],
                                                               self.model_details["lstm_recurrent_dropout"],
                                                               self.model_details["learning_rate"],
                                                               self.dataset,
                                                               self.train_casetype.name,
                                                               self.test_casetype.name,
                                                               self.dev_casetype.name)

    def create_model_bilstm(self,
                            lstm_hidden_units=1024,
                            lstm_dropout=0.1,
                            lstm_recurrent_dropout=0.1,
                            num_gpus=1,
                            learning_rate=1e-3):
        """
        TODO: add description
        :param lstm_hidden_units:
        :param lstm_dropout:
        :param lstm_recurrent_dropout:
        :param num_gpus:
        :param learning_rate:
        :return:
        """
        # create model
        self.model = Sequential()
        self.model.add(Bidirectional(layer=LSTM(units=lstm_hidden_units,
                                                return_sequences=True,
                                                dropout=lstm_dropout,
                                                recurrent_dropout=lstm_recurrent_dropout),
                                     input_shape=(self.max_sentence_length, self.dim_embedding_vec)))
        self.model.add(Dense(self.num_categories)) # self.model.add(TimeDistributed(Dense(self.num_categories, activation="relu")))
        self.model.add(Activation("softmax"))
        self.model = to_multi_gpu(self.model, n_gpus=num_gpus)
        self.model.compile(loss="categorical_crossentropy",
                           optimizer=Adam(learning_rate),
                           metrics=["accuracy"])
        if self.log_level.value >= LogLevel.LIMITED.value:
            self.model.summary()

    def create_model_bilstm_crf(self,
                            lstm_hidden_units=1024,
                            lstm_dropout=0.1,
                            lstm_recurrent_dropout=0.1,
                            num_gpus=1,
                            learning_rate=1e-3):
        """
        TODO: add description
        :param lstm_hidden_units:
        :param lstm_dropout:
        :param lstm_recurrent_dropout:
        :param num_gpus:
        :param learning_rate:
        :return:
        """
        # create model
        self.model = Sequential()
        self.model.add(Bidirectional(layer=LSTM(units=lstm_hidden_units,
                                                return_sequences=True,
                                                dropout=lstm_dropout,
                                                recurrent_dropout=lstm_recurrent_dropout),
                                                input_shape=(self.max_sentence_length, self.dim_embedding_vec)))
        self.model.add(TimeDistributed(Dense(self.num_categories, activation="relu")))
        crf = CRF(self.num_categories)
        self.model.add(crf)
        # self.model = to_multi_gpu(self.model, n_gpus=num_gpus)
        self.model.compile(optimizer=Adam(learning_rate), loss=crf.loss_function, metrics=[crf.accuracy]) # rmsprop

        if self.log_level.value >= LogLevel.LIMITED.value:
            self.model.summary()

    def save_model(self):
        """
        TODO: add description
        :return:
        """
        if self.log_level.value >= LogLevel.LIMITED.value:
            print("Save model to file...")
        # serialize model
        model_json = self.model.to_json()
        # save model
        with open("{}.json".format(self.model_name()), "w") as json_file:
            json_file.write(model_json)
        # save weights
        self.model.save_weights("{}.h5".format(self.model_name()))

    def try_load_model(self) -> bool:
        """
        TODO: add description
        :return:
        """
        return False  # save/load doesn't match (https://github.com/keras-team/keras/issues/4875) -> temporary disabled
        if isfile("{}.json".format(self.model_name())) and isfile("{}.h5".format(self.model_name())):
            if self.log_level.value >= LogLevel.LIMITED.value:
                print("Loading model from file...")
            # load model
            json_file = open("{}.json".format(self.model_name()), "r")
            loaded_model_json = json_file.read()
            json_file.close()
            self.model = model_from_json(loaded_model_json, custom_objects={"CRF": CRF})
            # load pre-trained weights
            self.model.load_weights("{}.h5".format(self.model_name()))
            # compile model
            if self.model_details["model"] == Model.BILSTM:
                self.model.compile(loss="categorical_crossentropy",
                                   optimizer=Adam(self.model_details["learning_rate"]),
                                   metrics=["accuracy"])
            else:
                crf = CRF(self.num_categories)
                self.model.compile(optimizer="rmsprop", loss=crf.loss_function, metrics=[crf.accuracy])

            return True
        return False

    def train_model(self,
                    batch_size=1024,
                    epochs=40):
        """
        TODO: add description
        :param batch_size:
        :param epochs:
        :return:
        """
        if self.model_details["model"] == Model.BILSTM_CRF:
            es = EarlyStopping(monitor="crf_viterbi_accuracy", mode="max", verbose=1, patience=4, min_delta=1e-3)
        else:
            es = EarlyStopping(monitor="val_accuracy", mode="max", verbose=1, patience=4, min_delta=1e-3)
        # mc = ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
        # fit model
        history = self.model.fit(self.train_x_embedded, to_categorical(self.train_y_int, self.num_categories),
                                 batch_size=batch_size,
                                 epochs=epochs,
                                 validation_data=(self.dev_x_embedded, to_categorical(self.dev_y_int, self.num_categories)),
                                 callbacks=[es])
        # store history to file
        np.savetxt(self.model_name() + "_history_epoch", history.epoch)
        np.savetxt(self.model_name() + "_history_loss", history.history["loss"])
        np.savetxt(self.model_name() + "_history_val_loss", history.history["val_loss"])

        if self.model_details["model"] == Model.BILSTM_CRF:
            np.savetxt(self.model_name() + "_history_accuracy", history.history["crf_viterbi_accuracy"])
            np.savetxt(self.model_name() + "_history_val_accuracy", history.history["val_crf_viterbi_accuracy"])
        else:
            np.savetxt(self.model_name() + "_history_val_accuracy", history.history["val_accuracy"])
            np.savetxt(self.model_name() + "_history_accuracy", history.history["accuracy"])

    def model_accuracy(self, X_embedded, y_int, dataset:str):
        """
        TODO: add description
        :return:
        """
        y_pred = self.model.predict(X_embedded)

        total = 0
        count = 0
        pred_itr = y_pred.__iter__()
        for sentence in y_int:
            total += len(sentence)
            pred_sentence = pred_itr.__next__()
            pred_sentence = pred_sentence.__iter__()
            for word in [self.int2word[x] for x in sentence]:
                pred_vec = pred_sentence.__next__()
                pred_word = self.int2word[np.argmax(pred_vec)]
                if word == "":
                    total -= 1  # padding
                else:
                    count += 1 if word == pred_word else 0
        accuracy = count / total
        print("{} accuracy: {}".format(dataset, accuracy))
        np.savetxt(self.model_name() + "_{}_accuracy".format(dataset), [accuracy])
        return accuracy

    def set_model_params(self,
                            model=Model.BILSTM_CRF,
                            lstm_hidden_units=1024,
                            lstm_dropout=0.1,
                            lstm_recurrent_dropout=0.1,
                            num_gpus=1,
                            learning_rate=1e-3):
        """
        TODO: add description
        :param model:
        :param lstm_hidden_units:
        :param lstm_dropout:
        :param lstm_recurrent_dropout:
        :param num_gpus:
        :param learning_rate:
        :return:
        """
        # set model details
        self.model_details = dict()
        self.model_details["model"] = model
        self.model_details["lstm_hidden_units"] = lstm_hidden_units
        self.model_details["lstm_dropout"] = lstm_dropout
        self.model_details["lstm_recurrent_dropout"] = lstm_recurrent_dropout
        self.model_details["num_gpus"] = num_gpus
        self.model_details["learning_rate"] = learning_rate

    def create_model(self):
        """
        TODO: add description
        :return:
        """
        if self.model_details["model"] == Model.BILSTM:
            self.create_model_bilstm(
                lstm_hidden_units=self.model_details["lstm_hidden_units"],
                lstm_dropout=self.model_details["lstm_dropout"],
                lstm_recurrent_dropout=self.model_details["lstm_recurrent_dropout"],
                num_gpus=self.model_details["num_gpus"],
                learning_rate=self.model_details["learning_rate"])
        elif self.model_details["model"] == Model.BILSTM_CRF:
            self.create_model_bilstm_crf(
                lstm_hidden_units = self.model_details["lstm_hidden_units"],
                lstm_dropout = self.model_details["lstm_dropout"],
                lstm_recurrent_dropout = self.model_details["lstm_recurrent_dropout"],
                num_gpus = self.model_details["num_gpus"],
                learning_rate = self.model_details["learning_rate"])
        else:
            raise RuntimeError("Undefined model.")


if __name__ == "__main__":
    """
    TODO: add description
    """
    # parse arguments
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument("-d", "--dataset", default=Dataset.PTB_DUMMY.name, choices=[x.name for x in Dataset])
    parser.add_argument("-ctr", "--traincasetype", default=CaseType.CASED.name, choices=[x.name for x in CaseType])
    parser.add_argument("-cte", "--testcasetype", default=CaseType.CASED.name, choices=[x.name for x in CaseType])
    parser.add_argument("-cde", "--devcasetype", default=CaseType.CASED.name, choices=[x.name for x in CaseType])

    parser.add_argument("-v", "--loglevel", default=LogLevel.FULL.name, choices=[x.name for x in LogLevel])
    parser.add_argument("-e", "--embedding", default=Embedding.ELMO.name, choices=[x.name for x in Embedding])
    parser.add_argument("-w", "--datasource_word2vec", default=DataSourceWord2Vec.GOOGLE_NEWS_300.name,
                        choices=[x.name for x in DataSourceWord2Vec])
    parser.add_argument("-g", "--datasource_glove", default=DataSourceGlove.COMMON_CRAWL_840B_CASED_300D.name,
                        choices=[x.name for x in DataSourceGlove])
    parser.add_argument("-b", "--batchsize", default="1024")
    parser.add_argument("-p", "--epochs", default="40")
    parser.add_argument("-m", "--model", default=Model.BILSTM_CRF.name, choices=[x.name for x in Model])
    parser.add_argument("-ng", "--numgpus", default="1")
    parser.add_argument("-lr", "--learningrate", default="1e-3")
    parser.add_argument("-hu", "--lstmhiddenunits", default="1024")
    parser.add_argument("-dr", "--lstmdropout", default="1e-1")
    parser.add_argument("-rdr", "--lstmrecdropout", default="1e-1")
    parser.add_argument("-s", "--hyperparamsearch", default="False", choices=["True", "False"])
    parser.add_argument("-c", "--cudadevices", default="None")


    args = parser.parse_args()

    # convert args to correct data types
    dataset: Dataset = Dataset[args.dataset]
    train_casetype: CaseType = CaseType[args.traincasetype]
    test_casetype: CaseType = CaseType[args.testcasetype]
    dev_casetype: CaseType = CaseType[args.devcasetype]

    log_level: LogLevel = LogLevel[args.loglevel]
    embedding: Embedding = Embedding[args.embedding]
    datasource_word2vec: DataSourceWord2Vec = DataSourceWord2Vec[args.datasource_word2vec]
    datasource_glove: DataSourceGlove = DataSourceGlove[args.datasource_glove]
    batch_size: int = int(args.batchsize)
    epochs: int = int(args.epochs)
    model: Model = Model[args.model]
    learning_rate: float = float(args.learningrate)
    num_gpus: int = int(args.numgpus)
    lstm_hidden_units: int = int(args.lstmhiddenunits)
    lstm_dropout: float = float(args.lstmdropout)
    lstm_recurrent_dropout: float = float(args.lstmrecdropout)
    hyperparameter_search: bool = args.hyperparamsearch == "True"
    cuda_devices: str = None if args.cudadevices == "None" else args.cudadevices

    if log_level.value >= LogLevel.LIMITED.value:
        print("Dataset is: {}".format(dataset.name))
        print("Casetype training data is: {}".format(train_casetype.name))
        print("Casetype test data is: {}".format(test_casetype.name))
        print("Casetype dev data is: {}".format(dev_casetype.name))

        print("Log level is: {}".format(log_level.name))
        print("Embedding is: {}".format(embedding.name))
        if embedding == Embedding.WORD2VEC:
            print("Data source word2vec is: {}".format(datasource_word2vec.name))
        if embedding == Embedding.GLOVE:
            print("Data source glove is: {}".format(datasource_glove.name))
        print("Model name is: {}".format(model.name))
        print("Batch size is: {}".format(batch_size))
        print("Max number of epochs is: {}".format(epochs))
        print("Learning rate is: {}".format(learning_rate))
        print("LSTM hidden units: {}".format(lstm_hidden_units))
        print("LSTM dropout is: {}".format(lstm_dropout))
        print("LSTM recurrent dropout is: {}".format(lstm_recurrent_dropout))
        print("Hyperparameter search is: {}".format(hyperparameter_search))
        print("Visible cuda devies: {}".format(cuda_devices))
        print("Number of GPUs is: {}".format(num_gpus))

    pos = POS(log_level=log_level,
              data_source_word2vec=datasource_word2vec,
              data_source_glove=datasource_glove)
    pos.set_cuda_visible_devices(devices=cuda_devices)
    pos.import_data(dataset=dataset,
                    train_casetype=train_casetype,
                    test_casetype=test_casetype,
                    dev_casetype=dev_casetype)
    pos.pad_sequence()
    pos.embedding(embedding=embedding)
    pos.map_y()

    if hyperparameter_search:
        _MODELS = [Model.BILSTM_CRF]
        _LSTM_HIDDEN_UNITS = [1, 2, 4, 8, 32, 128, 512]
        _LSTM_DROPOUT = [0.0, 0.2, 0.4]
        _LSTM_RECURRENT_DROPOUT = _LSTM_DROPOUT
        _LEARNING_RATE = [1e-1, 1e-3]

        for model in _MODELS:
            for lstm_hidden_units in _LSTM_HIDDEN_UNITS:
                for lstm_dropout in _LSTM_DROPOUT:
                    for lstm_recurrent_dropout in _LSTM_RECURRENT_DROPOUT:
                        for learning_rate in _LEARNING_RATE:
                            pos.set_model_params(model=model,
                                                 lstm_hidden_units=lstm_hidden_units,
                                                 lstm_dropout=lstm_dropout,
                                                 lstm_recurrent_dropout=lstm_recurrent_dropout,
                                                 num_gpus=num_gpus,
                                                 learning_rate=learning_rate)

                            if not pos.try_load_model():
                                pos.create_model()
                                pos.train_model()
                                pos.save_model()
                            pos.model_accuracy(X_embedded=pos.dev_x_embedded, y_int=pos.dev_y_int, dataset="dev")
                            pos.model_accuracy(X_embedded=pos.test_x_embedded, y_int=pos.test_y_int, dataset="test")
    else:
        pos.set_model_params(model=model,
                             lstm_hidden_units=lstm_hidden_units,
                             lstm_dropout=lstm_dropout,
                             lstm_recurrent_dropout=lstm_recurrent_dropout,
                             num_gpus=num_gpus,
                             learning_rate=learning_rate)

        if not pos.try_load_model():
            pos.create_model()
            pos.train_model()
            pos.save_model()
        pos.model_accuracy(X_embedded=pos.dev_x_embedded, y_int=pos.dev_y_int, dataset="dev")
        pos.model_accuracy(X_embedded=pos.test_x_embedded, y_int=pos.test_y_int, dataset="test")
        
    # exit
    if pos.log_level.value >= LogLevel.LIMITED.value:
        print("Exit.")
    sys.exit(0)

