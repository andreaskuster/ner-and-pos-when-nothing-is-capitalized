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


class Embedding(Enum):
    GLOVE = auto()
    WORD2VEC = auto()
    ELMO = auto()


class DataSourceWord2Vec(Enum):
    GOOGLE_NEWS_300: str = "word2vec-google-news-300"


class DataSourceGlove(Enum):
    COMMON_CRAWL_840B_CASED_300D: Tuple[str, str] = ("common_crawl_840b_cased", "glove.840B.300d.txt")


class POS:

    def __init__(self,
                 log_level: LogLevel = LogLevel.LIMITED,
                 data_source_word2vec: DataSourceWord2Vec = DataSourceWord2Vec.GOOGLE_NEWS_300,
                 data_source_glove: DataSourceGlove = DataSourceGlove.COMMON_CRAWL_840B_CASED_300D,
                 batch_size_embedding: int = 4096):
        # log level
        self.log_level: LogLevel = log_level  # 0: none, 1: limited, 2: full
        # dataset
        self.dataset = None
        # casetype
        self.casetype = None
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
                    casetype: CaseType,
                    test_size: float = 0.2):
        """

        :param dataset:
        :param casetype:
        :param test_size:
        :return:
        """
        # report action
        if self.log_level.value >= LogLevel.LIMITED.value:
            print("Importing data...")
        # store parameters
        self.dataset = dataset
        self.casetype = casetype
        # load data
        train_x, train_y, test_x, test_y, dev_x, dev_y = None, None, None, None, None, None
        if dataset == Dataset.PTB_DUMMY:
            # instantiate data loader
            ptb = PTB()
            # load first section
            if casetype == CaseType.CASED:
                data_x, data_y = ptb.load_data([0])
            elif casetype == CaseType.UNCASED:
                data_x, data_y = ptb.load_data_lowercase([0])
            elif casetype == CaseType.TRUECASE:
                data_x, data_y = ptb.load_data_truecase([0])
            # split data: train: 4 sentences, dev & test: 2 sentences each with 1 overlapping sentences
            train_x, train_y = data_x[1:5], data_y[1:5]
            dev_x, dev_y = data_x[0:2], data_y[0:2]
            test_x, test_y = data_x[4:6], data_y[4:6]

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
            if casetype == CaseType.CASED:
                train_x, train_y = ptb.load_data(train_sections)
                dev_x, dev_y = ptb.load_data(dev_sections)
                test_x, test_y = ptb.load_data(test_sections)
            elif casetype == CaseType.UNCASED:
                train_x, train_y = ptb.load_data_lowercase(train_sections)
                dev_x, dev_y = ptb.load_data_lowercase(dev_sections)
                test_x, test_y = ptb.load_data_lowercase(test_sections)
            elif casetype == CaseType.TRUECASE:
                train_x, train_y = ptb.load_data_truecase(train_sections)
                dev_x, dev_y = ptb.load_data_truecase(dev_sections)
                test_x, test_y = ptb.load_data_truecase(test_sections)

        elif dataset == Dataset.BROWN or dataset == Dataset.CONLL2000:
            # instantiate data loader
            data_loader = Brown() if dataset == Dataset.BROWN else CoNLL2000()
            # load data
            if casetype == CaseType.CASED:
                data_x, data_y = data_loader.load_data()
            elif casetype == CaseType.UNCASED:
                data_x, data_y = data_loader.load_data_lowercase()
            elif casetype == CaseType.TRUECASE:
                data_x, data_y = data_loader.load_data_truecase()
            # split data
            train_x, test_x, train_y, test_y = train_test_split(data_x, data_y,
                                                                test_size=test_size,
                                                                shuffle=True)  # do not expect data to be pre-shuffled
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

        :return:
        """
        # report action
        if self.log_level.value >= LogLevel.LIMITED.value:
            print("Padding sequence...")
        # update maximum sentence length
        for dataset in self.data_x:
            self.max_sentence_length = max(self.max_sentence_length, max([len(x) for x in self.data_x[dataset]]))
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
            path_data_x = "elmo_embedding_{}_{}_data_x.npz".format(self.dataset.name, self.casetype.name)
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
                        for i in range(num_batches):
                            # compute indices
                            start = i * self.batch_size_embedding
                            end = (i + 1) * self.batch_size_embedding
                            # do batch embedding
                            data_x_embedding.append(preprocessor.embedding(self.data_x[dataset][start:end]))
                            # report action
                            if self.log_level.value >= LogLevel.FULL.value:
                                print("elmo embedding round {}...".format(i))
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
                # store data to file
                path_data_x = "elmo_embedding_{}_{}_data_x.npz".format(self.dataset.name, self.casetype.name)
                np.savez(path_data_x,
                         train_x_embedded=self.dataset_x_embedded["train_x_embedded"],
                         test_x_embedded=self.dataset_x_embedded["test_x_embedded"],
                         dev_x_embedded=self.dataset_x_embedded["dev_x_embedded"])

        else:  # glove and word2vec are sequence independent -> process one-by-one
            for dataset in self.dataset_x:
                data_x_embedded = list()
                for sentence in self.data_x[dataset]:
                    data_x_embedded.append([preprocessor.word2vec(word) for word in sentence])
                self.dataset_map[self.data_x[dataset]] = np.array(data_x_embedded)

    def map_y(self):
        """

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
        if self.model_details is None:
            return ""
        return "{}_{}units_{}dropout_{}recdropout_{}lr".format(self.model_details["name"],
                                                               self.model_details["lstm_hidden_units"],
                                                               self.model_details["lstm_dropout"],
                                                               self.model_details["lstm_recurrent_dropout"],
                                                               self.model_details["learning_rate"],)

    def create_model_bilstm(self,
                            lstm_hidden_units=1024,
                            lstm_dropout=0.1,
                            lstm_recurrent_dropout=0.1,
                            num_gpus=1,
                            learning_rate=1e-3):
        # set model details
        self.model_details = dict()
        self.model_details["name"] = "bilstm"
        self.model_details["lstm_hidden_units"] = lstm_hidden_units
        self.model_details["lstm_dropout"] = lstm_dropout
        self.model_details["lstm_recurrent_dropout"] = lstm_recurrent_dropout
        self.model_details["learning_rate"] = learning_rate

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
        # set model details
        self.model_details = dict()
        self.model_details["name"] = "bilstmcrf"
        self.model_details["lstm_hidden_units"] = lstm_hidden_units
        self.model_details["lstm_dropout"] = lstm_dropout
        self.model_details["lstm_recurrent_dropout"] = lstm_recurrent_dropout
        self.model_details["learning_rate"] = learning_rate

        # create model
        self.model = Sequential()
        self.model.add(Bidirectional(layer=LSTM(units=lstm_hidden_units,
                                                return_sequences=True,
                                                dropout=lstm_dropout,
                                                recurrent_dropout=lstm_recurrent_dropout),
                                     input_shape=(self.max_len, self.dim_embedding_vec)))
        self.model.add(TimeDistributed(Dense(self.num_categories, activation="relu")))
        crf = CRF(self.num_categories)
        self.model.add(crf)
        # self.model = to_multi_gpu(self.model, n_gpus=num_gpus)
        self.model.compile(optimizer="rmsprop", loss=crf.loss_function, metrics=[crf.accuracy])

        if self.log_level.value >= LogLevel.LIMITED.value:
            self.model.summary()

    def save_model(self):
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
        if isfile("{}.json".format(self.model_name())) and isfile("{}.h5".format(self.model_name())):
            if self.log_level.value >= LogLevel.LIMITED.value:
                print("Loading model from file...")
            # load model
            json_file = open("{}.json".format(self.model_name()), "r")
            loaded_model_json = json_file.read()
            json_file.close()
            self.model = model_from_json(loaded_model_json)
            # load pre-trained weights
            self.model.load_weights("{}.h5".format(self.model_name()))
            return True
        return False

    def train_model(self,
                    batch_size=1024,
                    epochs=40):
        es = EarlyStopping(monitor="accuracy", mode="max", verbose=1, patience=4)
        # mc = ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
        # fit model
        history = self.model.fit(self.train_x_embedded, to_categorical(self.train_y_int, self.num_categories),
                                 batch_size=batch_size,
                                 epochs=epochs,
                                 callbacks=[es])
        # store history
        from matplotlib import pyplot
        pyplot.plot(history.history["accuracy"], label="train")
        pyplot.legend()
        pyplot.show()

    def model_accuracy(self):

        y_pred = self.model.predict(self.test_x_embedded)

        total = 0
        count = 0
        pred_itr = y_pred.__iter__()
        for sentence in self.test_y_int:
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
        print("accuracy: {}".format(accuracy))
        return accuracy


if __name__ == "__main__":
    # parse arguments
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument("-d", "--dataset", default="PTB_DUMMY", choices=[x.name for x in Dataset])
    parser.add_argument("-c", "--casetype", default="CASED", choices=[x.name for x in CaseType])
    parser.add_argument("-v", "--loglevel", default="FULL", choices=[x.name for x in LogLevel])
    parser.add_argument("-e", "--embedding", default="ELMO", choices=[x.name for x in Embedding])
    parser.add_argument("-w", "--datasource_word2vec", default="GOOGLE_NEWS_300",
                        choices=[x.name for x in DataSourceWord2Vec])
    parser.add_argument("-g", "--datasource_glove", default="COMMON_CRAWL_840B_CASED_300D",
                        choices=[x.name for x in DataSourceGlove])

    args = parser.parse_args()

    # convert args to correct data types
    dataset: Dataset = Dataset[args.dataset]
    casetype: CaseType = CaseType[args.casetype]
    log_level: LogLevel = LogLevel[args.loglevel]
    embedding: Embedding = Embedding[args.embedding]
    datasource_word2vec: DataSourceWord2Vec = DataSourceWord2Vec[args.datasource_word2vec]
    datasource_glove: DataSourceGlove = DataSourceGlove[args.datasource_glove]

    if log_level.value >= LogLevel.LIMITED.value:
        print("Dataset is: {}".format(dataset.name))
        print("Casetype is: {}".format(casetype.name))
        print("Log level is: {}".format(log_level.name))
        print("Embedding is: {}".format(embedding.name))
        print("Data source word2vec is: {}".format(datasource_word2vec.name))
        print("Data source glove is: {}".format(datasource_glove.name))

    pos = POS(log_level=log_level,
              data_source_word2vec=datasource_word2vec,
              data_source_glove=datasource_glove)
    pos.set_cuda_visible_devices(devices=None)
    pos.import_data(dataset=dataset,
                    casetype=casetype)
    pos.pad_sequence()
    pos.embedding(embedding=embedding)
    pos.map_y()
    if not pos.try_load_model():
        pos.create_model_bilstm()
        pos.train_model()
        pos.save_model()
    pos.model_accuracy()

    # exit
    if pos.log_level.value >= LogLevel.LIMITED.value:
        print("Exit.")
    sys.exit(0)
