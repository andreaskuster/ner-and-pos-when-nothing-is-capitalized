import os
import sys

from enum import Enum, auto
from argparse import ArgumentParser

from sklearn.model_selection import train_test_split

from datasets.ptb.penn_treebank import PTB
from datasets.brown.brown import Brown
from datasets.conll2000.conll2000 import CoNLL2000


class LogLevel(Enum):
    NO = 0
    LIMITED = 1
    FULL = 2


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


class POS:

    def __init__(self,
                 log_level: LogLevel = LogLevel.LIMITED):
        self.log_level: LogLevel = log_level  # 0: none, 1: limited, 2: full

        self.train_x, self.train_y, self.test_x, self.test_y, self.dev_x, self.dev_y = None, None, None, None, None, None
        self.dataset = {
            "train_x": None,
            "train_y": None,
            "test_x": None,
            "test_y": None,
            "dev_x": None,
            "dev_y": None
        }

        self.max_sentence_length: int = 0

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
        # return dataset
        return train_x, train_y, test_x, test_y, dev_x, dev_y

    def pad_sequence(self):

        # report action
        if self.log_level.value >= LogLevel.LIMITED.value:
            print("Padding sequence...")

        # update maximum sentence length
        for data_x in [self.train_x, self.test_x, self.dev_x]:
            self.max_sentence_length = max(self.max_sentence_length, max([len(x) for x in data_x]))

        # pad x
        for data_x in [self.train_x, self.test_x, self.dev_x]:
            for i in range(len(data_x)):
                no_pad = self.max_sentence_length - len(data_x[i])
                data_x[i] = list(data_x[i]) + ([""] * no_pad)

        # pad y
        for data_y in [self.train_y, self.test_y, self.dev_y]:
            for i in range(len(data_y)):
                no_pad = self.max_sentence_length - len(data_y[i])
                data_y[i] = list(data_y[i]) + ([""] * no_pad)


if __name__ == "__main__":

    # parse arguments
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument("-d", "--dataset", default="PTB", choices=[x.name for x in Dataset])
    parser.add_argument("-c", "--casetype", default="CASED", choices=[x.name for x in CaseType])
    args = parser.parse_args()

    # convert args to correct data types
    dataset: Dataset = Dataset[args.dataset]
    casetype: CaseType = CaseType[args.casetype]


    print("Dataset is: {}".format(dataset.name))
    print("Casetype is: {}".format(casetype.name))

    pos = POS()
    pos.import_data(dataset=dataset,
                    casetype=casetype)
    pos.pad_sequence()

    # exit
    if pos.log_level.value >= LogLevel.LIMITED.value:
        print("Exit.")
    sys.exit(0)
