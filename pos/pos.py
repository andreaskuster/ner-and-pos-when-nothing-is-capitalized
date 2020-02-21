import os
import sys
import argparse

from enum import Enum, auto

from sklearn.model_selection import train_test_split

from datasets.ptb.penn_treebank import PTB
from datasets.brown.brown import Brown
from datasets.conll2000.conll2000 import CoNLL2000


class Verbosity(Enum):
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


class POS:

    def __init__(self,
                 verbosity_level: Verbosity = Verbosity.LIMITED):
        self.verbosity_level = verbosity_level  # 0: none, 1: limited, 2: full

    def set_cuda_visible_devices(self,
                                 devices: str = None):
        """
        Masks visible cuda devices (i.e. useful for parallel job execution).
        :param devices: string with comma separated device indices
        :return None
        """
        # report action
        if self.verbosity_level.value >= Verbosity.LIMITED.value:
            print("Set visible cuda devices to {}".format(devices))
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
        if self.verbosity_level.value >= Verbosity.LIMITED.value:
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
            # split data: train: 10 sentences, dev & test: 4 sentences each with 2 overlapping sentences
            train_x, train_y = data_x[2:12], data_y[2:12]
            dev_x, dev_y = data_x[0:4], data_y[0:4]
            test_x, test_y = data_x[10:14], data_y[10:14]

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
        # return data
        return train_x, train_y, test_x, test_y, dev_x, dev_y


if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', action='store_true',
                        help="shows output")
    args = parser.parse_args()

    if args.output:
        print("This is some output")

    pos = POS()

    # exit
    if pos.verbosity_level.value >= Verbosity.LIMITED.value:
        print("Exit.")
    sys.exit(0)
