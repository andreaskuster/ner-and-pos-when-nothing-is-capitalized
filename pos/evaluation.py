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

import os.path

from enum import Enum, auto
from argparse import ArgumentParser

from pos import Model, CaseType

_BASE_PATH = "pos/results"


class Dataset(Enum):
    DEV_ACCURACY = "_dev_accuracy"
    TEST_ACCURACY = "_test_accuracy"
    HISTORY_ACCURACY = "_history_accuracy"
    HISTORY_EPOCH = "_history_epoch"
    HISTORY_LOSS = "_history_loss"
    HISTORY_VAL_ACCURACY = "_history_val_accuracy"
    HISTORY_VAL_LOSS = "_history_val_loss"


class Plot(Enum):
    LSTM_HIDDEN_UNITS_SERIES = auto()
    LSTM_HIDDEN_UNITS_SERIES_COMBINED = auto()
    EPOCH_SERIES = auto()
    EPOCH_SERIES_COMBINED = auto()
    LEARNING_RATE_SERIES = auto()
    LSTM_DROPOUT_SERIES = auto()


def load(parameter, model_list, lstm_hidden_units_list, lstm_dropout_list, learning_rate_list, casetype_train,
         casetype_test, casetype_dev):
    name, value, epoch = list(), list(), list()

    history_data = [Dataset.HISTORY_ACCURACY, Dataset.HISTORY_LOSS, Dataset.HISTORY_VAL_ACCURACY,
                    Dataset.HISTORY_VAL_LOSS]

    for model in model_list:
        for lstm_hidden_units in lstm_hidden_units_list:
            for lstm_dropout in lstm_dropout_list:
                lstm_recurrent_dropout = lstm_dropout
                for learning_rate in learning_rate_list:
                    base = "{}_{}_units_{}_dropout_{}_recdropout_{}_lr_{}_train_{}_test_{}_dev".format(
                        model.name,
                        lstm_hidden_units,
                        lstm_dropout,
                        lstm_recurrent_dropout,
                        learning_rate,
                        casetype_train.name,
                        casetype_test.name,
                        casetype_dev.name)

                    # name.append(base)
                    name.append("{}u_{}d_{}l".format(lstm_hidden_units, lstm_dropout, learning_rate))

                    value.append(np.loadtxt(os.path.join(_BASE_PATH, base + parameter.value)))
                    if parameter in history_data:
                        epoch.append(np.loadtxt(os.path.join(_BASE_PATH, base + Dataset.HISTORY_EPOCH.value)))
    return name, value, epoch


if __name__ == "__main__":

    _MODELS = [Model.BILSTM_CRF]
    _LSTM_HIDDEN_UNITS = [1, 2, 4, 8, 32, 128, 512]
    _LSTM_DROPOUT = [0.0, 0.2, 0.4]
    _LEARNING_RATE = [1e-1, 1e-3]
    _CASETYPE_TRAIN = CaseType.CASED
    _CASETYPE_TEST = CaseType.CASED
    _CASETYPE_DEV = CaseType.CASED

    parser: ArgumentParser = ArgumentParser()
    parser.add_argument("-d", "--dataset", default=Dataset.DEV_ACCURACY, choices=[x.name for x in Dataset])
    parser.add_argument("-p", "--plot", default=Plot.LSTM_HIDDEN_UNITS_SERIES, choices=[x.name for x in Plot])

    args = parser.parse_args()
    dataset: Dataset = Dataset[args.dataset]
    plot: Plot = Plot[args.plot]

    if plot == Plot.LSTM_HIDDEN_UNITS_SERIES:

        # load data
        name, value, _ = load(parameter=dataset,
                              model_list=_MODELS,
                              lstm_hidden_units_list=_LSTM_HIDDEN_UNITS,
                              lstm_dropout_list=[0.0],  # use best param
                              learning_rate_list=[1e-3],  # use best param
                              casetype_train=_CASETYPE_TRAIN,
                              casetype_test=_CASETYPE_TEST,
                              casetype_dev=_CASETYPE_DEV)

        from matplotlib import pyplot as plt

        plt.figure(num=None, figsize=(12, 4), dpi=300)

        plt.bar(range(len(_LSTM_HIDDEN_UNITS)), value, label=dataset.name)
        plt.title(plot.name)
        plt.ylabel("Accuracy")
        plt.xlabel("LSTM Hidden Units")
        plt.xticks(range(len(_LSTM_HIDDEN_UNITS)), _LSTM_HIDDEN_UNITS)  # rotation="vertical")
        plt.ylim(0.8, 1.0)
        plt.legend()

        plt.savefig("{}_{}.png".format(plot.name, dataset.name), bbox_inches="tight")
        plt.show()

    elif plot == Plot.LSTM_HIDDEN_UNITS_SERIES_COMBINED:

        # load data
        name0, value0, _ = load(parameter=Dataset.DEV_ACCURACY,
                                model_list=_MODELS,
                                lstm_hidden_units_list=_LSTM_HIDDEN_UNITS,
                                lstm_dropout_list=[0.0],  # use best param
                                learning_rate_list=[1e-3],  # use best param
                                casetype_train=_CASETYPE_TRAIN,
                                casetype_test=_CASETYPE_TEST,
                                casetype_dev=_CASETYPE_DEV)
        name1, value1, _ = load(parameter=Dataset.TEST_ACCURACY,
                                model_list=_MODELS,
                                lstm_hidden_units_list=_LSTM_HIDDEN_UNITS,
                                lstm_dropout_list=[0.0],  # use best param
                                learning_rate_list=[1e-3],  # use best param
                                casetype_train=_CASETYPE_TRAIN,
                                casetype_test=_CASETYPE_TEST,
                                casetype_dev=_CASETYPE_DEV)

        from matplotlib import pyplot as plt
        import numpy as np

        plt.figure(num=None, figsize=(12, 4), dpi=300)

        x = np.arange(len(_LSTM_HIDDEN_UNITS))  # the label locations
        width = 0.35  # the width of the bars

        plt.bar(x - width / 2, value0, width, label=Dataset.DEV_ACCURACY.name)
        plt.bar(x + width / 2, value1, width, label=Dataset.TEST_ACCURACY.name)

        plt.title(plot.name)
        plt.ylabel("Accuracy")
        plt.xlabel("LSTM Hidden Units")
        plt.xticks(range(len(_LSTM_HIDDEN_UNITS)), _LSTM_HIDDEN_UNITS)  # rotation="vertical")
        plt.ylim(0.8, 1.0)
        plt.legend()

        plt.savefig("{}_{}.png".format(plot.name, dataset.name), bbox_inches="tight")
        plt.show()

    elif plot == Plot.EPOCH_SERIES:

        name, value, epoch = load(parameter=dataset,
                                  model_list=_MODELS,
                                  lstm_hidden_units_list=[512],  # use best param
                                  lstm_dropout_list=[0.0],  # use best param
                                  learning_rate_list=[1e-3],  # use best param
                                  casetype_train=_CASETYPE_TRAIN,
                                  casetype_test=_CASETYPE_TEST,
                                  casetype_dev=_CASETYPE_DEV)
        name, value, epoch = name[0], value[0], epoch[0]

        from matplotlib import pyplot as plt

        plt.figure(num=None, figsize=(12, 4), dpi=300)

        plt.plot(epoch, value, label=dataset.name)

        plt.title(plot.name)
        plt.ylabel("Accuracy")
        plt.xlabel("Epochs")
        plt.xticks(range(len(value)), range(len(value)))  # rotation="vertical")

        plt.ylim(0.95, 1.0)
        plt.legend()

        plt.savefig("{}_{}.png".format(plot.name, dataset.name), bbox_inches="tight")
        plt.show()

    elif plot == Plot.EPOCH_SERIES_COMBINED:

        name, value, epoch = load(parameter=dataset,
                                  model_list=_MODELS,
                                  lstm_hidden_units_list=_LSTM_HIDDEN_UNITS,
                                  lstm_dropout_list=[0.0],  # use best param
                                  learning_rate_list=[1e-3],  # use best param
                                  casetype_train=_CASETYPE_TRAIN,
                                  casetype_test=_CASETYPE_TEST,
                                  casetype_dev=_CASETYPE_DEV)

        from matplotlib import pyplot as plt

        plt.figure(num=None, figsize=(12, 4), dpi=300)

        max_epoch = range(max([len(x) for x in epoch]))

        for i in range(len(epoch)):
            plt.plot(epoch[i], value[i], label="{}_{}_HIDDEN_UNITS".format(dataset.name, _LSTM_HIDDEN_UNITS[i]))

        plt.title(plot.name)
        plt.ylabel("Accuracy")
        plt.xlabel("Epochs")
        plt.xticks(range(len(max_epoch)), range(len(max_epoch)))  # rotation="vertical")

        if dataset in [Dataset.HISTORY_VAL_ACCURACY]:
            plt.ylim(0.85, 1.0)
        elif dataset in [Dataset.HISTORY_ACCURACY]:
            plt.ylim(0.55, 1.0)

        plt.legend()

        plt.savefig("{}_{}.png".format(plot.name, dataset.name), bbox_inches="tight")
        plt.show()

    elif plot == Plot.LEARNING_RATE_SERIES:

        name, value, epoch = load(parameter=dataset,
                                  model_list=_MODELS,
                                  lstm_hidden_units_list=[512],  # use best param
                                  lstm_dropout_list=[0.0],  # use best param
                                  learning_rate_list=_LEARNING_RATE,
                                  casetype_train=_CASETYPE_TRAIN,
                                  casetype_test=_CASETYPE_TEST,
                                  casetype_dev=_CASETYPE_DEV)

        from matplotlib import pyplot as plt

        plt.figure(num=None, figsize=(12, 4), dpi=300)

        max_epoch = range(max([len(x) for x in epoch]))

        for i in range(len(epoch)):
            plt.plot(epoch[i], value[i], label="{}_{}_LEARNING_RATE".format(dataset.name, _LEARNING_RATE[i]))

        plt.title(plot.name)
        plt.ylabel("Accuracy")
        plt.xlabel("Epochs")
        plt.xticks(range(len(max_epoch)), range(len(max_epoch)))  # rotation="vertical")

        plt.legend()

        plt.savefig("{}_{}.png".format(plot.name, dataset.name), bbox_inches="tight")
        plt.show()

    elif plot == Plot.LSTM_DROPOUT_SERIES:

        name, value, epoch = load(parameter=dataset,
                                  model_list=_MODELS,
                                  lstm_hidden_units_list=[512],  # use best param
                                  lstm_dropout_list=_LSTM_DROPOUT,
                                  learning_rate_list=[1e-3],  # use best param
                                  casetype_train=_CASETYPE_TRAIN,
                                  casetype_test=_CASETYPE_TEST,
                                  casetype_dev=_CASETYPE_DEV)

        from matplotlib import pyplot as plt

        plt.figure(num=None, figsize=(12, 4), dpi=300)

        max_epoch = range(max([len(x) for x in epoch]))

        for i in range(len(epoch)):
            plt.plot(epoch[i], value[i], label="{}_{}_LSTM_DROPOUT".format(dataset.name, _LSTM_DROPOUT[i]))

        plt.title(plot.name)
        plt.ylabel("Accuracy")
        plt.xlabel("Epochs")
        plt.xticks(range(len(max_epoch)), range(len(max_epoch)))  # rotation="vertical")

        plt.legend()

        plt.savefig("{}_{}.png".format(plot.name, dataset.name), bbox_inches="tight")
        plt.show()

    else:
        raise RuntimeError("Parameter {} not intended for plotting.".format(dataset.name))
