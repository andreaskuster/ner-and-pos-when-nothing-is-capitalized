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

import sys
import os.path

import numpy as np
import matplotlib.pyplot as plt

from enum import Enum, auto
from argparse import ArgumentParser

from pos import Model, CaseType

_BASE_PATH = "pos/results"  # path to the experiment output files


# enumeration of all experiment output files
class Dataset(Enum):
    DEV_ACCURACY = "_dev_accuracy"
    TEST_ACCURACY = "_test_accuracy"
    HISTORY_ACCURACY = "_history_accuracy"
    HISTORY_EPOCH = "_history_epoch"
    HISTORY_LOSS = "_history_loss"
    HISTORY_VAL_ACCURACY = "_history_val_accuracy"
    HISTORY_VAL_LOSS = "_history_val_loss"


# enumeration of all plot types
class Plot(Enum):
    LSTM_HIDDEN_UNITS_SERIES = auto()
    LSTM_HIDDEN_UNITS_SERIES_COMBINED = auto()
    EPOCH_SERIES = auto()
    EPOCH_SERIES_COMBINED = auto()
    LEARNING_RATE_SERIES = auto()
    LSTM_DROPOUT_SERIES = auto()


# method loading the corresponding experiment data
def load(parameter, model_list, lstm_hidden_units_list, lstm_dropout_list, learning_rate_list, casetype_train,
         casetype_test, casetype_dev):
    # init output lists
    name, value, epoch = list(), list(), list()
    # list all datasets that rely on the epoch dataset too
    history_data = [Dataset.HISTORY_ACCURACY, Dataset.HISTORY_LOSS, Dataset.HISTORY_VAL_ACCURACY,
                    Dataset.HISTORY_VAL_LOSS]
    # loop over all parameters
    for model in model_list:
        for lstm_hidden_units in lstm_hidden_units_list:
            for lstm_dropout in lstm_dropout_list:
                lstm_recurrent_dropout = lstm_dropout
                for learning_rate in learning_rate_list:
                    # create base address
                    base = "{}_{}_units_{}_dropout_{}_recdropout_{}_lr_{}_train_{}_test_{}_dev".format(
                        model.name,
                        lstm_hidden_units,
                        lstm_dropout,
                        lstm_recurrent_dropout,
                        learning_rate,
                        casetype_train.name,
                        casetype_test.name,
                        casetype_dev.name)
                    # create name list
                    name.append("{}u_{}d_{}l".format(lstm_hidden_units, lstm_dropout, learning_rate))
                    # add the value
                    value.append(np.loadtxt(os.path.join(_BASE_PATH, base + parameter.value)))
                    # add epoch parameter if required
                    if parameter in history_data:
                        epoch.append(np.loadtxt(os.path.join(_BASE_PATH, base + Dataset.HISTORY_EPOCH.value)))
    # return created list
    return name, value, epoch


if __name__ == "__main__":
    # hyper parameter grid search values
    _MODELS = [Model.BILSTM_CRF]
    _LSTM_HIDDEN_UNITS = [1, 2, 4, 8, 32, 128, 512]
    _LSTM_DROPOUT = [0.0, 0.2, 0.4]
    _LEARNING_RATE = [1e-1, 1e-3]
    _CASETYPE_TRAIN = CaseType.CASED
    _CASETYPE_TEST = CaseType.CASED
    _CASETYPE_DEV = CaseType.CASED
    # define command line arguments
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument("-d", "--dataset", default=Dataset.DEV_ACCURACY, choices=[x.name for x in Dataset])
    parser.add_argument("-p", "--plot", default=Plot.LSTM_HIDDEN_UNITS_SERIES, choices=[x.name for x in Plot])
    # parse args
    args = parser.parse_args()
    dataset: Dataset = Dataset[args.dataset]
    plot: Plot = Plot[args.plot]
    # init global plot settings
    plt.figure(num=None, figsize=(12, 4), dpi=300)
    # branch on plot type
    if plot == Plot.LSTM_HIDDEN_UNITS_SERIES:  # hidden units plot
        # load data
        name, value, _ = load(parameter=dataset,
                              model_list=_MODELS,
                              lstm_hidden_units_list=_LSTM_HIDDEN_UNITS,
                              lstm_dropout_list=[0.0],  # use best param
                              learning_rate_list=[1e-3],  # use best param
                              casetype_train=_CASETYPE_TRAIN,
                              casetype_test=_CASETYPE_TEST,
                              casetype_dev=_CASETYPE_DEV)
        # create bar plot
        plt.bar(range(len(_LSTM_HIDDEN_UNITS)), value, label=dataset.name)
        plt.title(plot.name)
        plt.ylabel("Accuracy")
        plt.xlabel("LSTM Hidden Units")
        plt.xticks(range(len(_LSTM_HIDDEN_UNITS)), _LSTM_HIDDEN_UNITS)  # map numbers to hidden unit size values
        plt.ylim(0.8, 1.0)  # y axis limits
        plt.legend()
    elif plot == Plot.LSTM_HIDDEN_UNITS_SERIES_COMBINED:  # combined hidden units plot
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
        # create bar plot
        x = np.arange(len(_LSTM_HIDDEN_UNITS))  # the label locations
        width = 0.35  # the width of the bars
        plt.bar(x - width / 2, value0, width, label=Dataset.DEV_ACCURACY.name)
        plt.bar(x + width / 2, value1, width, label=Dataset.TEST_ACCURACY.name)
        plt.title(plot.name)
        plt.ylabel("Accuracy")
        plt.xlabel("LSTM Hidden Units")
        plt.xticks(range(len(_LSTM_HIDDEN_UNITS)), _LSTM_HIDDEN_UNITS)  # map numbers to hidden unit size values
        plt.ylim(0.8, 1.0)  # y axis limits
        plt.legend()
    elif plot == Plot.EPOCH_SERIES:  # accuracy change over the training epochs, for the best params
        # load data
        name, value, epoch = load(parameter=dataset,
                                  model_list=_MODELS,
                                  lstm_hidden_units_list=[512],  # use best param
                                  lstm_dropout_list=[0.0],  # use best param
                                  learning_rate_list=[1e-3],  # use best param
                                  casetype_train=_CASETYPE_TRAIN,
                                  casetype_test=_CASETYPE_TEST,
                                  casetype_dev=_CASETYPE_DEV)
        # unpack
        name, value, epoch = name[0], value[0], epoch[0]
        # create line plot
        plt.plot(epoch, value, label=dataset.name)
        plt.title(plot.name)
        plt.ylabel("Accuracy")
        plt.xlabel("Epochs")
        plt.xticks(range(len(value)), range(len(value)))
        plt.ylim(0.95, 1.0)  # y axis limits
        plt.legend()
    elif plot == Plot.EPOCH_SERIES_COMBINED: # accuracy over epochs, different hidden unit sizes
        # load data
        name, value, epoch = load(parameter=dataset,
                                  model_list=_MODELS,
                                  lstm_hidden_units_list=_LSTM_HIDDEN_UNITS,
                                  lstm_dropout_list=[0.0],  # use best param
                                  learning_rate_list=[1e-3],  # use best param
                                  casetype_train=_CASETYPE_TRAIN,
                                  casetype_test=_CASETYPE_TEST,
                                  casetype_dev=_CASETYPE_DEV)
        # create plot
        max_epoch = range(max([len(x) for x in epoch]))
        # iterate over all subplots
        for i in range(len(epoch)):
            plt.plot(epoch[i], value[i], label="{}_{}_HIDDEN_UNITS".format(dataset.name, _LSTM_HIDDEN_UNITS[i]))
        plt.title(plot.name)
        plt.ylabel("Accuracy")
        plt.xlabel("Epochs")
        plt.xticks(range(len(max_epoch)), range(len(max_epoch)))  # map numbers to hidden unit size values
        # set y-axis limits
        if dataset in [Dataset.HISTORY_VAL_ACCURACY]:
            plt.ylim(0.85, 1.0)
        elif dataset in [Dataset.HISTORY_ACCURACY]:
            plt.ylim(0.55, 1.0)
        plt.legend()
    elif plot == Plot.LEARNING_RATE_SERIES:  # plot different learning rates
        # load data
        name, value, epoch = load(parameter=dataset,
                                  model_list=_MODELS,
                                  lstm_hidden_units_list=[512],  # use best param
                                  lstm_dropout_list=[0.0],  # use best param
                                  learning_rate_list=_LEARNING_RATE,
                                  casetype_train=_CASETYPE_TRAIN,
                                  casetype_test=_CASETYPE_TEST,
                                  casetype_dev=_CASETYPE_DEV)
        # find experiment with highest number of epochs
        max_epoch = range(max([len(x) for x in epoch]))
        # add all sub plots
        for i in range(len(epoch)):
            plt.plot(epoch[i], value[i], label="{}_{}_LEARNING_RATE".format(dataset.name, _LEARNING_RATE[i]))
        plt.title(plot.name)
        plt.ylabel("Accuracy")
        plt.xlabel("Epochs")
        plt.xticks(range(len(max_epoch)), range(len(max_epoch)))  # rotation="vertical")
        plt.legend()
    elif plot == Plot.LSTM_DROPOUT_SERIES: # plot different dropout rates
        # load data
        name, value, epoch = load(parameter=dataset,
                                  model_list=_MODELS,
                                  lstm_hidden_units_list=[512],  # use best param
                                  lstm_dropout_list=_LSTM_DROPOUT,
                                  learning_rate_list=[1e-3],  # use best param
                                  casetype_train=_CASETYPE_TRAIN,
                                  casetype_test=_CASETYPE_TEST,
                                  casetype_dev=_CASETYPE_DEV)
        # find experiment with highest number of epochs
        max_epoch = range(max([len(x) for x in epoch]))
        # add all sub plots
        for i in range(len(epoch)):
            plt.plot(epoch[i], value[i], label="{}_{}_LSTM_DROPOUT".format(dataset.name, _LSTM_DROPOUT[i]))
        plt.title(plot.name)
        plt.ylabel("Accuracy")
        plt.xlabel("Epochs")
        plt.xticks(range(len(max_epoch)), range(len(max_epoch)))  # rotation="vertical")
        plt.legend()
    else:
        raise RuntimeError("Parameter {} not intended for plotting.".format(dataset.name))
    # save and show plot
    plt.savefig("{}_{}.png".format(plot.name, dataset.name), bbox_inches="tight")
    plt.show()
    # exit
    sys.exit(0)
