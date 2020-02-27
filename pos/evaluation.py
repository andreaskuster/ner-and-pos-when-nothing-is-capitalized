import os.path

import numpy as np

from enum import Enum
from argparse import ArgumentParser

from pos import Model, CaseType


class Parameter(Enum):
    DEV_ACCURACY = "_dev_accuracy"
    TEST_ACCURACY = "_test_accuracy"
    HISTORY_ACCURACY = "_history_accuracy"
    HISTORY_EPOCH = "_history_epoch"
    HISTORY_LOSS = "_history_loss"
    HISTORY_VAL_ACCURACY = "_history_val_accuracy"
    HISTORY_VAL_LOSS = "_history_val_loss"


if __name__ == "__main__":

    _MODELS = [Model.BILSTM_CRF]
    _LSTM_HIDDEN_UNITS = [1, 2, 4, 8, 32, 128, 512]
    _LSTM_DROPOUT = [0.0, 0.2, 0.4]
    _LSTM_RECURRENT_DROPOUT = _LSTM_DROPOUT
    _LEARNING_RATE = [1e-1, 1e-3]
    _CASETYPE_TRAIN = CaseType.CASED
    _CASETYPE_TEST = CaseType.CASED
    _CASETYPE_DEV = CaseType.CASED

    _BASE_PATH = "pos/results"

    parser: ArgumentParser = ArgumentParser()
    parser.add_argument("-d", "--dataset", default=Parameter.DEV_ACCURACY, choices=[x.name for x in Parameter])
    args = parser.parse_args()
    parameter: Parameter = Parameter[args.dataset]

    name, value = list(), list()

    history_data = [Parameter.HISTORY_ACCURACY, Parameter.HISTORY_LOSS, Parameter.HISTORY_VAL_ACCURACY,
                       Parameter.HISTORY_VAL_LOSS]

    if parameter in history_data:
        epoch = list()

    for model in _MODELS:
        for lstm_hidden_units in _LSTM_HIDDEN_UNITS:
            for lstm_dropout in _LSTM_DROPOUT:
                for lstm_recurrent_dropout in _LSTM_RECURRENT_DROPOUT:
                    for learning_rate in _LEARNING_RATE:
                        base = "{}_{}_units_{}_dropout_{}_recdropout_{}_lr_{}_train_{}_test_{}_dev".format(
                            model.name,
                            lstm_hidden_units,
                            lstm_dropout,
                            lstm_recurrent_dropout,
                            learning_rate,
                            _CASETYPE_TRAIN.name,
                            _CASETYPE_TEST.name,
                            _CASETYPE_DEV.name)

                        name.append(base)
                        value.append(float(np.loadtxt(os.path.join(_BASE_PATH, base + parameter.value))))
                        if parameter in history_data:
                            epoch.append(float(np.loadtxt(os.path.join(_BASE_PATH, base + Parameter.HISTORY_EPOCH.value))))

    if parameter == Parameter.DEV_ACCURACY or parameter == Parameter.TEST_ACCURACY:

        from matplotlib import pyplot as plt
        from matplotlib import style

        style.use("ggplot")

        plt.bar(name, value, align="center")

        plt.title(parameter.name)
        plt.ylabel("Accuracy")
        plt.xlabel("Hyperparameters")

        plt.show()

    elif parameter in history_data:

        import matplotlib.pyplot as pyplot

        # TODO: finish implementation details
        # extract longest epoch number
        #max_epoch_len = max([len(x) for x in epoch])
        #x_epoch = range(max_epoch_len)

        # pad all sequences to that length
        #for index in range(len(value)):
        #    value[index] = value[index] + (value[index][-1]*(max_epoch_len-len(value[index])))

        for index in range(len(value)):
            pyplot.plot(epoch[index], value[index], label=name[index])

        pyplot.title(parameter.name)
        pyplot.show()

    else:
        raise RuntimeError("Parameter {} not intended for plotting.".format(parameter.name))
