# Part-of-speech Tagging (POS) - Experiments

## Code

### Usage
example command for experiment 1.1 from the paper:

```
python3 pos/pos.py --model BILSTM_CRF --dataset PTB --traincasetype CASED --devcasetype CASED --testcasetype CASED --embedding ELMO --batchsize 1024 --epochs 40 --learningrate 1e-3 --lstmhiddenunits 512 --lstmdropout 0.0 --lstmrecdropout 0.0 --numgpus 2
```

```
usage: pos.py [-h] [-d {PTB,PTB_DUMMY,PTB_REDUCED,BROWN,CONLL2000}]
              [-ctr {CASED,UNCASED,TRUECASE,CASED_UNCASED,HALF_MIXED}]
              [-cte {CASED,UNCASED,TRUECASE,CASED_UNCASED,HALF_MIXED}]
              [-cde {CASED,UNCASED,TRUECASE,CASED_UNCASED,HALF_MIXED}]
              [-v {NO,LIMITED,FULL}] [-e {GLOVE,WORD2VEC,ELMO}]
              [-w {GOOGLE_NEWS_300}] [-g {COMMON_CRAWL_840B_CASED_300D}]
              [-b BATCHSIZE] [-p EPOCHS] [-m {BILSTM,BILSTM_CRF}]
              [-ng NUMGPUS] [-lr LEARNINGRATE] [-hu LSTMHIDDENUNITS]
              [-dr LSTMDROPOUT] [-rdr LSTMRECDROPOUT] [-s {True,False}]
              [-c CUDADEVICES]

optional arguments:
  -h, --help            show this help message and exit
  -d {PTB,PTB_DUMMY,PTB_REDUCED,BROWN,CONLL2000}, --dataset {PTB,PTB_DUMMY,PTB_REDUCED,BROWN,CONLL2000}
  -ctr {CASED,UNCASED,TRUECASE,CASED_UNCASED,HALF_MIXED}, --traincasetype {CASED,UNCASED,TRUECASE,CASED_UNCASED,HALF_MIXED}
  -cte {CASED,UNCASED,TRUECASE,CASED_UNCASED,HALF_MIXED}, --testcasetype {CASED,UNCASED,TRUECASE,CASED_UNCASED,HALF_MIXED}
  -cde {CASED,UNCASED,TRUECASE,CASED_UNCASED,HALF_MIXED}, --devcasetype {CASED,UNCASED,TRUECASE,CASED_UNCASED,HALF_MIXED}
  -v {NO,LIMITED,FULL}, --loglevel {NO,LIMITED,FULL}
  -e {GLOVE,WORD2VEC,ELMO}, --embedding {GLOVE,WORD2VEC,ELMO}
  -w {GOOGLE_NEWS_300}, --datasource_word2vec {GOOGLE_NEWS_300}
  -g {COMMON_CRAWL_840B_CASED_300D}, --datasource_glove {COMMON_CRAWL_840B_CASED_300D}
  -b BATCHSIZE, --batchsize BATCHSIZE
  -p EPOCHS, --epochs EPOCHS
  -m {BILSTM,BILSTM_CRF}, --model {BILSTM,BILSTM_CRF}
  -ng NUMGPUS, --numgpus NUMGPUS
  -lr LEARNINGRATE, --learningrate LEARNINGRATE
  -hu LSTMHIDDENUNITS, --lstmhiddenunits LSTMHIDDENUNITS
  -dr LSTMDROPOUT, --lstmdropout LSTMDROPOUT
  -rdr LSTMRECDROPOUT, --lstmrecdropout LSTMRECDROPOUT
  -s {True,False}, --hyperparamsearch {True,False}
  -c CUDADEVICES, --cudadevices CUDADEVICES
```

### Data Import


### Sequence Padding


### Embedding


### Label Mapping

### Model



### Misc

#### Early Stopping

#### Hyperparameter Search

#### True Accuracy Report

#### Save / Load Model


## Hyperparameter Search
We performed hyperparameter grid search over the following parameters and values:

* LSTM hidden units: [1, 2, 4, 8, 32, 128, 512] (best:512)
* LSTM dropout: [0.0, 0.2, 0.4] (best: 0.0)
* LSTM recurrent dropout: [0.0, 0.2, 0.4] (best: 0.0)
* learning rate: [1e-1, 1e-3] (best: 1e-3)

The evaluation of the findings are discussed below.


### Dataset Split
We split all the dataset into training, development and testing data. The training data is used to train the model. For ptb, we used the sections described in the paper. For the other corpus, we used 80% of the data for training. The development data is used for evaluation during training and early stopping. Furthermore, it is used for the hyperparameter search. We used 10% of the data for development. The test dataset is never touched until the very last point, were we use it to compute the accuracy of our model with fresh data. It consists of the remaining 10% of data.

### Parameters

#### LSTM Hidden Units
We expected that the model will perform better (higher accuracy, lower loss) if we increase the number of lstm hidden units. As we can see in the two plots below, this hypothesis holds true. We therefore used the largest number of hidden units that we could efficiently run on our hardware. Furthermore, it is interesting that already very few units are capable (with enough training epochs) to reach a very high accuracy score.

![Plot: Validation Loss over Time](plots/EPOCH_SERIES_COMBINED_HISTORY_VAL_LOSS.png "Validation Loss over Time")

![Plot: Validation Accuracy over Time](plots/EPOCH_SERIES_COMBINED_HISTORY_VAL_ACCURACY.png "Validation Accuracy over Time")

#### Learning Rate
We expect that a small learning rate can (potentially) end up at a higher accuracy score, but is much slower in terms of convergence. The evaluation confirms this hypothesis. For better illustration, we added the plot for 8 hidden units (first one) with slower convergence too. 

![Plot: Validation Accuracy over Time (8 LSTM Hidden Units)](plots/LEARNING_RATE_SERIES_HISTORY_VAL_ACCURACY_8.png "Validation Accuracy over Time (8 LSTM Hidden Units)")

![Plot: Validation Accuracy over Time (512 LSTM Hidden Units)](plots/LEARNING_RATE_SERIES_HISTORY_VAL_ACCURACY.png "Validation Accuracy over Time (512 LSTM Hidden Units)")

#### Dropout
Dropout can prevent the model from overfitting. Interestingly, against our intuition, zero dropout performed best. It could be the case that LSTM networks do not overfit as fast as conventional neural networks.
![Plot: Validation Accuracy over Time](plots/LSTM_DROPOUT_SERIES_HISTORY_VAL_ACCURACY.png "Validation Accuracy over Time")


__Note__: Some of the plots already end after a few epochs. This is due to the early stopping feature implemented (training is stopped after accuracy improvement of less than 0.001 over 4 epochs).

## Paper & Additional Experiments Reproduction
Detail about the execution (i.e. all detailed parameters) can be found in the ![__SLURM script folder__](scripts). The stdout and stderr output of the executions can be found in the ![__logs__](logs) folder. The compiled and trained models can be found in the ![__models__](models) folder.  Furthermore, the reported accuracy and loss data can be found in the ![__results__](results) folder. 


## Computing Infrastructure
We used the following compute infrastructure to run the experiments:

  * __CPU__: AMD EPYC 7501 (64 cores / 128 threads)
  * __GPU__: 2x Nvidia V100 (32GB PCIe) \*
  * __Memory__: 512GB

\* the CRF layer does not have multi GPU support yet

## Runtime
The runtime for a single experiment on the compute infrastructure descibed above 
* data import
 * data loading: <5min
 * sequence padding: <5min
 * embedding:
   * glove: <5min
   * word2vec: <5min
   * elmo: >1h and huge memory consumption (to reduce this, we implemented batch-wise elmo embedding (parameter: batch_size_embedding) and load/store functionality to re-use the embedding for different experiments)
 * y mapping: <5min
* training:
  * model initialization: <5min
  * epoch: up to 3min, usually around 1.5-2min per epoch, maximum 40 epochs
* prediction and evaluation: <5min

__Total: 0.3-3h per experiment__


## Work Hours

* Preparation (reading paper, studying lstm, crf, word2vec, glove, elmo, ptb, ..): 10h
* Initial code implementation
  * dataset ptb: 5h
  * dataset brown: 3h
  * dataset conll200: 0.5h
  * embedding glove: 3h
  * embedding word2vec: 3h
  * embedding elmo: 6h
  * pos (import, y padding, y mapping, keras lstm crf model, eval function): 30h
   * rewrite from flat code block to class (incl argparse): 9h
   * add execution time reduction functionality (load/store embeddings, early stopping): 2h 
* Run experiments from paper (including bug fixes during this process): 5h
* Run additional experiments: 10h
* Write report: 15h
* Clean up code (add comments, function descriptions,..): 4h  

__Total: 100h__


## Evaluation

example command for the "Validation Accuracy over Time" plot:
```
python3 pos/evaluation.py --plot EPOCH_SERIES_COMBINED --dataset HISTORY_VAL_ACCURACY
```

```
usage: evaluation.py [-h]
                     [-d {DEV_ACCURACY,TEST_ACCURACY,HISTORY_ACCURACY,HISTORY_EPOCH,HISTORY_LOSS,HISTORY_VAL_ACCURACY,HISTORY_VAL_LOSS}]
                     [-p {LSTM_HIDDEN_UNITS_SERIES,LSTM_HIDDEN_UNITS_SERIES_COMBINED,EPOCH_SERIES,EPOCH_SERIES_COMBINED,LEARNING_RATE_SERIES,LSTM_DROPOUT_SERIES}]

optional arguments:
  -h, --help            show this help message and exit
  -d {DEV_ACCURACY,TEST_ACCURACY,HISTORY_ACCURACY,HISTORY_EPOCH,HISTORY_LOSS,HISTORY_VAL_ACCURACY,HISTORY_VAL_LOSS}, --dataset {DEV_ACCURACY,TEST_ACCURACY,HISTORY_ACCURACY,HISTORY_EPOCH,HISTORY_LOSS,HISTORY_VAL_ACCURACY,HISTORY_VAL_LOSS}
  -p {LSTM_HIDDEN_UNITS_SERIES,LSTM_HIDDEN_UNITS_SERIES_COMBINED,EPOCH_SERIES,EPOCH_SERIES_COMBINED,LEARNING_RATE_SERIES,LSTM_DROPOUT_SERIES}, --plot {LSTM_HIDDEN_UNITS_SERIES,LSTM_HIDDEN_UNITS_SERIES_COMBINED,EPOCH_SERIES,EPOCH_SERIES_COMBINED,LEARNING_RATE_SERIES,LSTM_DROPOUT_SERIES}

```

## Known Issues

1. Load/Save model: In order to lift the burden of re-training the model for a prediction task, we added the functionality of storing and loading the keras model from/to disk, including the trained weights. Unfortunately, this seems to be a known issue (see https://github.com/keras-team/keras/issues/4875) that the model weight are not loaded correctly. Therefore, and until this is fixed in the keras codebase, we have to re-train the model every time we want to perform predictions.

