
# University of Washington CSE 517 A Wi 20: Natural Language Processing - Project
As part of the CSE517 NLP class at UW, we seek to reproduce the results from the paper __[ner and pos when nothing is capitalized](../master/papers/ner_and_pos_when_nothing_is_capitalized.pdf)__ , provide a working implementation, insight into hyperparameters as well as additional experiments.

[Project Proposal](../master/proposal/document.pdf)

[Final Paper](../master/final_paper/emnlp-ijcnlp-2019.pdf)

# ner and pos when nothing is capitalized  
For those languages which use it, capitalization is an important signal for the fundamental NLP tasks of Named Entity Recognition (NER) and Part of Speech (POS) tagging. In fact, it is such a strong signal that model performance on these tasks drops sharply in common lowercased scenarios, such as noisy web text or machine translation outputs. In this work, we perform a systematic analysis of solutions to this problem, modifying only the casing of the train or test data using lowercasing and truecasing methods. While prior work and first impressions might suggest training a caseless model, or using a truecaser at test time, we show that the most effective strategy is a concatenation of cased and lowercased training data, producing a single model with high performance on both cased and uncased text. As shown in our experiments, this result holds across tasks and input representations. Finally, we show that our proposed solution gives an 8\% F1 improvement in mention detection on noisy out-of-domain Twitter data.

[Paper](../master/papers/ner_and_pos_when_nothing_is_capitalized.pdf)


# Findings

## Truecasing experiment

### Paper Reproduction (BiLSTM on Wikipedia)

#### Hypothesis
We expect to get similar results to those described in the paper.

#### Comparsion
| Test Set   | F1 Score (OOV when creating dictionary) | F1 Score (OOV at read) | F1 Score from the paper |
|:--         | :-:                                     |:-:                     |:-:                      |
| Wikipedia  | 92.71                                   | 92.65                  | 93.01                   |
| ConLL Train| 65.32                                   | 66.03                  | 78.85                   |
| ConLL Test | 63.28                                   | 63.49                  | 77.35                   |
| PTB 01-18  | 78.73                                   | 78.53                  | 86.91                   |
| PTB 22-24  | 78.69                                   | 78.47                  | 86.22                   |

While the paper does not provide a lot of detail on implementation, we were able to reproduce results shown in it closely enough to be confident in our implementation.

In particular we used an Adam optimizer (on default settings) and standard bidirectional, two layered LSTM with 300 hidden units. We used a batch size of 100, as metioned in [[Susanto]](./papers/trucasing.pdf). Then the model was trained for 30 epochs, and model with the smallest loss on validation set is chosen. That models loss on test set is reported above, as well as that model is made available for both NER and POS experiments.

Instead of using pre-trained encodings we are learning our own (since number of unique characters in train set is around 50). Due to the fact that both validation and test sets contain tokens not included in train set we are forced to use OOV tokens. Each time a sentence is read each character has a 0.5% chance of becoming an OOV token.

Lastly, to greatly increase training speed, all of our sentences are padded. Each padded has a target 0 (i.e. should not be capitalized), which counts towards training loss. However, padding does not count for either validation loss used to choose model epoch, and is not included in F1 score reported above.

#### Conclusion
On wikipedia dataset which we trained truecaser on we got performence similar to the one reported in the paper. The [original procedure](https://github.com/raymondhs/char-rnn-truecase) differed slightly from our version (mostly in a sense that it used char-rnn, while we used vanilla PyTorch), which can explain slight differences in results. However, achieving similar performence was not problematic with all information provided in the paper.

## POS Experiment

### Paper Reproduction (BiLSTM-CRF+POS+ELMo on PTB)

#### Hypothesis

1. Training on cased data does not perform well on uncased data, while training on uncased data performs well on uncased data.
2. Training on a concatenated dataset of uncased and cased data performs well on cased and uncased data. It does so, not due to the larger dataset, but rather works as good if we (randomly) lowercase 50% of the dataset.
3. Trying to solve the problem of (1) by truecasing the lowercased test data does not perform well, but it does perform well if the training data has been lowercased and truecased too.

We expected to get similar results to those described in the paper.

#### Comparison
| Experiment | Train Data | Test Data | Accuracy | Avg | Accuracy (Paper) | Avg (Paper) |
| --- | --- | --- | --- | --- | --- | --- |
| 1.1 | Cased | Cased | __97.30__ | - | __97.85__ | - |
| 1.2 | Cased | Uncased | 88.29 | 92.78 | 88.66 | 93.26 |
|  |  |  |  |  |  |  |
| 2 | Uncased | Uncased | __96.51__ | 96.51 | __97.45__ | 97.45 |
|  |  |  |  |  |  |  |
| 3.1 | C+U | Cased | 97.51 | - | 97.79 | - |
| 3.2 | C+U | Uncased | 96.59 | 97.05 | 97.35 | 97.57 |
|  |  |  |  |  |  |  |
| 3.5.1 | Half Mixed | Cased | __97.12__ | - | __97.85__ | - |
| 3.5.2 | Half Mixed | Uncased | 96.19 | __96.66__ | 97.36 | __97.61__ |
|  |  |  |  |  |  |  |
| 4 | Cased | Truecase | 95.04 | 95.04 | 95.21 | 95.21 |
|  |  |  |  |  |  |  |
| 5 | Truecase | Truecase | 96.61 | 96.61 | 97.38 | 97.38 |

#### Model Characteristics

__Train/Test/Dev data__:
* Train: Penn Treebank section 0-18 (usage: training)
* Dev: Penn TreeBank section 19-21 (usage: validation, i.e. early stopping and hyperparameter search)
* Test: Penn TreeBank section 22-24 (usage: reporting accuracy)

__Pre-Processing__: Depending on the experiment, we either used the imported data as-is (cased), lower-cased it (lowercase) or lower-cased it and then true-case predicted it (truecase). Furthermore, we also combinded the lowercased and cased dataset (C+U) and randomly lowercased 50% of the dataset (half mixed).

__Padding__: Pad all sentence to the lenght of the longest sentence using "NULL" charakters. (Note: The reported accuracy values are true accuracies, i.e. with padding removed. If we would not do this and the dataset contains very short and very long sentences, the accuracy of a prediction with only "NULL" characters for the whole sequence of the short sentence would be very high, even though the predictor is very bad.)

__Embedding__: ELMo word embedding, vector size: 1024

__LSTM Model__:
We used keras for the neural network implementation with the following configuration:

* __Sequential Model__:
 * __Input Layer__: 
   * BiLSTM Layer:
     * input shape: (max_sentence_lengt, 1024)
     * hidden units: 512
     * lstm dropout: 0.0
     * lstm recurrent dropout: 0.0
 * __Hidden Layer(s)__:
   * TimeDistributed Dense Layer
     * shape: num_labels
     * activation function: rectified linear unit
 * __Output Layer__:
   * CRF Layer:
     * shape: num_labels
* __Training__:
  * __Solver__:
    * Adam
    * learning rate: 0.001
  * __Epochs__:
    * max: 40 epochs
    * early stopping: stops after the validation accuracy did not increase by more than 0.001 over 4 conecutive epochs
   
__Evaluation__: After training, we predicted the label of the test set using the trained model, removed the padding and computed the accuracy (number of correctly predicted labels / number of labels).

__Note__: All the hyperparameters reported above are a result of the hyperparameter grid-search done previous to this experiment evaluation. Details about this and additional information about the code usage can be found [__here__](pos/README.md).


#### Conclusion
Even though we haven't seen any implementation details (except those described in the paper text), we reported the exact same accuracy scores (+/- 0.8%) and can confirm the hypothesis (1., 2., 3.) described above.



### Additional Experiments
The aim of the additional experiments is to find out if the hypothesis from the paper is more generally applicable. Therefore, we run the same experiments on LSTM models with different word embeddings (word2vec, glove, elmo) and without the CRF layer.
Furthermore, we extended the tests to different datasets, namely the Brown corpus, the CoNLL2000 corpus and a subset of the PTB corpus (train: section 0-4, dev: section 5-6, test: section 7-8). 


#### Hypothesis

1. For the coparison of the different embeddings (word2vec, glove, elmo) we expect elmo to perform better than the other two. Firstly, because of the better perfomance we concluded during the lecture (for a well trained elmo model) and secondly because it was the choice for this paper.
2. We expect the ELMo model with CRF layer to outperform the one without due to the findings from the paper linked below.
3. We expect to be able to conclude the hypothesis from the original experiments for the Twitter dataset as well

POS on penn treebank reduced dataset, brown and CoNLL2000 dataset, word2vec, glove and elmo, with/without CRF layer (additional experiments)

| Train Data | Test Data | Accuracy Word2vec CRF | Accuracy GloVe CRF | Accuracy ELMo | Accuracy ELMo CRF (paper experiment) |
| --- | --- | --- | --- | --- | --- | 
| __PTB Dataset__ |  |  |  |  |  | 
| Cased | Cased | 88.80 | 95.90 | 97.19 |  97.30 | 
| Cased | Uncased | 78.63 | 86.11 | 88.57 | 88.29 |
|  |  |  |  |  |  
| Uncased | Uncased | 80.97 | 94.97 | 96.52 | 96.51 |
|  |  |  |  |  | 
| C+U | Cased | 85.62 | 96.88 | 97.44 | 97.51 |
| C+U | Uncased | 86.67 | 95.84 | 96.60 | 96.59 |
|  |  |  |  |  | 
| Half Mixed | Cased | 87.45 | 95.79 | 97.30 |  97.12 |
| Half Mixed | Uncased | 82.86 | 94.90 | 96.36 | 96.19 |
|  |  |  |  |  | 
| Cased | Truecase | 85.74 | 93.82 | 94.78 | 95.04 |
| Truecase | Truecase | 86.64 | 95.20 | 96.56 | 96.61 |
|  |  |  |  |  |


| Train Data | Test Data | Accuracy ELMo CRF |
| --- | --- | --- |
| __PTB Reduced Dataset__ |  |  |
| Cased | Cased | 96.35 |
| Cased | Uncased | 88.38 |
|  |  |  |  |  |  |  | 
| Uncased | Uncased | 95.48 |
|  |  |  |  |  |  |  |
| C+U | Cased | 96.70 |
| C+U | Uncased | 95.73 |
|  |  |  |  |  |  |  |
| Half Mixed | Cased | 96.34 |
| Half Mixed  | Uncased | 95.08 |
|  |  |  |  |  |  |  |
| Cased | Truecase | 94.62 |
| Truecase | Truecase | 95.35 |
|  |  |  |  |  |  |  |
| __Brown__ |  |  |  |  |  |  | 
| Cased | Cased | 95.69 |
| Cased | Uncased | 83.30 |
|  |  |  |  |  |  |  | 
| Uncased | Uncased | 92.91 |
|  |  |  |  |  |  |  |
| C+U | Cased | 97.11 |
| C+U | Uncased | 95.83 |
|  |  |  |  |  |  |  |
| Half Mixed | Cased | 95.28 |
| Half Mixed  | Uncased | 92.56 |
|  |  |  |  |  |  |  |
| Cased | Truecase | 92.11 |
| Truecase | Truecase | 92.62 |
|  |  |  |  |  |  |  |
| __CoNLL 2000__ |  |  |  |  |  |  | 
| Cased | Cased | 97.80 |
| Cased | Uncased | 87.91 |
|  |  |  |  |  |  |  | 
| Uncased | Uncased | 96.83 |
|  |  |  |  |  |  |  |
| C+U | Cased | 99.00 |
| C+U | Uncased | 99.46 |
|  |  |  |  |  |  |  |
| Half Mixed | Cased | 97.65 |
| Half Mixed  | Uncased | 96.66 |
|  |  |  |  |  |  |  |
| Cased | Truecase | 95.40 |
| Truecase | Truecase | 96.79 |
|  |  |  |  |  |  |  |

#### Model Characteristics

In order to compare the outcome from the additional experiments to those from the paper reproduction, we used the exact same model, except the component specified in the table (i.e. ELMo embedding replaced with GloVe embedding).

#### Conclusion
1. ELMo indeed outperforms word2vec by ~10% and GloVe by ~2%.
2. The accuracy with and without CRF layer are rougly the same (+/-0.3% depending on the testcase). Considering the extra effort of addin the (non-standard) keras layer, the lack of multi gpu support, we conclude that we could would probably be better off withouth the CRF layer.
3.The hypothesis (1.,2.,3.) from the paper hold true (relative accuracy difference) for the reduced ptb dastaset, as well as the other datasets (brown and conll2000). Depending on the dataset, the absolute accuracy values differ i.e. the performance of the pos tagger for CoNLL2000 reaches >99% for the C+U experiment.

### Implementation details
Additional details about the part-of-speech tagging part of the paper can be found in the separate [__pos/README.md__](pos/README.md).

## NER experiment
#### Hypothesis

1. Training on cased data does not perform well on uncased data, while training on uncased data performs well on uncased data.
2. Training on a concatenated dataset of uncased and cased data performs well on cased and uncased data. It does so, not due to the larger dataset, but rather works as good if we (randomly) lowercase 50% of the dataset.
3. Trying to solve the problem of (1) by truecasing the lowercased test data does not perform well, but it does perform well if the training data has been lowercased and truecased too.
#### Model
BiLSTM-CRF using Glove + character embeddings trained on CoNLL
25 character embeddings trained using a BiLSTM + 300 pre-trained glove embeddings
BiLSTM Layer with 200 hidden layer dimension (Drop out of 0.5)
Highway layer
Output CRF layer
initial learning rate of 0.15 with adam
stopping criterion when the F-score doesn't improve over iterations

The model is validated on eng_testa and tested on end_testb


| Experiment | Train data | Test data | F1 Score | Avg | F1 Score from the paper   | Avg from the paper |
| ---   | --- | --- | --- | --- | --- | --- |
| 1.1   | Cased | Cased | 90.63 | - | 92.45| - |
| 1.2   | Cased | Uncased | 81.47 | 86.05 | 34.46 | 63.46 |
| 2     | Uncased | Uncased | 89.72 | 89.72 | 89.32 | 89.32 |
| 3.1   | Augment | Cased | 90.10 | - | 91.67| - |
| 3.2   | Augment | Uncased |88.65 | 89.38 | 89.31 | 90.49 |
| 3.5.1 | Half Mixed | Cased | 90.84 |  | 91.68 | - |
| 3.5.2 | Half Mixed | Uncased |  89.54 | 90.19 | 89.05| 90.37 |
| 4     | Cased | Truecase | 80.89 | 80.89 | 82.93 | 82.93 |
| 5     | Truecase | Truecase | 88.43 | 88.43 | 90.25 | 90.25 |

BiLSTM-CRF using Glove + character embeddings trained on CoNLL tested on Twitter Corpus
| Experiment | Train data | F1 Score |  F1 Score from the paper | 
| --- | --- | --- | --- |
| 1.1 | Cased |33.24   | 58.63| 
| 2 | Uncased |  14.54 | 53.13 | 
| 3 | Augment | 31.31  | 66.14| 
| 3.5 | Half Mixed | 32.94    | 64.69 |
| 4 | Cased-Truecase | 23.45  | 58.22 | 
| 5 | Truecase |  29.19  | 62.66 | 

 
# Implications

The most interesting difference in the cased variant, where there is a above 30\% gap between our and the original implementation. After closer investigation we discovered that the reason for it is huge different in its performance on uncased data (81.47 in our implementation vs 34.46 in original one). We do not have a firm intuition on why this is happening. However, it might be the case that models trained on cased dataset are highly unstable when tested on uncased data.

Overall, however we can see that relative performance of results is similar, and mixing cased and uncased data provides the best performance with our implementation. Because of this we believe our results support second hypothesis of the paper.

Our model did much worse than the original one. This is very counterintuitive, when we consider the fact that in the original dataset, our cased experiment generalized much better. Overall, we cannot support the thrid hypothesis from original paper.


# Resources

## Papers
[ner and pos when nothing is capitalized](../master/papers/ner_and_pos_when_nothing_is_capitalized.pdf)

[ner and pos when nothing is capitalized poster](../master/papers/poster.pdf)

[Bidirectional LSTM-CRF Models for Sequence Tagging](../master/papers/bilstm_crf.pdf)

[Deep contextualized word representations](../master/papers/elmo.pdf)

[GloVe: Global Vectors for Word Representation](../master/papers/glove.pdf)

[Learning to Capitalize with Character-Level Recurrent Neural Networks: An Empirical Study](./papers/truecasing.pdf)
