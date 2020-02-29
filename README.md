
# University of Washington CSE 517 A Wi 20: Natural Language Processing - Project
As part of the CSE517 NLP class at UW, we seek to reproduce the results from the paper __[ner and pos when nothing is capitalized](../master/papers/ner_and_pos_when_nothing_is_capitalized.pdf)__ , provide a working implementation, insight into hyperparameters as well as additional experiments.

[Project Proposal](../master/proposal/document.pdf)

# ner and pos when nothing is capitalized  
For those languages which use it, capitalization is an important signal for the fundamental NLP tasks of Named Entity Recognition (NER) and Part of Speech (POS) tagging. In fact, it is such a strong signal that model performance on these tasks drops sharply in common lowercased scenarios, such as noisy web text or machine translation outputs. In this work, we perform a systematic analysis of solutions to this problem, modifying only the casing of the train or test data using lowercasing and truecasing methods. While prior work and first impressions might suggest training a caseless model, or using a truecaser at test time, we show that the most effective strategy is a concatenation of cased and lowercased training data, producing a single model with high performance on both cased and uncased text. As shown in our experiments, this result holds across tasks and input representations. Finally, we show that our proposed solution gives an 8\% F1 improvement in mention detection on noisy out-of-domain Twitter data.

[Paper](../master/papers/ner_and_pos_when_nothing_is_capitalized.pdf)

# Code
[https://github.com/andreaskuster/uw-nlp](https://github.com/andreaskuster/uw-nlp)

# Findings

## Truecasing experiment

### Paper Reproduction (BiLSTM on Wikipedia)

#### Hypothesis
We expect to get similar results to those described in the paper.

#### Comparsion
| Test Set   | F1 Score from our code | F1 Score from the paper |
|:--         |:-:                     |:-:                      |
| Wikipedia  | 92.65                  | 93.01                   |
| ConLL Train| 66.03                  | 78.85                   |
| ConLL Test | 63.49                  | 77.35                   |
| PTB 01-18  | 78.53                  | 86.91                   |
| PTB 22-24  | 78.47                  | 86.22                   |

While the paper does not provide a lot of detail on implementation, we were able to reproduce results shown in it closely enough to be confident in our implementation.

In particular we used an Adam optimizer (on default settings) and standard bidirectional, two layered LSTM with 300 hidden units. We used a batch size of 100, as metioned in [[Susanto]](./papers/trucasing.pdf). Then the model was trained for 30 epochs, and model with the smallest loss on validation set is chosen. That models loss on test set is reported above, as well as that model is made available for both NER and POS experiments.

Instead of using pre-trained encodings we are learning our own (since number of unique characters in train set is around 50). Due to the fact that both validation and test sets contain tokens not included in train set we are forced to use OOV tokens. Each time a sentence is read each character has a 0.5% chance of becoming an OOV token.

Lastly, to greatly increase training speed, all of our sentences are padded. Each padded has a target 0 (i.e. should not be capitalized), which counts towards training loss. However, padding does not count for either validation loss used to choose model epoch, and is not included in F1 score reported above.

#### Conclusion
On wikipedia dataset which we trained truecaser on we got performence similar to the one reported in the paper. The [original procedure](https://github.com/raymondhs/char-rnn-truecase) differed slightly from our version (mostly in a sense that it used char-rnn, while we used vanilla PyTorch), which can explain slight differences in results. However, achieving similar performence was not problematic with all information provided in the paper.

<b>TODO</b>: Summarize for other datasets.

## POS experiment

### Paper Reproduction (BiLSTM-CRF+POS+ELMo on PTB)

#### Hypothesis
We expect to get similar results to those described in the paper.

#### Comparison
| Experiment | Train Data | Test Data | Accuracy | Avg | Accuracy (Paper) | Avg (Paper) |
| --- | --- | --- | --- | --- | --- | --- |
| 1.1 | Cased | Cased | 97.30 | - | 97.85 | - |
| 1.2 | Cased | Uncased | 88.29 | 92.78 | 88.66 | 93.26 |
|  |  |  |  |  |  |  |
| 2 | Uncased | Uncased | 96.51 | 96.51 | 97.45 | 97.45 |
|  |  |  |  |  |  |  |
| 3.1 | C+U | Cased | 97.51 | - | 97.79 | - |
| 3.2 | C+U | Uncased | 96.59 | 97.05 | 97.35 | 97.57 |
|  |  |  |  |  |  |  |
| 3.5.1 | Half Mixed | Cased | 97.12 | - | 97.85 | - |
| 3.5.2 | Half Mixed | Uncased | 96.19 | 96.66 | 97.36 | 97.61 |
|  |  |  |  |  |  |  |
| 4 | Cased | Truecase | 95.04 | 95.04 | 95.21 | 95.21 |
|  |  |  |  |  |  |  |
| 5 | Truecase | Truecase | 96.61 | 96.61 | 97.38 | 97.38 |

#### Model Characteristics

Train/Test/Dev data: TODO

Pre-Processing: TODO

Padding: TODO

Embedding: TODO

LSTM Model:
* description of layers: TODO
* hyperparameters (#units, batch size, epochs, solver,..) including hyperparameter search (and graph of it) : TODO
* graphs: learning rate for different hyperparemeters

Evaluation: TODO

Computational Requirements: HW / average runtime, #trials, #GPU hours

#### Conclusion
TODO



### Additional Experiments
The goal of the additional experiments is to find out if the hypothesis from the paper is more generally applicable on different datasets, embeddings and LSTM models. Therefore, we run the tests on the brown and CoNLL2000 corpus, using word2vec, glove and elmo embeddings, and different LSTM models.

#### Hypothesis
We expect to get similar results to those described in the paper.

#### Comparison
POS on penn treebank reduced dataset, brown and CoNLL2000 dataset, word2vec, glove and elmo, with/without CRF layer (additional experiments)
| Train data | Test data | accuracy w2v | accuracy w2v crf | accuracy glove | accuracy glove crf | accuracy elmo | accuracy elmo crf| Code |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| __PTB Dataset__ |  |  |  |  |  |  |  | 
| Cased | Cased | TODO | TODO | TODO | TODO | TODO | TODO | TODO | 97.47 |
| Cased | Uncased | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| Uncased | Uncased | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| Cased | Truecase | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| __PTB Reduced Dataset__ |  |  |  |  |  |  |  | 
| Cased | Cased | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| Cased | Uncased | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| Uncased | Uncased | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| Cased | Truecase | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| __Brown__ |  |  |  |  |  |  |  | 
| Cased | Cased | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| Cased | Uncased | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| Uncased | Uncased | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| Cased | Truecase | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| __CoNLL 2000__ |  |  |  |  |  |  |  | 
| Cased | Cased | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| Cased | Uncased | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| Uncased | Uncased | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| Cased | Truecase | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |


#### Model Characteristics

Train/Test/Dev data: TODO

Pre-Processing: TODO

Padding: TODO

Embedding: TODO

LSTM Model:
* description of layers: TODO
* hyperparameters (#units, batch size, epochs, solver,..) including hyperparameter search (and graph of it) : TODO
* graphs: learning rate for different hyperparemeters

Evaluation: TODO

Computational Requirements: HW / average runtime, #trials, #GPU hours

#### Conclusion
TODO


## NER experiment
BiLSTM-CRF using ELMo + Glove + character embeddings trained on CoNLL
| Experiment | Train data | Test data | F1 Score | Avg | F1 Score from the paper   | Avg from the paper |
| ---   | --- | --- | --- | --- | --- | --- |
| 1.1   | Cased | Cased |  |  | 92.45| - |
| 1.2   | Cased | Uncased |  |  | 34.46 | 63.46 |
| 2     | Uncased | Uncased |  |  | 89.32 | 89.32 |
| 3.1   | Augment | Cased |  |  | 91.67| - |
| 3.2   | Augment | Uncased |  |  | 89.31 | 90.49 |
| 3.5.1 | Half Mixed | Cased |  |  | 91.68 | - |
| 3.5.2 | Half Mixed | Uncased |  |  | 89.05| 90.37 |
| 4     | Cased | Truecase |  |  | 82.93 | 82.93 |
| 5     | Truecase | Truecase |  |  | 90.25 | 90.25 |

BiLSTM-CRF using ELMo + Glove + character embeddings trained on CoNLL tested on Twitter Corpus
| Experiment | Train data | F1 Score |  F1 Score from the paper | 
| --- | --- | --- | --- |
| 1.1 | Cased |   | 58.63| 
| 2 | Uncased |   | 53.13 | 
| 3 | Augment |   | 66.14| 
| 3.5 | Half Mixed |     | 64.69 |
| 4 | Cased |   | 58.22 | 
| 5 | Truecase |    | 62.66 | 

 
### Additional Experiments
We run the same tests on Groningen Meaning Bank (GMB) dataset using the same or adjusted model and embeddings.

#### Hypothesis
We expect to get similar results to those described in the paper.

#### Comparison
| Experiment | Train data | F1 Score |
| --- | --- | --- | 
| 1.1 | Cased |   | 
| 2 | Uncased |   | 
| 3 | Augment |   | 
| 3.5 | Half Mixed |     | 
| 4 | Cased |   | 
| 5 | Truecase |    | 


#### Conclusion
TODO

# Implications

Description if the results were reproducible, what was not, ... TODO

# Resources

## Papers
[ner and pos when nothing is capitalized](../master/papers/ner_and_pos_when_nothing_is_capitalized.pdf)

[ner and pos when nothing is capitalized poster](../master/papers/poster.pdf)

[Bidirectional LSTM-CRF Models for Sequence Tagging](../master/papers/bilstm_crf.pdf)

[Deep contextualized word representations](../master/papers/elmo.pdf)

[GloVe: Global Vectors for Word Representation](../master/papers/glove.pdf)

[Learning to Capitalize with Character-Level Recurrent Neural Networks: An Empirical Study](./papers/truecasing.pdf)
