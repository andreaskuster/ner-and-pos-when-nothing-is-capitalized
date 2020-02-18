# ner and pos when nothing is capitalized  
For those languages which use it, capitalization is an important signal for the fundamental NLP tasks of Named Entity Recognition (NER) and Part of Speech (POS) tagging. In fact, it is such a strong signal that model performance on these tasks drops sharply in common lowercased scenarios, such as noisy web text or machine translation outputs. In this work, we perform a systematic analysis of solutions to this problem, modifying only the casing of the train or test data using lowercasing and truecasing methods. While prior work and first impressions might suggest training a caseless model, or using a truecaser at test time, we show that the most effective strategy is a concatenation of cased and lowercased training data, producing a single model with high performance on both cased and uncased text. As shown in our experiments, this result holds across tasks and input representations. Finally, we show that our proposed solution gives an 8\% F1 improvement in mention detection on noisy out-of-domain Twitter data.

[Paper](../master/papers/ner_and_pos_when_nothing_is_capitalized.pdf)




## Truecasing experiment

| Truecaser trained on Wikipedia corpus | Test Set | F1 Score | F1 Score from the paper |
|---|---|---|---|
|BiLSTM| Wikipedia | 92.65| 93.01|
|| ConLL Train| | 78.85|
|| ConLL Test | | 77.35|
|| PTB 01-18 | | 86.91|
|| PTB 22-24 | | 86.22|

## POS experiment
BiLSTM-CRF on PTB (paper reproduction)
| Experiment | Train data | Test data | accuracy | Avg | accuracy from the paper | Avg from the paper | Code |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1.1 | Cased | Cased |  |  | 97.85 | - |  |
| 1.2 | Cased | Uncased |  |  | 88.66 | 93.26 |  |
| 2 | Uncased | Uncased |  |  | 97.45 | 97.45 |  |
| 3.1 | Augment | Cased |  |  | 97.79 | - |  |
| 3.2 | Augment | Uncased |  |  | 97.35 | 97.57 |  |
| 3.5.1 | Half Mixed | Cased |  |  | 97.85 | - |  |
| 3.5.2 | Half Mixed | Uncased |  |  | 97.36 | 97.61 |  |
| 4 | Cased | Truecase |  |  | 95.21 | 95.21 |  |
| 5 | Truecase | Truecase |  |  | 97.38 | 97.38 |  |

POS on brown and CoNLL2000 dataset, word2vec, glove and elmo, with/without CRF layer (additional experiments)
| Train data | Test data | accuracy w2v | accuracy w2v crf | accuracy glove | accuracy glove crf | accuracy elmo | accuracy elmo crf| Code |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| __Brown__ |  |  |  |  |  |  |  | 
| Cased | Cased |  |  |  |  |  |  |  |  |
| Cased | Uncased |  |  |  |  |  |  |  |  |
| Uncased | Uncased |  |  |  |  |  |  |  |  |
| Cased | Truecase |  |  |  |  |  |  |  |  |
| __CoNLL 2000__ |  |  |  |  |  |  |  | 
| Cased | Cased |  |  |  |  |  |  |  |  |
| Cased | Uncased |  |  |  |  |  |  |  |  |
| Uncased | Uncased |  |  |  |  |  |  |  |  |
| Cased | Truecase |  |  |  |  |  |  |  |  |

## NER experiment
BiLSTM-CRF trained on CoNLL
| Experiment | Train data | Test data | F1 Score | Avg | F1 Score from the paper | Avg from the paper |
| --- | --- | --- | --- | --- | --- | --- |
| 1.1 | Cased | Cased |  |  | 92.45| - |
| 1.2 | Cased | Uncased |  |  | 34.46 | 63.46 |
| 2 | Uncased | Uncased |  |  | 89.32 | 89.32 |
| 3.1 | Augment | Cased |  |  | 91.67| - |
| 3.2 | Augment | Uncased |  |  | 89.31 | 90.49 |
| 3.5.1 | Half Mixed | Cased |  |  | 91.68 | - |
| 3.5.2 | Half Mixed | Uncased |  |  | 89.05| 90.37 |
| 4 | Cased | Truecase |  |  | 82.93 | 82.93 |
| 5 | Truecase | Truecase |  |  | 90.25 | 90.25 |

BiLSTM-CRF trained on CoNLL tested on Twitter Corpus
| Experiment | Train data | F1 Score |  F1 Score from the paper | 
| --- | --- | --- | --- |
| 1.1 | Cased |   | 58.63| 
| 2 | Uncased |   | 53.13 | 
| 3 | Augment |   | 66.14| 
| 3.5 | Half Mixed |     | 64.69 |
| 4 | Cased |   | 58.22 | 
| 5 | Truecase |    | 62.66 | 
