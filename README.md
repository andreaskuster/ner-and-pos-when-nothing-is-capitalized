# ner and pos when nothing is capitalized

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
| Experiment | Train data | Test data | accuracy | Avg | accuracy from the paper | Avg from the paper |
| --- | --- | --- | --- | --- | --- | --- |
| 1.1 | Cased | Cased |  |  | 97.85 | - |
| 1.2 | Cased | Uncased |  |  | 88.66 | 93.26 |
| 2 | Uncased | Uncased |  |  | 97.45 | 97.45 |
| 3.1 | Augment | Cased |  |  | 97.79 | - |
| 3.2 | Augment | Uncased |  |  | 97.35 | 97.57 |
| 3.5.1 | Half Mixed | Cased |  |  | 97.85 | - |
| 3.5.2 | Half Mixed | Uncased |  |  | 97.36 | 97.61 |
| 4 | Cased | Truecase |  |  | 95.21 | 95.21 |
| 5 | Truecase | Truecase |  |  | 97.38 | 97.38 |

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
