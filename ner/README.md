## Code

### Usage
example command for experiment 1.1 from the paper:

```
python ner/train.py --train dataset/eng.train --dev dataset/eng.testa --test dataset/eng.testb --tag_scheme iobes --train_mode cased --dev_mode cased --zeros 0  --char_lstm_dim 25 --char_bidirect 1 --pre_emb models/glove.840B.300d.txt --word_dim 300 --word_lstm_dim 200 --crf 1 --use_gpu 1 --name model
```



### Dataset Split
eng.train from conll2003 is used for training, eng.testa and eng.testb for validation and testing


## Computing Infrastructure
We used the following compute infrastructure to run the experiments:

  * __CPU__: Intel(R) Core(TM) i7-6850K CPU  (12 cores / 144 threads )
  * __GPU__:  Nvidia 1080 GTX (16GB PCIe) \*
  * __Memory__: 64GB

\* the CRF layer does not have multi GPU support yet

## Runtime
The runtime for a single experiment on the compute infrastructure descibed above 
* data import
 * data loading: <2min
 * sequence padding: <2min
 * embeddings: < 5min
* training: 5hours
* prediction and evaluation: <5min

__Total: 5h per experiment__



## Evaluation

example command for the "Validation Accuracy over Time" plot:
```
python ner/eval.py --test dataset/eng.testb --case 0 --crf --use_gpu 1 --model_path models/model --map_path models/mapping.pkl 
```
###casing
0: cased
1: uncased
2: truecased