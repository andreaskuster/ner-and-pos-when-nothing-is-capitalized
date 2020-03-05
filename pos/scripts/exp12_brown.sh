#!/usr/bin/env bash
#SBATCH --mem 500G
#SBATCH -c 128
#SBATCH -o /users2/kustera/log/nlp_out_exp12_brown.txt
#SBATCH -e /users2/kustera/log/nlp_err_exp12_brown.txt
#SBATCH -t 4:00:00
#SBATCH --partition amdv100

module load python/3.7.2
module load CMake
module load GCC
source ~/.bashrc

conda activate dev
cd ~/uw-nlp && python3 pos/pos.py --model BILSTM_CRF --dataset BROWN --traincasetype CASED --devcasetype CASED --testcasetype UNCASED --embedding ELMO --batchsize 1024 --epochs 40 --learningrate 1e-3 --lstmhiddenunits 512 --lstmdropout 0.0 --lstmrecdropout 0.0 --numgpus 2

