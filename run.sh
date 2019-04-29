#!/bin/bash

RUN_MODE=${1:-"train"}
DATA_DIR=~/data/nlp/nmt
MODEL_DIR=$(dirname "$0")/model

mkdir -p $MODEL_DIR
mkdir -p $DATA_DIR

if [ $RUN_MODE == "train" ]
then
    python $(dirname "$0")/src/train.py --model_dir $MODEL_DIR \
                                        --data_dir $DATA_DIR
elif [ $RUN_MODE == "eval" ]
then
    python $(dirname "$0")/src/eval.py --model_dir $MODEL_DIR
elif [ $RUN_MODE == "reinforce" ]
then
    python $(dirname "$0")/src/reinforce.py --model_dir $MODEL_DIR
else
    echo "Invalid run mode command $RUN_MODE."
fi
