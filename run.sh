#!/bin/bash

MODEL_NAME=${1:-"xlnet"}
DATA_NAME=${2:-"dailydialog"}
RUN_MODE=${3:-"train"}
DATA_DIR=${4:-$(dirname "$0")/data}
DOWNLOAD_DIR=${5:-$DATA_DIR}
MODEL_DIR=$(dirname "$0")/model

mkdir -p $MODEL_DIR
mkdir -p $DATA_DIR
mkdir -p $DOWNLOAD_DIR

if [ $RUN_MODE == "train" ]
then
    python $(dirname "$0")/src/train.py --model_name $MODEL_NAME \
                                        --data_name $DATA_NAME \
                                        --model_dir $MODEL_DIR \
                                        --data_dir $DATA_DIR \
                                        --download_dir $DOWNLOAD_DIR
elif [ $RUN_MODE == "eval" ]
then
    python $(dirname "$0")/src/eval.py --model_name $MODEL_NAME \
                                       --data_name $DATA_NAME \
                                       --model_dir $MODEL_DIR \
                                       --data_dir $DATA_DIR
else
    echo "Invalid run mode `$RUN_MODE`."
fi

