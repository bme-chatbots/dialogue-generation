#!/bin/bash

MODEL_NAME="xlnet"
DATA_NAME="dailydialog"
RUN_MODE="train"
DATA_DIR=$(dirname "$0")/data
DOWNLOAD_DIR=$DATA_DIR
MODEL_DIR=$(dirname "$0")/model

while [ "$1" != "" ]; do
    case $1 in
        -m | --model_name )     shift
                                MODEL_NAME=$1
                                ;;
        -d | --data_name )      shift
                                DATA_NAME=$1
                                ;;
        -r | --run_mode )       shift
                                RUN_MODE=$1
                                ;;
        --model_dir )           shift
                                MODEL_DIR=$1
                                ;;    
        --data_dir )            shift
                                DATA_DIR=$1
                                ;;  
        --download_dir )        shift
                                DOWNLOAD_DIR=$1
                                ;;                                
        -h | --help )           usage
                                exit
                                ;;
        * )                     usage
                                exit 1
    esac
    shift
done

mkdir -p $MODEL_DIR
mkdir -p $DATA_DIR
mkdir -p $DOWNLOAD_DIR

if [ $RUN_MODE == "train" ]
then
    python -m src.train --model_name $MODEL_NAME \
                        --data_name $DATA_NAME \
                        --model_dir $MODEL_DIR \
                        --data_dir $DATA_DIR \
                        --download_dir $DOWNLOAD_DIR
elif [ $RUN_MODE == "eval" ]
then
    python -m src.eval --model_name $MODEL_NAME \
                       --data_name $DATA_NAME \
                       --model_dir $MODEL_DIR \
                       --data_dir $DATA_DIR
else
    echo "Invalid run mode `$RUN_MODE`."
fi
