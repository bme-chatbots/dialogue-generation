#!/bin/bash

# @author:    Patrik Purgai
# @copyright: Copyright 2019, gpt2-chatbot
# @license:   MIT
# @email:     purgai.patrik@gmail.com
# @date:      2019.07.12.


DATA_DIR=$(dirname "$0")/../data/personachat

if [ ! -d "$DATA_DIR" ]; then
    python $(dirname "$0")/../data/prepare_personachat.py
fi

python $(dirname "$0")/../src/train.py --data_dir $DATA_DIR \
                                       --batch_size 4 \
                                       --max_size 20

