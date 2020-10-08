#!/bin/bash

export TOKENIZERS_PARALLELISM=true

python run_training.py hydra/job_logging=disabled
