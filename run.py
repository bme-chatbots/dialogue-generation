"""
@author:    Patrik Purgai
@copyright: Copyright 2019, dialogue-generation
@license:   MIT
@email:     purgai.patrik@gmail.com
@date:      2019.08.22.
"""

import sys
import torch
import argparse
import os

from argparse import REMAINDER

from src.train import (
    main as train,
    setup_train_args)

from src.eval import (
    main as evaluate,
    setup_eval_args)

from src.data import (
    setup_data_args,
    create_dataset)

from src.model import (
    setup_model_args)

from torch.multiprocessing import spawn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--mode',
        type=str,
        default='train')
    parser.add_argument(
        '--num_devices', 
        type=int, 
        default=torch.cuda.device_count(),
        help='The number of processes to launch on each node.')
    parser.add_argument(
        '--master_addr', 
        default='127.0.0.1', 
        type=str,
        help='Master node (rank 0) address')
    parser.add_argument(
        '--master_port', 
        default='29500', 
        type=str,
        help='Master node (rank 0) free port')
    
    setup_train_args(parser)
    setup_eval_args(parser)
    setup_data_args(parser)
    setup_model_args(parser)

    args = parser.parse_args()

    args.distributed = args.num_devices > 1 and args.cuda

    if args.mode == 'train':
        # creating dataset so it will already be
        # downloaded in case of multi gpu training
        create_dataset(args=args)

        if args.distributed:
            world_size = str(args.num_devices)

            os.environ['WORLD_SIZE'] = world_size
            os.environ['MASTER_ADDR'] = args.master_addr
            os.environ['MASTER_PORT'] = args.master_port

            nprocs = args.num_devices
            spawn(train, args=(args, ), nprocs=nprocs)

        else:
            train(rank=-1, args=args)

    else:
        evaluate(args)


if __name__ == '__main__':
    main()
