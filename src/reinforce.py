"""

@author:    Patrik Purgai
@copyright: Copyright 2019, chatbot
@license:   MIT
@email:     purgai.patrik@gmail.com
@date:      2019.04.04.
"""

# pylint: disable=import-error
# pylint: disable=no-name-in-module
# pylint: disable=no-member
# pylint: disable=not-callable

import argparse
import torch
import random
import os

import numpy as np

from os.path import exists, join, dirname, abspath
from torch.optim import Adam

from datetime import datetime
from tqdm import tqdm
from collections import OrderedDict

from data import (
    create_datasets, 
    setup_data_args, 
    get_special_indices)

from model import (
    create_model, 
    create_criterion,
    setup_model_args)

from beam import (
    beam_search, 
    setup_beam_args)

from train import (
    eval_step,
    train_step,
    create_optimizer,
    create_criterion)


PROJECT_DIR = join(abspath(dirname(__file__)), '..')


def setup_train_args():
    """
    Sets up the training arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Maximum number of epochs for training.')
    parser.add_argument(
        '--cuda',
        type=bool,
        default=torch.cuda.is_available(),
        help='Device for training.')
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1e-3,
        help='Learning rate for the model.')
    parser.add_argument(
        '--patience',
        type=int,
        default=10,
        help='Allowed number of epochs without progress.')
    parser.add_argument(
        '--model_dir',
        type=str,
        default=join(PROJECT_DIR, 'model.{}'.format(
            datetime.today().strftime('%j%H%m'))),
        help='Path of the model checkpoints.')

    setup_data_args(parser)
    setup_model_args(parser)
    setup_beam_args(parser)

    return parser.parse_args()


def save_state(model, optimizer, path):
    """
    Saves the model and optimizer state.
    """
    model_path = join(path, 'model.pt')
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(state, model_path)
    print('Saving model to {}'.format(model_path))


def load_state(model, optimizer, path):
    """
    Loads the model and optimizer state.
    """
    try:
        model_path = join(path, 'model.pt')
        state = torch.load(model_path)

        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        print('Loading model from {}'.format(model_path))
    except FileNotFoundError:
        pass


def compute_reward(self, args):
    pass


def sample_step(model, args):
    pass


def train(model, datasets, indices, args, device):
    """
    Performs training, validation and testing.
    """
    src_pad_index, trg_pad_index, _, _ = indices
    src_pad_index = torch.tensor(src_pad_index)
    src_pad_index.to(device)

    train, valid, test = datasets
    criterion = create_criterion(args, trg_pad_index)
    optimizer = create_optimizer(args, model.parameters())

    load_state(model, optimizer, args.model_dir)

    for epoch in range(args.epochs):

        # Running training loop.
        with tqdm(total=len(train)) as pbar:
            pbar.set_description('epoch {}'.format(epoch))
            model.train()

            for batch in train:
                loss, accuracy = train_step(
                    model=model, 
                    criterion=criterion, 
                    optimizer=optimizer,
                    indices=indices, 
                    batch=batch)

                pbar.set_postfix(ordered_dict=OrderedDict(
                    loss=loss, acc=accuracy))
                pbar.update()

        # Running validation loop.
        with tqdm(total=len(valid)) as pbar:
            pbar.set_description('validation')
            model.eval()
            results = []

            with torch.no_grad():
                for batch in valid:
                    loss, accuracy, preds = eval_step(
                        model=model,
                        criterion=criterion,
                        indices=indices,
                        device=device,
                        beam_size=1,
                        batch=batch)
                
                    pbar.set_postfix(ordered_dict=OrderedDict(
                        loss=loss, acc=accuracy))
                    pbar.update()

        save_state(model, optimizer, args.model_dir)

    # Running testing loop.
    with tqdm(total=len(test)) as pbar:
        pbar.set_description('testing')
        model.eval()

        with torch.no_grad():
            for batch in test:
                loss, accuracy = eval_step(
                    model=model,
                    criterion=criterion,
                    indices=indices,
                    device=device,
                    beam_size=args.beam_size,
                    batch=batch)
                
                pbar.set_postfix(ordered_dict=OrderedDict(
                    loss=loss, acc=accuracy))
                pbar.update()


def main():
    args = setup_train_args()
    device = torch.device('cuda' if args.cuda else 'cpu') 

    datasets, vocabs = create_datasets(args, device)
    indices = get_special_indices(vocabs)

    model = create_model(args, vocabs, indices, device)

    train(model, datasets, indices, args, device)


if __name__ == '__main__':
    main()
