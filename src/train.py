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


def create_optimizer(args, parameters):
    """
    Creates an ADAM optimizer.
    """
    optimizer = Adam(lr=args.learning_rate, params=parameters)
    return optimizer


def compute_loss(outputs, targets, criterion, pad_index):
    """
    Computes the loss and accuracy with masking.
    """
    scores, preds = outputs

    scores_view = scores.view(-1, scores.size(-1))
    targets_view = targets.view(-1)
    loss = criterion(scores_view, targets_view)
    
    notnull = targets.ne(pad_index)
    target_tokens = notnull.long().sum().item()
    correct = ((targets == preds) * notnull).sum().item()

    accuracy = correct / target_tokens
    loss = loss / target_tokens

    return loss, accuracy


def train_step(model, criterion, optimizer, batch, indices):
    """
    Performs a single step of training.
    """
    optimizer.zero_grad()

    _, trg_pad_index, _, _ = indices
    inputs, targets = batch.src, batch.trg
    max_len = targets.size(1)

    # Apply teacher forcing at a 0.5 ratio.
    if random.random() > 0.5:
        outputs = model(inputs=inputs, max_len=max_len)
    else:
        outputs = model(inputs=inputs, targets=targets, 
                        max_len=max_len)

    loss, accuracy = compute_loss(
        outputs=outputs, 
        targets=targets, 
        criterion=criterion,
        pad_index=trg_pad_index)

    loss.backward()

    optimizer.step()

    return loss.detach().cpu().numpy(), accuracy


def eval_step(model, criterion, batch, indices, device, beam_size):
    """
    Performs a single step of evaluation.
    """
    inputs, targets = batch.src, batch.trg
    src_pad_index, trg_pad_index, _, _ = indices
    
    if beam_size > 1:
        outputs = beam_search(
            model=model, 
            inputs=inputs,
            indices=indices,
            beam_size=beam_size,
            device=device,
            max_len=targets.size(1))
    else:
        outputs = model(inputs=inputs, max_len=targets.size(1))

    scores, preds = outputs
    batch_size, sequence_length = targets.size()

    # Handling the case when greedy or beam search decoding
    # terminates before target sequence length.
    if scores.size(1) < sequence_length:
        size_diff = sequence_length - scores.size(1)
        scores = torch.cat([scores, torch.zeros(
            [batch_size, size_diff, scores.size(2)])], dim=1) 
        preds = torch.cat([preds, src_pad_index.expand(
            batch_size, size_diff)], dim=1)

        outputs = scores, preds

    loss, accuracy = compute_loss(
        outputs=outputs, 
        targets=targets, 
        criterion=criterion,
        pad_index=trg_pad_index)

    return loss.cpu().numpy(), accuracy, preds


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
