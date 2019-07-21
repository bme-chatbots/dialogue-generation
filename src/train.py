"""
@author:    Patrik Purgai
@copyright: Copyright 2019, sentiment-analysis
@license:   MIT
@email:     purgai.patrik@gmail.com
@date:      2019.07.12.
"""

# pylint: disable=import-error
# pylint: disable=no-name-in-module
# pylint: disable=no-member
# pylint: disable=not-callable

import torch
import argparse
import os

from model import (
    create_model,
    setup_model_args)

from data import (
    create_dataset,
    setup_data_args)

from collections import OrderedDict
from tqdm import tqdm
from datetime import datetime

try:
    from apex import amp
    APEX_INSTALLED = True
except ImportError:
    APEX_INSTALLED = False

from torch.nn.functional import (
    cross_entropy, softmax)
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

from os.path import (
    exists, join,
    dirname, abspath)


def setup_train_args():
    """
    Sets up the training arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--epochs',
        type=int,
        default=1000,
        help='Maximum number of epochs for training.')
    parser.add_argument(
        '--cuda',
        type=bool,
        default=torch.cuda.is_available(),
        help='Device for training.')
    parser.add_argument(
        '--mixed',
        type=bool,
        default=APEX_INSTALLED,
        help='Use mixed precision training.')
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1e-4,
        help='Learning rate for the model.')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=128,
        help='Batch size during training.')
    parser.add_argument(
        '--patience',
        type=int,
        default=10,
        help='Number of steps before restart.')
    parser.add_argument(
        '--model_dir',
        type=str,
        default=join(
            abspath(dirname(__file__)),
            '..', 'model.{}'.format(
                datetime.today().strftime('%j%H%m'))),
        help='Path of the model checkpoints.')

    setup_data_args(parser)
    setup_model_args(parser)

    return parser.parse_args()


def save_state(model, optimizer, avg_acc, epoch, path):
    """
    Saves the model and optimizer state.
    """
    model_path = join(path, 'model.pt')
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'avg_acc': avg_acc,
        'epoch': epoch
    }
    print('Saving model to {}'.format(model_path))
    # making sure the model saving is not left in a
    # corrupted state after a keyboard interrupt
    while True:
        try:
            torch.save(state, model_path)
            break
        except KeyboardInterrupt:
            pass


def load_state(model, optimizer, path, device):
    """
    Loads the model and optimizer state.
    """
    try:
        model_path = join(path, 'model.pt')
        state_dict = torch.load(
            model_path, map_location=device)

        model.load_state_dict(state_dict['model'])
        optimizer.load_state_dict(state_dict['optimizer'])
        print('Loading model from {}'.format(model_path))
        return state_dict['avg_acc'], state_dict['epoch']

    except FileNotFoundError:
        return 0, 0


def create_optimizer(args, parameters):
    """
    Creates an adam or swats optimizer with cyclical 
    learning rate.
    """
    optimizer = Adam(params=parameters, weight_decay=1e-6)

    return optimizer


def compute_lr(step, factor=3e-3, warmup=50, eps=1e-7):
    """
    Calculates learning rate with warm up.
    """
    if step < warmup:
        return (1 + factor) ** step
    else:
        # after reaching maximum number of steps
        # the lr is decreased by factor as well
        return max(((1 + factor) ** warmup) *
                   ((1 - factor) ** (step - warmup)), eps)


def compute_loss(logits, targets):
    """
    Computes the loss and accuracy with masking.
    """
    loss = cross_entropy(logits, targets)

    scores = softmax(logits, dim=-1)
    _, preds = scores.max(dim=-1)

    accuracy = (targets == preds).float().mean()

    return loss, accuracy.item()


def main():
    """
    Performs training, validation and testing.
    """
    args = setup_train_args()
    device = torch.device('cuda' if args.cuda else 'cpu')

    # creating dataset and storing dataset splits, fields
    # as individual variables for convenience
    train, valid, test, vocab_size = create_dataset(
        args=args, device=device)

    model = create_model(
        args=args, vocab_size=vocab_size, device=device)

    optimizer = create_optimizer(
        args=args, parameters=model.parameters())

    best_avg_acc, init_epoch = load_state(
        model, optimizer, args.model_dir, device)

    if args.mixed and args.cuda:
        model, optimizer = amp.initialize(
            model, optimizer, opt_level='O2')

    patience = 0

    def convert_to_tensor(ids):
        """
        Convenience function for converting int32
        ndarray to torch int64.
        """
        return torch.as_tensor(ids).long().to(device)

    def train_step(batch):
        """
        Performs a single step of training.
        """
        optimizer.zero_grad()

        inputs, attn_mask, labels = batch

        inputs = convert_to_tensor(inputs)
        attn_mask = convert_to_tensor(attn_mask)
        labels = convert_to_tensor(labels)

        logits = model(
            inputs=inputs, 
            attn_mask=attn_mask)

        loss, accuracy = compute_loss(
            logits=logits, 
            targets=labels)

        backward(loss)
        optimizer.step()

        return loss.item(), accuracy

    def eval_step(batch):
        """
        Performs a single step of training.
        """
        inputs, attn_mask, labels = batch

        inputs = convert_to_tensor(inputs)
        attn_mask = convert_to_tensor(attn_mask)
        labels = convert_to_tensor(labels)

        logits = model(
            inputs=inputs, 
            attn_mask=attn_mask)

        loss, accuracy = compute_loss(
            logits=logits, 
            targets=labels)

        return loss.item(), accuracy

    def backward(loss):
        """
        Backpropagates the loss in either mixed or
        normal precision mode.
        """
        # cuda is required for mixed precision training.
        if args.mixed and args.cuda:
            with amp.scale_loss(loss, optimizer) as scaled:
                scaled.backward()
        else:
            loss.backward()

    scheduler = LambdaLR(optimizer, compute_lr)

    for epoch in range(init_epoch, args.epochs):
        # running training loop
        loop = tqdm(train)
        loop.set_description('{}'.format(epoch))
        model.train()
        avg_acc = []

        for batch in loop:
            loss, accuracy = train_step(batch)

            avg_acc.append(accuracy)

            loop.set_postfix(ordered_dict=OrderedDict(
                loss=loss, acc=accuracy))

        scheduler.step()

        avg_acc = sum(avg_acc) / len(avg_acc)
        print('avg train acc: {:.4}'.format(avg_acc))

        loop = tqdm(test)
        model.eval()
        avg_acc = []

        # running validation loop
        with torch.no_grad():
            for batch in loop:
                loss, accuracy = eval_step(batch)

                avg_acc.append(accuracy)

                loop.set_postfix(ordered_dict=OrderedDict(
                    loss=loss, acc=accuracy))

            avg_acc = sum(avg_acc) / len(avg_acc)
        
        print('avg valid acc: {:.4}'.format(avg_acc))

        if avg_acc > best_avg_acc:
            best_avg_acc = avg_acc
            save_state(
                model, optimizer, best_avg_acc,
                epoch, args.model_dir)
        else:
            patience += 1
            if patience == args.patience:
                break


if __name__ == '__main__':
    main()
