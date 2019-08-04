"""
@author:    Patrik Purgai
@copyright: Copyright 2019, dialogue-generation
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
    compute_size,
    create_model,
    setup_model_args)

from data import (
    create_dataset,
    setup_data_args)

from collections import OrderedDict
from tqdm import tqdm
from math import ceil
from datetime import datetime

try:
    from apex import amp
    APEX_INSTALLED = True
except ImportError:
    APEX_INSTALLED = False

from torch.nn.functional import (
    cross_entropy, softmax,
    kl_div, log_softmax)

from torch.nn.utils import clip_grad_norm_
from torch.optim import SGD

from pytorch_transformers import (
    WarmupLinearSchedule, AdamW)

from os.path import (
    exists, join,
    dirname, abspath)


def setup_train_args():
    """
    Sets up the training arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--max_epochs',
        type=int,
        default=1000,
        help='Maximum number of epochs for training.')
    parser.add_argument(
        '--cuda',
        type=bool,
        default=torch.cuda.is_available(),
        help='Device for training.')
    # TODO XLNet produces NaN with apex
    parser.add_argument(
        '--mixed',
        type=bool,
        default=False,
        help='Use mixed precision training.')
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1e-2,
        help='Learning rate for the model.')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='Batch size during training.')
    parser.add_argument(
        '--patience',
        type=int,
        default=10,
        help='Number of patience epochs before termination.')
    parser.add_argument(
        '--model_dir',
        type=str,
        default=join(
            abspath(dirname(__file__)),
            '..', 'model.{}'.format(
                datetime.today().strftime('%j%H%m'))),
        help='Path of the model checkpoints.')
    parser.add_argument(
        '--grad_accum_steps',
        type=int,
        default=3,
        help='Number of steps for grad accum.')
    parser.add_argument(
        '--warmup_steps',
        type=int,
        default=500,
        help='Number of steps for warmup schedule.')
    parser.add_argument(
        '--total_steps',
        type=int,
        default=1000000,
        help='Number of steps for warmup schedule.')

    setup_data_args(parser)
    setup_model_args(parser)

    return parser.parse_args()


def save_state(model, optimizer, avg_acc, epoch, step,
               path):
    """
    Saves the model and optimizer state.
    """
    model_path = join(path, 'model.pt')
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'avg_acc': avg_acc,
        'epoch': epoch,
        'step': step
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
        return (
            state_dict['avg_acc'],
            state_dict['epoch'],
            state_dict['step']
        )

    except FileNotFoundError:
        return 0, 0, 0


def create_optimizer(args, parameters):
    """
    Creates an adam or swats optimizer with cyclical 
    learning rate.
    """
    # optimizer = SGD(
    #     lr=args.learning_rate, params=parameters,
    #     weight_decay=1e-6, nesterov=True, momentum=0.9)

    optimizer = AdamW(params=parameters, weight_decay=1e-6)

    return optimizer


def create_criterion(pad_idx, vocab_size, device,
                     smoothing=0.1):
    """
    Creates label smoothing loss with kl-divergence for the
    seq2seq model.
    """
    confidence = 1.0 - smoothing
    # initializes the target distribution vector with
    # smoothing value divided by the number of other
    # valid tokens
    smoothed = torch.full(
        (vocab_size, ), smoothing / (vocab_size - 2))
    smoothed = smoothed.to(device)
    smoothed[pad_idx] = 0

    def label_smoothing(outputs, targets):
        """
        Computes the kl-divergence between the preds and the 
        smoothed target distribution.
        """
        smoothed_targets = smoothed.repeat(targets.size(0), 1)
        smoothed_targets.scatter_(
            1, targets.unsqueeze(1), confidence)
        smoothed_targets.masked_fill_(
            (targets == pad_idx).unsqueeze(1), 0)

        return kl_div(outputs, smoothed_targets,
                      reduction='batchmean')

    return label_smoothing


def compute_loss(outputs, labels, pad_idx, criterion):
    """
    Computes the loss and accuracy with masking.
    """
    logits = outputs[0]

    scores = log_softmax(logits, dim=-1)

    scores_view = scores.view(-1, scores.size(-1))
    labels_view = labels.view(-1)

    loss = cross_entropy(
        scores_view, labels_view, ignore_index=pad_idx)

    _, preds = scores_view.max(dim=-1)

    accuracy = labels_view == preds
    accuracy = accuracy.float().mean().item()

    return loss, accuracy


def main():
    """
    Performs training, validation and testing.
    """
    args = setup_train_args()
    device = torch.device('cuda' if args.cuda else 'cpu')

    # creating dataset and storing dataset splits
    # as individual variables for convenience
    datasets, tokenizer = create_dataset(
        args=args, device=device)

    vocab_size = len(tokenizer)

    model = create_model(
        args=args, vocab_size=vocab_size,
        device=device)

    optimizer = create_optimizer(
        args=args, parameters=model.parameters())

    pad_idx = tokenizer.convert_tokens_to_ids(
        tokenizer.pad_token)

    criterion = create_criterion(
        pad_idx=pad_idx, vocab_size=vocab_size,
        device=device)

    # loading previous state of the training
    best_avg_acc, init_epoch, step = load_state(
        model=model, optimizer=optimizer,
        path=args.model_dir, device=device)

    if args.mixed and args.cuda:
        model, optimizer = amp.initialize(
            model, optimizer, opt_level='O2')

    train, valid, test = datasets

    # computing the sizes of the splits
    train_dataset, train_size = train
    num_train_steps = ceil(train_size / args.batch_size)

    valid_dataset, valid_size = valid
    num_valid_steps = ceil(valid_size / args.batch_size)

    test_dataset, test_size = test
    num_test_steps = ceil(test_size / args.batch_size)

    patience = 0

    def convert_to_tensor(ids):
        """
        Convenience function for converting int32
        ndarray to torch int64.
        """
        return torch.as_tensor(ids).long().to(device)

    def forward_step(batch):
        """
        Applies forward pass with the given batch.
        """
        inputs, labels = batch

        labels = convert_to_tensor(labels)

        # converting the batch of inputs to torch tensor
        inputs = [convert_to_tensor(m) for m in inputs]

        input_ids, token_type_ids, attn_mask, \
            perm_mask, target_map = inputs

        outputs = model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attn_mask,
            perm_mask=perm_mask,
            target_mapping=target_map.float())

        loss, accuracy = compute_loss(
            outputs=outputs,
            labels=labels,
            pad_idx=pad_idx,
            criterion=criterion)

        return loss, accuracy

    def train_step(batch):
        """
        Performs a single step of training.
        """
        nonlocal step

        loss, accuracy = forward_step(batch)

        if torch.isnan(loss).item():
            print('skipping step (nan)')
            # returning None values when a NaN loss
            # is encountered and skipping backprop
            # so model grads will not be corrupted
            return None, None

        loss /= args.grad_accum_steps

        backward(loss)
        clip_grad_norm(1.0)

        step += 1

        if step % args.grad_accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

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

    def clip_grad_norm(max_norm):
        """
        Applies gradient clipping.
        """
        if args.mixed and args.cuda:
            clip_grad_norm_(
                amp.master_params(optimizer), max_norm)
        else:
            clip_grad_norm_(model.parameters(), max_norm)

    scheduler = WarmupLinearSchedule(
        optimizer=optimizer,
        warmup_steps=args.warmup_steps,
        t_total=args.total_steps)

    for epoch in range(init_epoch, args.max_epochs):
        # running training loop
        loop = tqdm(train_dataset(), total=num_train_steps)
        loop.set_description('{}'.format(epoch))
        model.train()
        avg_acc = []

        for batch in loop:
            try:
                loss, accuracy = train_step(batch)

                avg_acc.append(accuracy)

                loop.set_postfix(ordered_dict=OrderedDict(
                    loss=loss, acc=accuracy))

                scheduler.step(epoch=step)

            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print('skipping step (oom)')

        if len(avg_acc) > 0:
            avg_acc = sum(avg_acc) / len(avg_acc)
        else:
            avg_acc = 0.0

        print('avg train acc: {:.4}'.format(avg_acc))

        loop = tqdm(valid_dataset(), total=num_valid_steps)
        model.eval()
        avg_acc = []

        # running validation loop
        with torch.no_grad():
            for batch in loop:
                loss, accuracy = forward_step(batch)

                avg_acc.append(accuracy)

                loop.set_postfix(ordered_dict=OrderedDict(
                    loss=loss, acc=accuracy))

            avg_acc = sum(avg_acc) / len(avg_acc)

        print('avg valid acc: {:.4}'.format(avg_acc))

        if avg_acc > best_avg_acc:
            best_avg_acc = avg_acc
            save_state(
                model=model, optimizer=optimizer,
                avg_acc=best_avg_acc, epoch=epoch,
                step=step, path=args.model_dir)

        else:
            patience += 1
            if patience == args.patience:
                break

    loop = tqdm(test_dataset(), total=num_test_steps)
    model.eval()
    avg_acc = []

    # running testing loop
    with torch.no_grad():
        for batch in loop:
            loss, accuracy = forward_step(batch)

            avg_acc.append(accuracy)

            loop.set_postfix(ordered_dict=OrderedDict(
                loss=loss, acc=accuracy))

        avg_acc = sum(avg_acc) / len(avg_acc)


if __name__ == '__main__':
    main()
