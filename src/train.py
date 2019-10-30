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
# pylint: disable=used-before-assignment

from src.model import (
    compute_size,
    create_model,
    setup_model_args)
from src.data import (
    create_dataset,
    setup_data_args,
    create_dummy_batch)
import sys
import torch
import argparse
import logging
import os

import numpy as np

from tensorboardX import SummaryWriter
from collections import OrderedDict
from tqdm import tqdm
from math import ceil
from datetime import datetime
from statistics import mean
from functools import wraps

try:
    from apex import amp
    APEX_INSTALLED = True
except ImportError:
    APEX_INSTALLED = False

from torch.nn.functional import (
    cross_entropy, softmax,
    kl_div, log_softmax,
    nll_loss)

from torch.distributed import (
    all_reduce, ReduceOp)

from torch.optim.lr_scheduler import LambdaLR
from torch.nn.utils import clip_grad_norm_
from torch.nn.parallel import (
    DistributedDataParallel)

from pytorch_transformers import (
    WarmupLinearSchedule, AdamW)

from os.path import (
    exists, join,
    abspath, dirname)

# HACK to enable launching with
# python src/train.py
PROJECT_PATH = join(abspath(dirname(__file__)), '..')
if PROJECT_PATH not in sys.path:
    sys.path.append(PROJECT_PATH)


def setup_train_args():
    """
    Sets up the training arguments.
    """
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group('train')
    group.add_argument(
        '--max_epochs',
        type=int,
        default=25,
        help='Maximum number of epochs for training.')
    group.add_argument(
        '--no_cuda',
        action='store_true',
        help='Device for training.')
    # TODO XLNet produces NaN with apex
    group.add_argument(
        '--mixed',
        action='store_true',
        help='Use mixed precision training.')
    group.add_argument(
        '--learning_rate',
        type=float,
        default=1e-4,
        help='Learning rate for the model.')
    group.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='Batch size during training.')
    group.add_argument(
        '--patience',
        type=int,
        default=5,
        help='Number of patience epochs before termination.')
    group.add_argument(
        '--grad_accum_steps',
        type=int,
        default=8,
        help='Number of steps for grad accum.')
    group.add_argument(
        '--eval_every_step',
        type=int,
        default=3000,
        help='Evaluation frequency in steps.')
    group.add_argument(
        '--local_rank',
        type=int,
        default=-1,
        help='Local rank for the script.')

    setup_data_args(parser)
    setup_model_args(parser)

    return parser.parse_args()


def load_state(model_dir, model, optimizer, logger,
               device):
    """
    Loads the model and optimizer state.
    """
    try:
        model_path = join(model_dir, 'model.pt')
        state_dict = torch.load(
            model_path, map_location=device)

        model.load_state_dict(state_dict['model'])
        optimizer.load_state_dict(state_dict['optimizer'])

        logger.info('Loading model from {}'.format(
            model_path))

        return (
            state_dict['val_loss'],
            state_dict['epoch'],
            state_dict['step']
        )

    except FileNotFoundError:
        return np.inf, 0, 0


def create_logger(model_dir):
    """
    Creates a logger that outputs information to a
    file and the standard output as well.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')

    # setting up logging to the console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.DEBUG)
    logger.addHandler(console_handler)

    # setting up logging to a file
    log_path = join(model_dir, 'training.log')
    file_handler = logging.FileHandler(
        filename=log_path)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    return logger


def create_optimizer(args, parameters):
    """
    Creates an adam optimizer.
    """
    optimizer = AdamW(
        lr=args.learning_rate,
        params=parameters,
        weight_decay=1e-6)

    return optimizer


def compute_lr(step, factor=3e-3, warmup=3, eps=1e-7):
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


def compute_loss(outputs, targets, ignore_idx):
    """
    Computes the loss and accuracy.
    """
    logits = outputs[0]

    logits_view = logits.view(-1, logits.size(-1))
    targets_view = targets.view(-1)

    log_probs = log_softmax(logits_view, dim=-1)

    loss = nll_loss(
        log_probs, targets_view,
        ignore_index=ignore_idx,
        reduction='sum')

    _, preds = log_probs.max(dim=-1)

    # computing accuracy without including the
    # values at the ignore indices
    not_ignore = targets_view.ne(ignore_idx)
    num_targets = not_ignore.long().sum().item()

    correct = (targets_view == preds) & not_ignore
    correct = correct.float().sum()

    accuracy = correct / num_targets
    loss = loss / num_targets

    return loss, accuracy


def main():
    """
    Performs training, validation and testing.
    """
    args = setup_train_args()

    args.cuda = torch.cuda.is_available() \
        and not args.no_cuda

    model_dir = join(args.model_dir, args.model,
                     args.name)

    os.makedirs(model_dir, exist_ok=True)

    logger = create_logger(model_dir=model_dir)

    if args.mixed and not APEX_INSTALLED:
        logger.warn(
            '--mixed passed but apex is not installed.')

    args.mixed = args.mixed and APEX_INSTALLED \
        and args.cuda

    master_process = args.local_rank in [0, -1]
    args.distributed = args.local_rank != -1

    if args.distributed:
        # use distributed training if local rank is given
        # and GPU training is requested
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)

        torch.distributed.init_process_group(
            backend='nccl',
            init_method='env://',
            rank=args.local_rank)

    else:
        device = torch.device(
            'cuda' if args.cuda else 'cpu')

    # creating dataset and storing dataset splits
    # as individual variables for convenience
    datasets, tokenizer, max_len = create_dataset(
        args=args, master_process=master_process)

    pad_idx = tokenizer.convert_tokens_to_ids(
        tokenizer.pad_token)
    vocab_size = len(tokenizer)

    # TODO fix xlnet nan with mixed precision
    if 'xlnet' in args.model:
        args.mixed = False

    model = create_model(
        args=args, model_dir=model_dir,
        vocab_size=vocab_size)

    model = model.to(device)

    optimizer = create_optimizer(
        args=args, parameters=model.parameters())

    if master_process:
        writer = SummaryWriter(
            logdir=model_dir,
            flush_secs=100)

    # loading previous state of the training
    best_val_loss, init_epoch, step = load_state(
        model_dir=model_dir, model=model,
        optimizer=optimizer, logger=logger,
        device=device)

    if args.mixed:
        model, optimizer = amp.initialize(
            model, optimizer, opt_level='O2')

    if args.distributed:
        model = DistributedDataParallel(
            model, device_ids=[args.local_rank],
            output_device=args.local_rank)

    world_size = int(os.environ.get('WORLD_SIZE', 1))

    train, valid, test = [
        (split, ceil(
            size / args.batch_size / world_size))
        for split, size in datasets]

    # computing the sizes of the dataset splits
    train_dataset, num_train_steps = train
    valid_dataset, num_valid_steps = valid
    test_dataset, num_test_steps = test

    patience, skip, loss, acc = 0, 1, 0, 0

    def reduce_tensor(tensor):
        """
        Averages a tensor across gpus.
        """
        reduced = tensor.clone()
        all_reduce(reduced, op=ReduceOp.SUM)
        reduced /= world_size

        return reduced

    def forward_step(batch):
        """
        Applies forward pass with the given batch.
        """
        inputs, targets = batch

        outputs = model(
            inputs=inputs,
            half=args.mixed)

        # converting targets from ndarray
        targets = torch.as_tensor(targets)
        targets = targets.long().to(device)

        loss, accuracy = compute_loss(
            outputs=outputs,
            targets=targets,
            ignore_idx=pad_idx)

        if args.distributed:
            # reducing accuracy accross devices
            # for more accurate logging
            accuracy = reduce_tensor(accuracy)

        return loss, accuracy.item()

    def train_step(batch):
        """
        Performs a single step of training.
        """
        nonlocal step, skip

        loss, accuracy = forward_step(batch)

        if torch.isnan(loss).item():
            # during distributed training NaN
            # values are not handled
            if args.distributed:
                raise ValueError(
                    'NaN values encountered.')

            logger.debug('skipping step (nan)')
            # returning None values when a NaN loss
            # is encountered and skipping backprop
            # so model grads will not be corrupted

            skip += 1
            return None, None

        loss /= args.grad_accum_steps

        backward(loss)
        clip_grad_norm(1.0)

        if step % args.grad_accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        if args.distributed:
            # reducing loss accross devices for
            # more accurate logging
            loss = reduce_tensor(loss)

        return loss.item(), accuracy

    def backward(loss):
        """
        Backpropagates the loss in either mixed or
        normal precision mode.
        """
        # cuda is required for mixed precision training.
        if args.mixed:
            with amp.scale_loss(loss, optimizer) as scaled:
                scaled.backward()
        else:
            loss.backward()

    def clip_grad_norm(max_norm):
        """
        Applies gradient clipping.
        """
        if args.mixed:
            clip_grad_norm_(
                amp.master_params(optimizer), max_norm)
        else:
            clip_grad_norm_(model.parameters(), max_norm)

    def evaluate(dataset, num_steps):
        """
        Constructs a validation loader and evaluates
        the model.
        """
        loop = tqdm(
            dataset(),
            total=num_steps,
            disable=not master_process,
            desc='Eval')

        model.eval()

        for batch in loop:
            loss, acc = forward_step(batch)

            loop.set_postfix(ordered_dict=OrderedDict(
                loss=loss.item(), acc=acc))

            yield loss.item()

    def save_state():
        """
        Saves the model and optimizer state.
        """
        model_path = join(model_dir, 'model.pt')

        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'val_loss': best_val_loss,
            'epoch': epoch + 1,
            'step': step
        }

        logger.info('Saving model to {}'.format(model_path))
        # making sure the model saving is not left in a
        # corrupted state after a keyboard interrupt
        while True:
            try:
                torch.save(state, model_path)
                break
            except KeyboardInterrupt:
                pass

    scheduler = LambdaLR(optimizer, compute_lr)

    if master_process:
        logger.info(str(vars(args)))

    try:
        # initializing cuda buffer to avoid OOM errors
        dummy_batch = create_dummy_batch(
            args, ignore_idx=pad_idx)

        train_step(dummy_batch)

    except RuntimeError as e:
        if 'out of memory' in e:
            msg = 'Not enough memory, lower ' + \
                'the `--batch_size` or `--max_len`'

            if not args.checkpointed:
                msg += ', use the `--checkpointed` flag'
            
            if not APEX_INSTALLED:
                msg += ' or install apex for mixed precision'

            logger.info(msg + '.')
            return
        raise e

    for epoch in range(init_epoch, args.max_epochs):
        # running training loop
        loop = tqdm(
            train_dataset(),
            total=num_train_steps,
            disable=not master_process,
            desc='Train {}'.format(epoch))

        train_loss = []

        model.train()

        for batch in loop:
            loss, accuracy = train_step(batch)

            step += 1

            if master_process and loss is not None:
                train_loss.append(loss)

                # logging to tensorboard
                writer.add_scalar('train/loss', loss, step)
                writer.add_scalar('train/acc', acc, step)

            if not step % args.eval_every_step:
                with torch.no_grad():
                    val_loss = mean(evaluate(
                        dataset=valid_dataset,
                        num_steps=num_valid_steps))

                # switching back to training
                model.train()

                if master_process:
                    logger.info('val loss: {:.4}'.format(
                        val_loss))

                    # logging to tensorboard
                    writer.add_scalar(
                        'val/loss', val_loss, step)

                if val_loss < best_val_loss:
                    patience = 0
                    best_val_loss = val_loss

                    if master_process:
                        save_state()

                else:
                    patience += 1
                    if patience == args.patience:
                        # terminate when max patience
                        # level is hit
                        break

            loop.set_postfix(ordered_dict=OrderedDict(
                loss=loss, acc=acc, skip=skip))

        if len(train_loss) > 0:
            train_loss = mean(train_loss)
        else:
            train_loss = 0.0

        if master_process:
            logger.info('train loss: {:.4}'.format(
                train_loss))

        scheduler.step()

    if master_process:
        writer.close()

    with torch.no_grad():
        test_loss = mean(evaluate(
            dataset=test_dataset,
            num_steps=num_test_steps))

    if master_process:
        logger.info('test loss: {:.4}'.format(
            test_loss))


if __name__ == '__main__':
    try:
        main()

    except KeyboardInterrupt:
        # exiting training with Ctrl + C
        pass
