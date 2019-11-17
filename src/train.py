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
    
import sys
import json
import torch
import random
import argparse
import logging
import os

import numpy as np

from contextlib import contextmanager
from tabulate import tabulate
from tensorboardX import SummaryWriter

from collections import (
    OrderedDict, defaultdict)

from math import ceil
from datetime import datetime
from statistics import mean
from functools import partial

try:
    from apex import amp
    APEX_INSTALLED = True
except ImportError:
    APEX_INSTALLED = False

from torch.nn.functional import (
    softmax, log_softmax,
    nll_loss, cross_entropy)

from torch.distributed import (
    all_reduce, ReduceOp, barrier)

from torch.optim.lr_scheduler import LambdaLR
from torch.nn.utils import clip_grad_norm_
from torch.nn.parallel import (
    DistributedDataParallel)

from transformers import AdamW

from os.path import (
    exists, join,
    abspath, dirname)

# HACK to enable launching with
# python src/train.py
PROJECT_PATH = join(abspath(dirname(__file__)), '..')
if PROJECT_PATH not in sys.path:
    sys.path.append(PROJECT_PATH)

from src.data import (
    create_dataset,
    setup_data_args,
    create_dummy_batch)

from src.model import (
    compute_size,
    create_model,
    setup_model_args)

    
def setup_train_args():
    """
    Sets up the training arguments.
    """
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group('train')
    group.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path of the config file.')
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
        '--fp16',
        action='store_true',
        help='Use fp16 precision training.')
    group.add_argument(
        '--lr',
        type=float,
        default=1e-5,
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
        '--schedule',
        type=str,
        default='noam',
        choices=['noam', 'noamwd'],
        help='Type of learning rate scheduling.')
    group.add_argument(
        '--warmup_steps',
        type=int,
        default=16000,
        help='Number of warmup steps.')
    group.add_argument(
        '--total_steps',
        type=int,
        default=1000000,
        help='Number of optimization steps.')
    group.add_argument(
        '--grad_accum_steps',
        type=int,
        default=2,
        help='Number of steps for grad accum.')
    group.add_argument(
        '--local_rank',
        type=int,
        default=-1,
        help='Local rank for the script.')
    group.add_argument(
        '--notebook',
        action='store_true',
        help='Set true if you are using IPython notebook.')
    group.add_argument(
        '--clip_grad',
        type=float,
        default=None,
        help='Gradient clipping norm value.')
    group.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for training.')

    setup_data_args(parser)
    setup_model_args(parser)

    return parser.parse_args()


def set_random_seed(args):
    """
    Sets the random seed for training.
    """
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_state(
        model_dir, model, optimizer, logger, device):
    """
    Loads the model and optimizer state.
    """
    try:
        model_path = join(model_dir, 'last.pt')
        state_dict = torch.load(
            model_path, map_location=device)

        model.load_state_dict(state_dict['model'])
        optimizer.load_state_dict(state_dict['optimizer'])

        logger.info('Loading model from {}'.format(
            model_path))

        return (
            state_dict['best_valid_loss'],
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
        lr=args.lr,
        params=parameters,
        weight_decay=0.01)

    return optimizer


# implementation is from DialoGPT repo
def noam_decay(step, warmup_steps, d_model):
    """
    Learning rate schedule described in
    https://arxiv.org/pdf/1706.03762.pdf.
    """
    return (
        d_model ** (-0.5) * min(step ** (-0.5), 
        step * warmup_steps**(-1.5)))


# implementation is from DialoGPT repo
def noamwd_decay(
        step, warmup_steps, d_model, rate=0.5,
        decay_steps=1000, start_step=500):
    """
    Learning rate schedule optimized for huge batches.
    """
    rate_exp = max(step - start_step + decay_steps, 0) \
        // decay_steps

    return (
        d_model ** (-0.5) *  min(step ** (-0.5), 
        step * warmup_steps ** (-1.5)) *
        rate ** (rate_exp))


# implementation is from DialoGPT repo
def set_lr(step, optimizer, schedule, lr,
           warmup_steps, d_model):
    """
    Learning rate scheduler that applies either
    noam or noamwd rule.
    """
    if schedule == 'noam':
        lr_this_step = lr * 1e4 * \
            noam_decay(step + 1, warmup_steps, d_model)

    elif schedule == 'noamwd':
        lr_this_step = lr * 1e4 * noamwd_decay(
            step + 1, warmup_steps, d_model)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_this_step


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

    acc = (correct / num_targets).item()
    loss = loss / num_targets

    ppl = torch.exp(loss).item()

    if ppl == float('inf'):
        ppl = 1e20

    return loss, acc, ppl


def main():
    """
    Performs training, validation and testing.
    """
    args = setup_train_args()

    if args.notebook:
        from tqdm import tqdm_notebook as tqdm
    else:
        from tqdm import tqdm

    # if config is provided, then load it
    if args.config is not None:
        with open(args.config, 'r') as fh:
            config = json.load(fh)

        for arg in config:
            setattr(args, arg, config[arg])

    args.cuda = torch.cuda.is_available() \
        and not args.no_cuda

    # setting random seed for reproducibility
    if args.seed:
        set_random_seed(args)

    model_dir = join(
        args.model_dir, args.model, args.name)

    os.makedirs(model_dir, exist_ok=True)
    logger = create_logger(model_dir=model_dir)

    if args.fp16 and not APEX_INSTALLED:
        logger.warn(
            '--fp16 passed but apex is not installed.')

    args.fp16 = args.fp16 and APEX_INSTALLED \
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

    if args.distributed:
        # creating the dataset and model only on
        # a single process ( downloading )
        if master_process:
            _, tokenizer, _ = create_dataset(
                args, master_process)

            vocab_size = len(tokenizer)

            create_model(args, model_dir, vocab_size)

        # other threads are waiting for the data init
        barrier()

    datasets, tokenizer, max_len = create_dataset(
        args=args, master_process=master_process)

    pad_idx = tokenizer.convert_tokens_to_ids(
        tokenizer.pad_token)
    vocab_size = len(tokenizer)

    model = create_model(args, model_dir, vocab_size)
    model = model.to(device)

    # TODO fix xlnet nan with mixed precision
    if 'xlnet' in args.model:
        args.fp16 = False

    optimizer = create_optimizer(
        args=args, parameters=model.parameters())

    if master_process:
        writer = SummaryWriter(
            logdir=model_dir, flush_secs=100)

    # loading previous state of the training
    best_valid_loss, init_epoch, step = load_state(
        model_dir=model_dir, model=model,
        optimizer=optimizer, logger=logger,
        device=device)

    if args.fp16:
        model, optimizer = amp.initialize(
            model, optimizer, opt_level='O2')

    d_model = model.config.d_model if 'xlnet' in \
        args.model else model.config.n_embd

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

    patience, skip, loss, accuracy = 0, 1, 0, 0

    set_lr_fn = partial(
        set_lr, 
        optimizer=optimizer, 
        schedule=args.schedule, 
        lr=args.lr, 
        warmup_steps=args.warmup_steps,
        d_model=d_model)

    if master_process:
        # loading history for training logs
        history_path = join(model_dir, 'history.json')

        history = defaultdict(list)

        # NOTE the hardcoded values to keep track of
        # in the history
        metrics = ['loss', 'acc', 'ppl']
        headers = ['epoch'] + \
            ['train_' + m for m in metrics] + \
            ['valid_' + m for m in metrics]

        if exists(history_path):
            with open(history_path, 'r') as fh:
                history = json.load(fh)

    def print_results(results):
        """
        Prints the history to the standard output.
        """
        data = list(zip(*[history[h] for h in headers]))

        table = tabulate(
            tabular_data=data,
            headers=headers,
            floatfmt='.3f')

        # computing the tabular table string and
        # printing only the last element
        print(table.split('\n')[-1])

        msg = ', '.join(
            '{}: {}'.format(n, r) for
            n, r in results.items())

        logger.info(msg)

    def record_history(results):
        """
        Records the results and prints them.
        """
        # saving history and handling unexpected
        # keyboard interrupt
        for header in headers:
            history[header].append(results[header])

        while True:
            try:
                with open(history_path, 'w') as fh:
                    json.dump(history, fh)
                break
            except KeyboardInterrupt:
                pass

    @contextmanager
    def skip_error():
        """
        Convenience function for skipping errors.
        """
        nonlocal skip

        try:
            # checking out of memory error and
            # proceeding if only a single GPU
            # is used for the training
            yield

        except RuntimeError as e:
            if 'out of memory' in str(e):
                if args.distributed:
                    raise e
                skip += 1

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

        outputs = model(inputs, half=args.fp16)

        # converting targets from ndarray
        targets = torch.as_tensor(targets)
        targets = targets.long().to(device)

        loss, acc, ppl = compute_loss(
            outputs=outputs,
            targets=targets,
            ignore_idx=pad_idx)

        if args.distributed:
            # reducing accuracy accross devices
            # for more accurate logging
            acc = reduce_tensor(acc)

        return loss, acc, ppl

    def train_step(batch):
        """
        Performs a single step of training.
        """
        nonlocal step, skip

        loss, acc, ppl = forward_step(batch)

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

        if args.clip_grad is not None:
            clip_grad_norm(args.clip_grad)

        if step % args.grad_accum_steps == 0:
            set_lr_fn(step)
            optimizer.step()
            optimizer.zero_grad()

        if args.distributed:
            # reducing loss accross devices for
            # more accurate logging
            loss = reduce_tensor(loss)

        step += 1

        return {
            'loss': loss.item(), 
            'acc': acc, 
            'ppl': ppl
        }

    def backward(loss):
        """
        Backpropagates the loss in either mixed or
        normal precision mode.
        """
        # cuda is required for mixed precision training.
        if args.fp16:
            with amp.scale_loss(
                    loss, optimizer) as scaled:
                scaled.backward()
        else:
            loss.backward()

    def clip_grad_norm(max_norm):
        """
        Applies gradient clipping.
        """
        if args.fp16:
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
            dataset(), 'eval',
            num_steps, False,
            disable=not master_process)

        model.eval()

        for batch in loop:
            with skip_error():
                loss, accuracy, ppl = forward_step(batch)

                loop.set_postfix(OrderedDict(
                    loss=loss.item(), ppl=ppl, acc=accuracy))

                yield loss.item(), accuracy, ppl

    def save_state(name):
        """
        Saves the model and optimizer state.
        """
        model_path = join(model_dir, name + '.pt')

        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_valid_loss': best_valid_loss,
            'valid_loss': valid_loss,
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

    if master_process:
        train_args = vars(args)
        logger.info(str(train_args))
        
        print()
        print(tabulate(train_args.items(), tablefmt='presto'))
        print()

    try:
        # initializing cuda buffer to avoid OOM errors
        dummy_batch = create_dummy_batch(
            args, ignore_idx=pad_idx)

        train_step(dummy_batch)

    except (RuntimeError, ValueError) as e:
        if 'out of memory' in str(e):
            msg = 'Not enough memory, there might ' + \
                'be several out of memory error during ' + \
                'training. To avoid this lower ' + \
                'the `--batch_size` or `--max_len`'

            if not args.grad_ckpt:
                msg += ', use the `--checkpointed` flag'

            if not APEX_INSTALLED:
                msg += ' or install apex for fp16 precision'

            logger.info(msg + '.')

        if args.distributed:
            return

    # creating table of history with correctly
    # arranged values for each header
    if master_process:
        table = list(zip(*[history[h] for h in headers]))
        print(tabulate(table, headers, floatfmt='.3f'))

    for epoch in range(init_epoch, args.max_epochs):
        # running training loop
        loop = tqdm(
            train_dataset(), 'train {}'.format(epoch),
            num_train_steps, False,
            disable=not master_process)

        train_metrics = defaultdict(list)

        model.train()

        for batch in loop:
            with skip_error():
                results = train_step(batch)

                loss = results['loss']
                if master_process and loss is not None:
                    # adding the results to history
                    # and logging them to tensorboard
                    for metric, value in results.items():
                        train_metrics[metric].append(value)
                        writer.add_scalar(
                            'train/' + metric, value, step)

            loop.set_postfix(OrderedDict(
                **results, skip=skip))

        train_metrics = {
            metric: mean(values) if len(values) > 0 else 0.0
            for metric, values in train_metrics.items()
        }

        with torch.no_grad():
            valid_metrics = zip(*evaluate(
                dataset=valid_dataset,
                num_steps=num_valid_steps))

        valid_loss, valid_acc, valid_ppl = [
            mean(values) if len(values) > 0 else 0.0
            for values in valid_metrics
        ]

        # switching back to training
        model.train()

        if master_process:
            results = {'epoch': epoch}

            results.update(train_metrics)

            results.update({
                'valid_loss': valid_loss,
                'valid_acc': valid_acc,
                'valid_ppl': valid_ppl
            })

            record_history(results)
            print_results(results)

            # logging to tensorboard
            writer.add_scalar(
                'val/loss', valid_loss, step)
            writer.add_scalar(
                'val/acc', valid_acc, step)
            writer.add_scalar(
                'val/ppl', valid_ppl, step)

        if master_process:
            save_state(name='last')

        if valid_loss < best_valid_loss:
            patience = 0
            best_valid_loss = valid_loss

            if master_process:
                save_state(name='best')

        else:
            patience += 1
            if patience == args.patience:
                # terminate when max patience
                # level is hit
                break

        if step == args.total_steps:
            break

    if master_process:
        writer.close()

    with torch.no_grad():
        test_metrics = zip(*evaluate(
            dataset=test_dataset,
            num_steps=num_test_steps))

    test_loss, test_acc, test_ppl = [
        mean(values) if len(values) > 0 else 0.0
        for values in test_metrics
    ]

    if master_process:
        logger.info('test loss: {:.4}'.format(
            test_loss))


if __name__ == '__main__':
    try:
        main()

    except KeyboardInterrupt:
        # exiting training with Ctrl + C
        pass
