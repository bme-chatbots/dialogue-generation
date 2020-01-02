"""
@author:    Patrik Purgai
@copyright: Copyright 2019, gpt2-chatbot
@license:   MIT
@email:     purgai.patrik@gmail.com
@date:      2019.09.30.
"""

# pylint: disable=no-member

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import re
import sys
import os
import argparse
import random
import torch
import json

import numpy as np

try:
    from apex import amp
    APEX_INSTALLED = True
except ImportError:
    APEX_INSTALLED = False

from itertools import product
from tabulate import tabulate
from functools import partial
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm

from ignite.contrib.handlers import (
    ProgressBar)

from ignite.handlers import (
    ModelCheckpoint, EarlyStopping)

from ignite.metrics import RunningAverage

from ignite._utils import _to_hours_mins_secs

from torch.distributed import (
    all_reduce, ReduceOp)

from torch.nn.utils import clip_grad_norm_

from torch.nn.functional import (
    nll_loss, log_softmax,
    linear)

from torch.nn.parallel import (
    DistributedDataParallel)

from torch.optim import AdamW

from os.path import (
    exists, join,
    abspath, dirname)

# HACK to enable launching with
# python src/train.py
PROJECT_DIR = join(abspath(dirname(__file__)), '..')
if PROJECT_DIR not in sys.path:
    sys.path.append(PROJECT_DIR)

from src.model import (
    GPT2,
    setup_model_args)

from src.utils import (
    SPECIAL_TOKENS,
    set_random_seed,
    get_last_checkpoint,
    download_config,
    load_config,
    download_tokenizer,
    load_tokenizer,
    download_pretrained,
    execute_with_master,
    load_weights,
    SafeEngine,
    Events)

from src.data import (
    create_dataset,
    setup_data_args)


def setup_train_args():
    """
    Sets up the training arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_dir',
        type=str,
        default=join(PROJECT_DIR, 'model', 'test'),
        # datetime.today().strftime('%y.%m.%d-%H:%M:%S'),
        help='Maximum number of epochs for training.')
    parser.add_argument(
        '--max_epochs',
        type=int,
        default=25,
        help='Maximum number of epochs for training.')
    parser.add_argument(
        '--no_cuda',
        action='store_true',
        help='Device for training.')
    parser.add_argument(
        '--fp16',
        action='store_true',
        help='Use fp16 precision training.')
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-4,
        help='Learning rate for the model.')
    parser.add_argument(
        '--schedule',
        type=str,
        default='noam',
        choices=['noam', 'noamwd', 'linear'],
        help='Type of learning rate scheduling.')
    parser.add_argument(
        '--warmup_steps',
        type=int,
        default=16000,
        help='Number of warmup steps.')
    parser.add_argument(
        '--total_steps',
        type=int,
        default=1000000,
        help='Number of optimization steps.')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size during training.')
    parser.add_argument(
        '--patience',
        type=int,
        default=5,
        help='Number of patience epochs before termination.')
    parser.add_argument(
        '--grad_accum_steps',
        type=int,
        default=2,
        help='Number of steps for grad accum.')
    parser.add_argument(
        '--clip_grad_norm',
        type=float,
        default=None,
        help='radient clipping value.')
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Seed for reproducibility.')
    parser.add_argument(
        '--force_download',
        action='store_true',
        help='Download files even if they exist.')
    parser.add_argument(
        '--local_rank',
        type=int,
        default=-1,
        help='Local rank for the script.')

    setup_data_args(parser)
    setup_model_args(parser)

    return parser.parse_args()


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
        step * warmup_steps**(-1.5)) *
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

    elif schedule == 'linear':
        pass

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_this_step


def compute_loss(logits, targets, ignore_idx):
    """
    Computes the loss and accuracy.
    """
    targets = targets.reshape(-1)

    logits_view = logits.reshape(-1, logits.size(-1))

    log_probs = log_softmax(logits_view, dim=-1)

    loss = nll_loss(
        log_probs, targets,
        ignore_index=ignore_idx,
        reduction='sum')

    _, preds = log_probs.max(dim=-1)

    # computing accuracy without including the
    # values at the ignore indices
    not_ignore = targets.ne(ignore_idx)
    num_targets = not_ignore.long().sum().item()

    correct = (targets == preds) & not_ignore
    correct = correct.float().sum()

    acc = correct / num_targets
    loss = loss / num_targets

    ppl = torch.exp(loss).item()

    return {'loss': loss, 'acc': acc, 'ppl': ppl}


def main():
    """
    Performs training, validation and testing.
    """
    args = setup_train_args()

    def get_save_pattern(asset_name):
        """
        Returns the saving file pattern.
        """
        return r'{}_{}_\d+.pth'.format(
            args.pretrained, asset_name)

    def load_asset(asset_state_path, asset):
        """
        Loads the asset state from the path.
        """
        asset_state = torch.load(
            asset_state_path, map_location=device)
        asset.load_state_dict(asset_state)

    args.cuda = torch.cuda.is_available() \
        and not args.no_cuda

    # setting random seed for reproducibility
    if args.seed: set_random_seed(args)

    os.makedirs(args.model_dir, exist_ok=True)

    if args.fp16 and not APEX_INSTALLED:
        msg = '--fp16 passed but apex is not installed.'
        logger.warning(msg)

    args.distributed = args.local_rank != -1
    args.fp16 = args.fp16 and args.cuda and \
        APEX_INSTALLED

    if args.distributed:
        # use distributed training if local rank is 
        # given and GPU training is requested
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)

        torch.distributed.init_process_group(
            backend='nccl',
            init_method='env://',
            rank=args.local_rank)

    else:
        device = torch.device(
            'cuda' if args.cuda else 'cpu')

    # creating tokenizer and fetching vocab size
    tokenizer, specials = load_tokenizer(args)

    # tokenzier must be downloaded
    if tokenizer is None:
        with execute_with_master(args):
            download_tokenizer(args)

        tokenizer, specials = load_tokenizer(args)

    args.vocab_size = len(tokenizer.encoder)

    # loading config and downloading it is not found
    config = load_config(args)

    if config is None:
        with execute_with_master(args):
            download_config(args)

        config = load_config(args)

    # loading model from checkpoint or from
    # pretrained weights
    model_ckpt_path = get_last_checkpoint(
        model_dir=args.model_dir,
        pattern=get_save_pattern('model'))

    if model_ckpt_path is not None:
        config.vocab_size += len(SPECIAL_TOKENS)
        model = GPT2(config)

    else:
        with execute_with_master(args):
            # checking if pretrained weights exists
            # otherwise they are downloaded
            download_pretrained(args)

        model_state = load_weights(args.pretrained)
        model = GPT2(config)
        model.load_state_dict(model_state, strict=False)
        # if the training is resumed then the weights
        # have to be loaded after amp initialization

        model.expand_embeddings(len(SPECIAL_TOKENS))
        config.vocab_size += len(SPECIAL_TOKENS)

    model = model.to(device)
    params = model.parameters()
    optimizer = create_optimizer(args, params)

    optim_ckpt_path = get_last_checkpoint(
        model_dir=args.model_dir, 
        pattern=get_save_pattern('optim'))

    # using apex if required and loading its state
    if args.fp16:
        model, optimizer = amp.initialize(
            model, optimizer, opt_level='O2')

        amp_ckpt_path = get_last_checkpoint(
            model_dir=args.model_dir,
            pattern=get_save_pattern('amp'))

        if amp_ckpt_path is not None:
            load_asset(amp_ckpt_path, amp)

    if model_ckpt_path is not None:
        load_asset(model_ckpt_path, model)

    if optim_ckpt_path is not None:
        load_asset(optim_ckpt_path, optimizer)

    if args.distributed:
        model = DistributedDataParallel(
            model, device_ids=[args.local_rank],
            output_device=args.local_rank)

    args.world_size = int(
        os.environ.get('WORLD_SIZE', 1))

    # creating dataset split loaders
    train_dataset, valid_dataset = create_dataset(
        args, tokenizer, specials)

    def reduce_tensor(tensor):
        """
        Averages a tensor across gpus.
        """
        reduced = tensor.clone()
        all_reduce(reduced, op=ReduceOp.SUM)
        reduced /= world_size

        return reduced

    def to_tensor(array):
        """
        Converts the provided tf array to torch
        tensor.
        """
        return torch.from_numpy(
            array.numpy()).to(device)

    def unpack_tensors(batch):
        """
        Creates `input_ids`, `attn_mask` and
        `type_ids` from `inputs` tensor.
        """
        input_ids, attn_mask, type_ids = [
            to_tensor(batch[name]) for name in
            ['input_ids', 'attn_mask', 'type_ids']
        ]

        input_ids, targets = \
            input_ids[:, :-1].long(), \
            input_ids[:, 1:].long()

        attn_mask = attn_mask[:, :-1].bool()
        type_ids = type_ids[:, :-1].long()

        return (input_ids, attn_mask, type_ids), targets

    def forward_step(batch):
        """
        Applies forward pass with the given batch.
        """
        inputs, targets = unpack_tensors(batch)

        input_ids, attn_mask, type_ids = inputs

        outputs = model(
            input_ids=input_ids,
            attn_mask=attn_mask,
            type_ids=type_ids)

        metrics = compute_loss(
            outputs[0], targets, specials.PAD)

        return metrics

    def train_step(engine, batch):
        """
        Propagates the inputs forward and updates
        the parameters.
        """
        model.train()

        results = forward_step(batch)

        loss = results['loss'] / args.grad_accum_steps

        backward(loss)

        if args.clip_grad_norm is not None:
            clip_grad_norm(args.clip_grad_norm)

        if engine.state.iteration % \
                args.grad_accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        results['loss'] = loss.item()

        return results

    def eval_step(engine, batch):
        """
        Propagates the inputs forward without
        storing any gradients.
        """
        model.eval()

        with torch.no_grad():
            results = forward_step(batch)

        results['loss'] = results['loss'].item()

        return results

    def backward(loss):
        """
        Backpropagates the loss in either mixed or
        normal precision mode.
        """
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
            params = amp.master_params(optimizer)
            clip_grad_norm_(params, max_norm)
        else:
            clip_grad_norm_(
                model.parameters(), max_norm)

    def get_value(result, name):
        """
        Returns the value of the metrics from the
        results dict.
        """
        return min(max(result[name], -1e20), 1e20)

    # setting up logging and progress bar
    trainer = SafeEngine(train_step)
    valid_eval = SafeEngine(eval_step)

    metrics = ['loss', 'acc', 'ppl']

    tracked = product([trainer, valid_eval], metrics)
    for engine, name in tracked:
        output_transform = partial(get_value, name=name)

        metric = RunningAverage(
            output_transform=output_transform, alpha=0.8)
        metric.attach(engine, name)

    l_bar = '{desc}[{n_fmt}/{total_fmt}] '
    bar = '{percentage:3.0f}%|{bar}'
    r_bar = '{rate_fmt}{postfix} [{elapsed}<{remaining}]'

    pbar = ProgressBar(bar_format=l_bar + bar + r_bar)
    pbar.attach(trainer, metric_names=metrics)

    # adding model checkpoint handler
    checkpoint = ModelCheckpoint(
        args.model_dir, args.pretrained,
        n_saved=5,
        require_empty=False,
        save_as_state_dict=True,
        score_function=lambda e: -e.state.metrics['loss'])

    checkpoint_dict = {
        'model': model,
        'optim': optimizer
    }

    if args.fp16:
        checkpoint_dict['amp'] = amp

    valid_eval.add_event_handler(
        Events.COMPLETED, checkpoint, checkpoint_dict)

    early_stopping = EarlyStopping(
        patience=args.patience,
        score_function=lambda e: -e.state.metrics['loss'],
        trainer=trainer)

    valid_eval.add_event_handler(
        Events.COMPLETED, early_stopping)

    # loading history for training logs
    history_path = join(args.model_dir, 'history.json')

    history = defaultdict(list)

    # NOTE the hardcoded values to keep track of
    # in the history
    headers = ['epoch'] + list(
        ''.join(n) for n in  product(
            ['train_', 'valid_'], metrics))

    if exists(history_path):
        with open(history_path, 'r') as fh:
            history = json.load(fh)

    def record_history(results):
        """
        Records the results to the history.
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

    def run_eval(evaluator, loader, prefix):
        """
        Runs evaluation with the given evaluator.
        """
        evaluator.run(loader)
        results = evaluator.state.metrics

        return {
            prefix + '_' + metric: results[metric]
            for metric in metrics
        }

    @trainer.on(Events.STARTED)
    def setup_state(engine):
        """
        Sets the epoch to the resumed epoch.
        """
        if 'epoch' in history and \
                len(history['epoch']) > 0:
            engine.state.epoch = history['epoch'][-1]

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_eval_results(trainer):
        """
        Logs the training results.
        """
        results = {'epoch': trainer.state.epoch}

        results.update({
            'train_' + metric: \
                trainer.state.metrics[metric]
            for metric in metrics
        })

        results.update(run_eval(
            valid_eval, valid_dataset, 'valid'))

        record_history(results)

        data = list(zip(*[history[h] for h in headers]))
        table = tabulate(data, headers, floatfmt='.3f')

        print(table.split('\n')[-1])

    @trainer.on(Events.EXCEPTION_RAISED)
    def handle_exception(engine, e):
        if isinstance(e, KeyboardInterrupt) and \
                engine.state.iteration > 1:
            engine.terminate()

        elif isinstance(e, RuntimeError):
            if 'out of memory' in str(e):
                pass

            else:
                raise e

    print()
    print(tabulate(vars(args).items(), tablefmt='presto'))
    print()

    data = list(zip(*[history[h] for h in headers]))

    # printing the initial table headers and 
    # previous results of the training if resuming
    print(tabulate(data, headers, floatfmt='.3f'))

    trainer.run(train_dataset, args.max_epochs)


if __name__ == '__main__':
    main()

