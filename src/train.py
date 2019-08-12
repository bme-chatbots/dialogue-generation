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
import logging
import os

import numpy as np

from model import (
    compute_size,
    create_model,
    setup_model_args)

from data import (
    create_dataset,
    setup_data_args)

from tensorboardX import SummaryWriter
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
from torch.nn.parallel import (
    DistributedDataParallel)

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
        default=15,
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
        default=5,
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
        default=4,
        help='Number of steps for grad accum.')
    parser.add_argument(
        '--warmup_steps',
        type=int,
        default=3,
        help='Number of steps for warmup schedule.')
    parser.add_argument(
        '--total_steps',
        type=int,
        default=1000000,
        help='Number of total steps.')
    parser.add_argument(
        '--local_rank',
        type=int,
        default=-1,
        help='Local rank for distributed training.')

    setup_data_args(parser)
    setup_model_args(parser)

    return parser.parse_args()


def save_state(args, state, logger):
    """
    Saves the model and optimizer state.
    """
    model_path = join(args.model_dir, 'model.pt')

    logger.info('Saving model to {}'.format(model_path))
    # making sure the model saving is not left in a
    # corrupted state after a keyboard interrupt
    while True:
        try:
            torch.save(state, model_path)
            break
        except KeyboardInterrupt:
            pass


def load_state(args, model, optimizer, logger, device):
    """
    Loads the model and optimizer state.
    """
    try:
        model_path = join(args.model_dir, 'model.pt')
        state_dict = torch.load(
            model_path, map_location=device)

        model.load_state_dict(state_dict['model'])
        optimizer.load_state_dict(state_dict['optimizer'])
        logger.info(
            'Loading model from {}'.format(model_path))
        return (
            state_dict['avg_loss'],
            state_dict['epoch'],
            state_dict['step']
        )

    except FileNotFoundError:
        return np.inf, 0, 0
    

def create_summary_writer(args, model, logger, device):
    """
    Creates a tensorboard record writer.
    """
    writer = SummaryWriter(
        logdir=args.model_dir,
        flush_secs=20)

    try:
        writer.add_graph(model, torch.ones([2, 2]))

    except Exception as e:
        logger.warn(
            'Failed to save model graph: {}'.format(e))

    return writer
    

def create_logger(args):
    """
    Creates a logger that outputs information to a
    file and the standard output as well.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')

    # setting up logging to the console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # setting up logging to a file
    filename = '{date}.log'.format(
        date=datetime.today().strftime(
            '%m-%d-%H-%M'))

    log_path = join(args.model_dir, filename)
    file_handler = logging.FileHandler(
        filename=log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def create_optimizer(args, parameters):
    """
    Creates an adam optimizer.
    """
    optimizer = AdamW(
        lr=args.learning_rate, 
        params=parameters, weight_decay=1e-6)

    return optimizer


def compute_loss(outputs, labels):
    """
    Computes the loss and accuracy.
    """
    logits = outputs[0]

    logits_view = logits.view(-1, logits.size(-1))
    labels_view = labels.view(-1)

    loss = cross_entropy(logits_view, labels_view)

    _, preds = logits_view.max(dim=-1)

    accuracy = labels_view == preds
    accuracy = accuracy.float().mean().item()

    return loss, accuracy


def main():
    """
    Performs training, validation and testing.
    """
    args = setup_train_args()
    distributed = args.local_rank != -1
    master_process = args.local_rank in [0, -1]

    logger = create_logger(args)

    if distributed and args.cuda:
        # use distributed training if local rank is given
        # and GPU training is requested
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)

        torch.distributed.init_process_group(
            backend='nccl', init_method='env://')

    else:
        device = torch.device(
            'cuda' if args.cuda else 'cpu')

    # creating dataset and storing dataset splits
    # as individual variables for convenience
    datasets, tokenizer = create_dataset(
        args=args, device=device,
        distributed=distributed)

    vocab_size = len(tokenizer)

    model = create_model(
        args=args, vocab_size=vocab_size,
        device=device)

    optimizer = create_optimizer(
        args=args, parameters=model.parameters())

    writer = create_summary_writer(
        args=args, model=model, 
        logger=logger, device=device)

    # loading previous state of the training
    best_avg_loss, init_epoch, step = load_state(
        args=args, model=model, optimizer=optimizer,
        logger=logger, device=device)

    if args.mixed and args.cuda:
        model, optimizer = amp.initialize(
            model, optimizer, opt_level='O2')

    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[args.local_rank], 
            output_device=args.local_rank)

    # TODO get world size here instead of 1
    train, valid, test = [
        (split, ceil(size / args.batch_size / 1)) 
        for split, size in datasets]

    # computing the sizes of the dataset splits
    train_dataset, num_train_steps = train
    valid_dataset, num_valid_steps = valid
    test_dataset, num_test_steps = test

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
            labels=labels)

        return loss, accuracy

    def train_step(batch):
        """
        Performs a single step of training.
        """
        nonlocal step

        loss, accuracy = forward_step(batch)

        if torch.isnan(loss).item():
            logger.warn('skipping step (nan)')
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

    # TODO fix scheduling
    # scheduler = WarmupLinearSchedule(
    #     optimizer=optimizer,
    #     warmup_steps=args.warmup_steps,
    #     t_total=args.total_steps)
    
    if master_process:
        logger.info(str(vars(args)))

    # stepping optimizer from initial (0) learning rate
    # if init_epoch == 0:
    #     scheduler.step()

    for epoch in range(init_epoch, args.max_epochs):
        # running training loop
        loop = tqdm(
            train_dataset(), 
            total=num_train_steps,
            disable=not master_process)

        loop.set_description('{}'.format(epoch))
        model.train()
        avg_loss = []

        for batch in loop:
            try:
                loss, acc = train_step(batch)

                # logging to tensorboard
                if master_process:
                    writer.add_scalar('train/loss', loss, step)
                    writer.add_scalar('train/acc', acc, step)

                if loss is not None:
                    avg_loss.append(loss)

                loop.set_postfix(ordered_dict=OrderedDict(
                    loss=loss, acc=acc))

            except RuntimeError as e:
                if 'out of memory' in str(e):
                    logger.warn('skipping step (oom)')

        # scheduler.step(epoch=epoch)

        if len(avg_loss) > 0:
            avg_loss = sum(avg_loss) / len(avg_loss)
        else:
            avg_loss = 0.0

        if master_process:
            logger.info('train loss: {:.4}'.format(avg_loss))

        loop = tqdm(
            valid_dataset(), 
            total=num_valid_steps,
            disable=not master_process)

        model.eval()
        avg_loss = []

        # running validation loop
        with torch.no_grad():
            for batch in loop:
                loss, accuracy = forward_step(batch)

                avg_loss.append(loss.item())

                loop.set_postfix(ordered_dict=OrderedDict(
                    loss=loss.item(), acc=accuracy))

            avg_loss = sum(avg_loss) / len(avg_loss)

            if master_process:
                writer.add_scalar('valid/loss', avg_loss, step)
        
        if master_process:
            logger.info('valid loss: {:.4}'.format(avg_loss))

        if avg_loss < best_avg_loss:
            patience = 0
            best_avg_loss = avg_loss
            if master_process:
                save_state(
                    args=args,
                    logger=logger,
                    state={
                        'model': model.state_dict(),
                        'optimzier': optimizer.state_dict(),
                        'avg_loss': avg_loss,
                        'epoch': epoch + 1,
                        'step': step
                    })

        else:
            patience += 1
            if patience == args.patience:
                # terminate when max patience level is hit
                break

    writer.close()

    loop = tqdm(
        test_dataset(), 
        total=num_test_steps,
        disable=not master_process)

    model.eval()
    avg_loss = []

    # running testing loop
    with torch.no_grad():
        for batch in loop:
            loss, accuracy = forward_step(batch)

            avg_loss.append(loss.item())

            loop.set_postfix(ordered_dict=OrderedDict(
                loss=loss.item(), acc=accuracy))

        avg_loss = sum(avg_loss) / len(avg_loss)


if __name__ == '__main__':
    main()
