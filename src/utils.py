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

import random
import time
import torch
import json
import os
import re
import contextlib
import requests
import argparse

import numpy as np
import tensorflow as tf

from termcolor import colored
from tqdm import tqdm

from ignite._utils import (
    _to_hours_mins_secs)

from ignite.engine import Engine, Events

from torch.distributed import barrier

from os.path import (
    join, dirname, abspath, split, exists)

from src.encoder import get_encoder


URL = 'https://storage.googleapis.com/gpt-2/models/'

PROJECT_DIR = join(abspath(dirname(__file__)), '..')
CACHE_DIR = join(PROJECT_DIR, '.cache')

CONFIG_FILE_NAME = 'hparams.json'

MODEL_ASSETS = [
    'checkpoint',
    'model.ckpt.data-00000-of-00001',
    'model.ckpt.index',
    'model.ckpt.meta'
]

TOKENIZER_ASSETS = [
    'encoder.json',
    'vocab.bpe'
]

SPECIAL_TOKENS = ['SP1', 'SP2', 'PAD']


def list_files(data_dir, pattern):
    """
    Returns list of files from the directory
    that match the pattern.
    """
    return [
        join(data_dir, file_name) for file_name
        in os.listdir(data_dir) if
        re.match(pattern, file_name) is not None
    ]


def get_last_checkpoint(model_dir, pattern):
    """
    Returns the latest file from the directory.
    """
    files = sorted(
        list_files(model_dir, pattern),
        key=lambda x: int(
            re.search(r'\d+', x).group(0)))

    return files[-1] if len(files) > 0 else None


@contextlib.contextmanager
def execute_with_master(args):
    """
    Context manager for performing code blocks
    with master worker only.
    """
    try:
        if args.local_rank in (1, -1):
            yield

    finally:
        if args.distributed:
            barrier()


def load_weights(model_name):
    """
    Load tf checkpoints in a pytorch model.
    """
    ckpt_path = join(
        CACHE_DIR, model_name, 'model.ckpt')

    init_vars = tf.train.list_variables(ckpt_path)
    state_dict = {}

    for name, _ in init_vars:
        weight = tf.train.load_variable(ckpt_path, name)

        # removing the `model`
        name = re.sub(r'h(\d+)', r'h/\1', name[6:])
        name = (
            name.replace('/w', '/weight')
            .replace('/g', '/weight')
            .replace('/b', '/bias')
            .replace('wpe', 'wpe/weight')
            .replace('wte', 'wte/weight')
            .replace('/', '.')
        )

        state_dict[name] = torch.from_numpy(
            weight).squeeze(0)

    return state_dict


def set_random_seed(args):
    """
    Sets the random seed for training.
    """
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    if args.cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def download_config(args):
    """
    Downloads the config for the model.
    """
    dump_path = join(
        CACHE_DIR, args.pretrained, CONFIG_FILE_NAME)

    download_file(dump_path)


def load_config(args):
    """
    Loads the config gile from the cached assets.
    """
    file_path = join(
        CACHE_DIR, args.pretrained, CONFIG_FILE_NAME)

    if not exists(file_path):
        return None

    config = vars(args)

    with open(file_path, 'r') as fh:
        hparams = json.load(fh)

    for key, value in hparams.items():
        if key in config:
            msg = '{} flag with {} value ignored, ' + \
                  'using {} from hparams.json instead.'
            logging.warning(msg.format(
                key, config[key], value))

        config[key] = value

    return argparse.Namespace(**config)


def download_tokenizer(args):
    """
    Downloads the tokenizer asset files.
    """
    for file_name in TOKENIZER_ASSETS:
        download_file(
            join(CACHE_DIR, args.pretrained, file_name))


def load_tokenizer(args):
    """
    Loads the tokenizer for the model.
    """
    encoder_path, vocab_path = [
        join(CACHE_DIR, args.pretrained, file_name)
        for file_name in TOKENIZER_ASSETS
    ]

    if not exists(vocab_path) or not exists(encoder_path):
        return None, None

    tokenizer = get_encoder(args.pretrained, CACHE_DIR)

    vocab_size = len(tokenizer.encoder)

    # additional tokens are placed at the end of the
    # tokenizer vocab
    specials = {
        token: vocab_size + idx for token, idx
        in zip(
            SPECIAL_TOKENS, range(len(SPECIAL_TOKENS)))
    }

    specials['EOS'] = tokenizer.encoder['<|endoftext|>']

    return tokenizer, argparse.Namespace(**specials)


def download_pretrained(args):
    """
    Downloads the model with all of its assets.
    """
    for file_name in MODEL_ASSETS:
        dump_path = join(
            CACHE_DIR, args.pretrained, file_name)

        if not exists(dump_path) or args.force_download:
            download_file(dump_path)


def download_file(dump_path):
    """
    Downloads the provided model from google.
    """
    os.makedirs(
        abspath(dirname(dump_path)), exist_ok=True)

    # determining the dump path for pretrained models
    dump_path = dump_path.replace('\\', '/')
    splits = dump_path.split('/')
    file_name = splits[-1]
    model_name = splits[-2]

    request = requests.get(
        URL + model_name + '/' + file_name, stream=True)

    with open(dump_path, 'wb') as fh:
        file_size = int(request.headers['content-length'])
        chunk_size = 1000

        display_text = file_name if len(file_name) < 15 \
            else file_name[:7] + ' ... ' + file_name[-7:]

        with tqdm(
                ncols=100,
                desc='Downloading {}'.format(
                    colored(
                        display_text, attrs=['bold'])),
                total=file_size,
                unit_scale=True,
                unit='B') as pbar:

            for chunk in request.iter_content(
                    chunk_size=chunk_size):
                fh.write(chunk)
                pbar.update(chunk_size)


class SafeEngine(Engine):
    """
    Simple extension to the Ignite `Engine` class
    to check for exception inside the loop.
    """

    def _run_once_on_dataset(self):
        start_time = time.time()

        try:
            for batch in self.state.dataloader:
                self.state.batch = batch
                self.state.iteration += 1
                self._fire_event(Events.ITERATION_STARTED)

                try:
                    self.state.output = \
                        self._process_function(
                            self, self.state.batch)
                except BaseException as e:
                    self._logger.error(
                        'Current step is terminating '
                        'due to exception: %s.',
                        str(e))
                    self._handle_exception(e)

                self._fire_event(
                    Events.ITERATION_COMPLETED)
                if self.should_terminate or \
                        self.should_terminate_single_epoch:
                    self.should_terminate_single_epoch = \
                        False
                    break

        except BaseException as e:
            self._logger.error(
                'Current run is terminating '
                'due to exception: %s.',
                str(e))
            self._handle_exception(e)

        time_taken = time.time() - start_time
        hours, mins, secs = _to_hours_mins_secs(time_taken)

        return hours, mins, secs

