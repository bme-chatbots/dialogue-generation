"""
@author:    Patrik Purgai
@copyright: Copyright 2019, gpt2-chatbot
@license:   MIT
@email:     purgai.patrik@gmail.com
@date:      2019.08.11.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json
import shutil
import os
import re

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

from tqdm import tqdm

from itertools import (
    zip_longest, chain)

from absl import app, flags
from datetime import datetime
from copy import deepcopy
from math import ceil
from collections import defaultdict
from multiprocessing import Pool

from os.path import (
    exists, join,
    dirname, abspath)


def setup_data_args(parser):
    """
    Parses related arguments.
    """
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='Path of the root data directory.')
    parser.add_argument(
        '--n_workers',
        type=int,
        default=8,
        help='Number of workers for multiprocessing.')
    parser.add_argument(
        '--tfrecord_size',
        type=int,
        default=500000,
        help='Maximum number of examples in a tfrecord.')
    parser.add_argument(
        '--max_size',
        type=int,
        default=500,
        help='Max number of tokens in a sequence.')


def group_elements(iterable, group_size):
    """
    Collect data into fixed-length chunks.
    """
    groups = [iter(iterable)] * group_size

    return zip_longest(*groups)


def int64_feature(value):
    """
    Creates an int64 feature from integers.
    """
    return tf.train.Feature(
        int64_list=tf.train.Int64List(value=value))


def read_file(file_name):
    """
    Reads lines from a file.
    """
    with open(file_name, 'r') as fh:
        for line in fh:
            yield line.strip()


def generate_lines(file_names):
    """
    Generates lines from the provided files.
    """
    for file_name in file_names:
        yield from read_file(file_name)


def generate_examples(file_names):
    """
    Generates examples from the provided files.
    """
    for line in generate_lines(file_names):
        yield json.loads(line)


def transform_split(
        args, file_names, split_name, tokenizer,
        specials):
    """
    Transforms the provided split.
    """
    def is_not_none(e):
        """
        Helper function to filter nulls.
        """
        return e is not None

    def generate_groups():
        """
        Generates groups for serialization.
        """
        groups = group_elements(
            generate_examples(file_names),
            args.tfrecord_size)

        # pairing groups to unique numbersnand 
        # filtering nulls from zip_longest
        groups = (
            (idx, filter(is_not_none, group))
            for idx, group in enumerate(groups))

        yield from groups

    # each group is the content for a single
    # tfrecord file and they are processed
    # concurrently determined by `n_workers`

    dump_path = join(
        args.data_dir, split_name + '.{}.tfrecord')

    pool = Pool(processes=args.n_workers)

    def generate_results():
        """
        Performs serialization and generates
        the resulting file names and sizes.
        """
        for batch in group_elements(
                generate_groups(), args.n_workers):
            # converting iterators to list so resources
            # are not shared in concurrent workers
            batch = filter(is_not_none, batch)
            batch = [(
                list(group),
                tokenizer,
                dump_path.format(idx),
                specials.EOS)
                for idx, group in batch]

            loop = tqdm(
                pool.imap(write_tfrecord, batch),
                desc='Serializing {} split'.format(
                    split_name),
                leave=False)

            yield from loop

    # generates split sizes and filenames 
    # of the tfrecords
    file_names, sizes = zip(*generate_results())

    return file_names, sum(sizes)


def write_tfrecord(params):
    """
    Converts the provided examples to ids and writes
    them to tfrecords.
    """
    examples, tokenizer, file_name, eos_id = params

    def create_feature(dialog):
        """
        Creates a feature list from a document.
        """
        input_ids, type_ids = [], []

        for idx, utterance in enumerate(dialog):
            ids = tokenizer.encode(utterance)
            ids.append(eos_id)

            input_ids.extend(ids)
            type_ids.extend([idx % 2] * len(ids))

        features = {
            'input_ids': int64_feature(list(input_ids)),
            'type_ids': int64_feature(list(type_ids))
        }

        return features

    with tf.io.TFRecordWriter(file_name) as writer:
        for example in examples:

            example = tf.train.Example(
                features=tf.train.Features(
                    feature=create_feature(
                        example['dialog'])))

            writer.write(example.SerializeToString())

    return file_name, len(examples)


def create_loader(
        args, filenames, specials, size, shuffle=False):
    """
    Creates a generator that iterates through the
    dataset.
    """
    pad_id = tf.constant(specials.PAD, tf.int64)

    def parse_example(example):
        """
        Parses a dialog from the serialized datafile.
        """
        features = {
            'input_ids': tf.io.VarLenFeature(tf.int64),
            'type_ids': tf.io.VarLenFeature(tf.int64)
        }

        parsed_example = \
            tf.io.parse_single_example(
                example, features=features)

        parsed_example['input_ids'] = \
            tf.sparse.to_dense(
                parsed_example['input_ids']
            )[:args.max_size]

        parsed_example['type_ids'] = \
            tf.sparse.to_dense(
                parsed_example['type_ids']
            )[:args.max_size]

        return parsed_example

    def compute_length(example):
        """
        Computes the length of the example.
        """
        return tf.size(example['input_ids'])

    def prepare_inputs(example):
        """
        Creates the attention mask tensor and replaces
        1 and 0s in type id tensor with SP ids.
        """
        example['attn_mask'] = tf.equal(
            example['input_ids'], pad_id)

        type_ids = tf.dtypes.cast(
            example['type_ids'], tf.bool)

        cond = type_ids if \
            tf.random.uniform(shape=(1, )) > 0.5 else \
            ~type_ids

        example['type_ids'] = tf.where(
            cond, x=specials.SP1, y=specials.SP2)

        return example

    dataset = tf.data.TFRecordDataset(filenames)

    if args.distributed:
        dataset = dataset.shard(
            num_shards=args.world_size,
            index=args.local_rank)

    dataset = dataset.shuffle(1000)
    dataset = dataset.map(
        parse_example,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # TODO this should be tuned for TPU support
    # find solution for converting dataset with
    # strategy
    dataset = dataset.apply(
        tf.data.experimental.bucket_by_sequence_length(
            element_length_func=compute_length,
            bucket_batch_sizes=[
                args.batch_size,
                args.batch_size
            ],
            bucket_boundaries=[200],
            padded_shapes={
                'input_ids': tf.TensorShape([None]),
                'type_ids': tf.TensorShape([None])
            },
            padding_values={
                'input_ids': pad_id,
                'type_ids': pad_id
            }))

    # creating attention masking tensor
    dataset = dataset.map(
        prepare_inputs,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset = dataset.prefetch(
        tf.data.experimental.AUTOTUNE)

    return DatasetWrapper(args, dataset, size)


class DatasetWrapper:
    """
    Wrapers the tf.data.Dataset as an object
    with similar interface as torch.utils.Dataset.
    """

    def __init__(
            self, args, dataset, size):
        self.dataset = dataset
        self.size = size
        self.samples_per_step = args.batch_size * \
            args.world_size

    def __iter__(self):
        return iter(self.dataset)

    def __len__(self):
        return ceil(self.size / self.samples_per_step)


def parse_config(data_dir, config):
    """
    Returns the training, validation and test files.
    """
    def list_files(pattern):
        return tf.io.gfile.glob(join(data_dir, pattern))

    splits = [
        (list_files(config[split]), split)
        for split in ['train', 'valid']]

    assert all(len(s) for s in splits), \
        'At splits must have at least one file.'

    return splits


def create_dataset(args, tokenizer, specials):
    """
    Transforms the dataset and provides iterators to it.
    """
    assert exists(args.data_dir), \
        '{} does not exist.'.format(args.data_dir)

    metadata_path = join(args.data_dir, 'metadata.json')
    config_path = join(args.data_dir, 'config.json')

    if not exists(metadata_path):
        # if dataset does not exist then create it
        # by tokenizing the raw files

        assert exists(config_path), \
            'No `config.json` in {}'.format(args.data_dir)

        with open(config_path, 'r') as fh:
            config = json.load(fh)

        splits = parse_config(args.data_dir, config)

        train, valid = [
            transform_split(
                args=args,
                file_names=file_names,
                split_name=split_name,
                tokenizer=tokenizer,
                specials=specials)
            for file_names, split_name in splits]

        train_files, train_size = train
        valid_files, valid_size = valid

        print('Saving metadata to {}'.format(
            metadata_path))

        # save the location of the files in a metadata
        # json object and delete the file in case of
        # interrupt so it wont be left in corrupted state
        with open(metadata_path, 'w') as fh:
            try:
                json.dump({
                    'train': train,
                    'valid': valid,
                }, fh)
            except KeyboardInterrupt:
                shutil.rmtree(metadata_path)

    else:
        print('Loading metadata from {}'.format(
            metadata_path))

        with open(metadata_path, 'r') as fh:
            metadata = json.load(fh)

        train_files, train_size = metadata['train']
        valid_files, valid_size = metadata['valid']

    with tf.device('cpu'):
        train_dataset = create_loader(
            args=args,
            filenames=train_files,
            size=train_size,
            specials=specials,
            shuffle=True)

        valid_dataset = create_loader(
            args=args,
            filenames=valid_files,
            size=valid_size,
            specials=specials)

    return train_dataset, valid_dataset

