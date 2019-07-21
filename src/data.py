"""
@author:    Patrik Purgai
@copyright: Copyright 2019, dialogue-generation
@license:   MIT
@email:     purgai.patrik@gmail.com
@date:      2019.07.20.
"""

# pylint: disable=no-name-in-module
# pylint: disable=import-error

import torch
import requests
import shutil
import os
import random
import re
import json
import copy

from tqdm import tqdm
from itertools import zip_longest, chain
from math import ceil

from os.path import (
    exists, join,
    dirname, abspath,
    basename, splitext)

from torch.utils.data import (
    Dataset, DataLoader,
    Sampler)

from pytorch_transformers import (
    XLNetTokenizer)

from collate import padded_collate


SP1 = '<sp1>'
SP2 = '<sp2>'


def setup_data_args(parser):
    """
    Sets up the data arguments.
    """
    parser.add_argument(
        '--data_dir',
        type=str,
        default=join(abspath(dirname(__file__)), '..', 'data'),
        help='Path of the data root directory.')
    parser.add_argument(
        '--download_dir',
        type=str,
        default=join(abspath(dirname(__file__)), '..', 'data'),
        help='Path of the download directory.')
    parser.add_argument(
        '--file_size',
        type=int,
        default=100000,
        help='Max number of examples in a single file.')
    parser.add_argument(
        '--max_len',
        type=int,
        default=100,
        help='Maximum length of the sequences.')
    parser.add_argument(
        '--max_history',
        type=str,
        default=4,
        help='Maximum number of turns in history.')
    parser.add_argument(
        '--force_new',
        action='store_true',
        help='If set recreates the dataset even if it exists.')


def download(args):
    """
    Downloads and extracts the daily dialog dataset from 
    parlai.
    """
    base_url = 'http://parl.ai/downloads/dailydialog/'
    filename = 'dailydialog.tar.gz'

    if not exists(args.download_dir):
        os.mkdir(args.download_dir)

    if not exists(args.data_dir):
        os.mkdir(args.data_dir)

    url = base_url + filename
    download_path = join(args.download_dir, filename)

    if not exists(download_path):
        print('Downloading dataset to {}'.format(
            download_path))
        with requests.Session() as session:
            response = session.get(
                url, stream=True, timeout=5)

            with open(download_path, 'wb') as f:
                for chunk in tqdm(response.iter_content(
                        2 ** 15)):
                    if chunk:
                        f.write(chunk)

    extract_path = join(args.data_dir, 'dailydialog')

    if not exists(extract_path):
        print('Extracting dataset to {}'.format(
            extract_path))
        shutil.unpack_archive(download_path, args.data_dir)


def transform(args, tokenizer):
    """
    Transforms the dataset to numericalized format.
    """
    extract_path = join(args.data_dir, 'dailydialog')

    print('Transforming dataset')

    train_path = join(extract_path, 'train.json')
    train = save_examples(args, train_path, tokenizer)

    valid_path = join(extract_path, 'valid.json')
    valid = save_examples(args, valid_path, tokenizer)

    test_path = join(extract_path, 'test.json')
    test = save_examples(args, test_path, tokenizer)

    return train, valid, test


def grouper(iterable, group_size, fillvalue=None):
    """
    Collect data into fixed-length chunks or blocks
    """
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * group_size

    return zip_longest(*args, fillvalue=fillvalue)


def save_examples(args, data_path, tokenizer):
    """
    Creates numericalizes examples from a raw dailydialog
    file and serializes them.
    """
    name = basename(splitext(data_path)[0])

    special_ids = tokenizer.convert_tokens_to_ids([
        tokenizer.bos_token,
        tokenizer.eos_token,
        SP1, SP2
    ])

    dialogues = generate_dialogues(
        args=args,
        data_path=data_path,
        tokenizer=tokenizer)

    groups = grouper(
        iterable=dialogues,
        group_size=args.file_size)

    num_examples = 0
    filenames = []
    for idx, group in enumerate(groups):
        filename = join(
            args.data_dir,
            '{}{}.pt'.format(name, idx))

        filenames.append(filename)

        examples = [
            (src, trg) for src, trg in
            generate_examples(group, special_ids)
        ]

        torch.save({'examples': examples}, filename)
        num_examples += len(examples)

    return filenames, num_examples


def generate_examples(dialogues, special_ids):
    """
    Generates id examples from dialogues.
    """
    sos_id, eos_id, sp1_id, sp2_id = special_ids

    for example in dialogues:
        if example is None:
            # reaching the last segment of the last
            # file so we can break from the loop
            break

        source, target = example
        source = list(chain(*[
            [sp1_id if idx % 2 == 0 else sp2_id]
            + seq for idx, seq
            in enumerate(source[::-1])
        ][::-1]))

        # only the source sentence must have an
        # sos provided because the target is sos
        # is created by the model

        source = [sos_id] + source + [eos_id]
        target = target + [eos_id]

        yield source, target


def generate_dialogues(args, data_path, tokenizer):
    """
    Generates dialogues from the raw dailydialog file.
    """
    for dialogue in tqdm(read_file(data_path)):
        ids = []
        for utterance in dialogue:
            ids.append(tokenizer.encode(utterance))

        for end_idx in range(1, len(ids)):
            begin_idx = max(end_idx - args.max_history, 0)
            source = [text for text in ids[begin_idx:end_idx]]
            target = ids[end_idx]

            yield source, target


def read_file(data_path):
    """
    Reads the contents of a raw dailydialog file.
    """
    with open(data_path, 'r') as fh:
        for line in fh:
            dialogue = json.loads(
                line.strip())['dialogue']

            yield [ex['text'] for ex in dialogue]


def create_loader(args, filenames, num_examples, shuffle=False):
    """
    Creates a generator that iterates through the
    dataset.
    """
    file_dataset = FileDataset(filenames)
    file_loader = DataLoader(
        file_dataset, 
        collate_fn=lambda x: x[0])

    for examples in file_loader:
        sampler = BucketSampler(
            examples, shuffle=shuffle)

        example_loader = DataLoader(
            examples, 
            batch_size=args.batch_size,
            num_workers=4, 
            sampler=sampler, 
            pin_memory=True, 
            collate_fn=padded_collate)

        yield from example_loader


class FileDataset(Dataset):

    def __init__(self, filenames):
        self.filenames = filenames

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        examples = torch.load(
            filename)['examples']

        return examples

    def __len__(self):
        return len(self.filenames)


class BucketSampler(Sampler):

    def __init__(self, data_source, bucket_size=1000,
                 shuffle=True):
        self.bucket_size = bucket_size
        self.shuffle = shuffle

        self.sorted = sorted(
            range(len(data_source)),
            key=lambda i: len(data_source[i][0]))

    def __iter__(self):
        for idx in range(
                0, len(self.sorted), self.bucket_size):
            indices = self.sorted[idx: idx + self.bucket_size]
            indices = list(copy.deepcopy(indices))

            if self.shuffle:
                random.shuffle(indices)

            yield from indices

    def __len__(self):
        return len(self.sorted)


def create_dataset(args, device):
    """
    Downloads the DailyDialog dataset, converts it
    to tokens and returns iterators over the train and
    test splits.
    """
    metadata_path = join(
        args.data_dir, 'metadata.json')

    if args.force_new:
        shutil.rmtree(metadata_path)

    if not exists(metadata_path):
        # if dataset does not exist then create it
        # downloading and tokenizing the raw files
        download(args)

        tokenizer = XLNetTokenizer.from_pretrained(
            'xlnet-base-cased')
        tokenizer.add_tokens([SP1, SP2])

        (train_files, train_size), \
            (valid_files, valid_size), \
            (test_files, test_size) = transform(
                args, tokenizer)

        tokenizer.save_pretrained(args.data_dir)

        # save the location of the files in a metadata
        # json object and delete the file in case of
        # interrupt so it wont be left in corrupted state
        with open(metadata_path, 'w') as fh:
            try: 
                json.dump({
                    'train': [train_files, train_size],
                    'valid': [valid_files, valid_size],
                    'test': [test_files, test_size]
                }, fh)
            except KeyboardInterrupt:
                shutil.rmtree(metadata_path)

    else:
        with open(metadata_path, 'r') as fh:
            filenames = json.load(fh)

        train_files, train_size = filenames['train']
        valid_files, valid_size = filenames['valid']
        test_files, test_size = filenames['test']

        tokenizer = XLNetTokenizer.from_pretrained(
            args.data_dir)

    train = create_loader(
        args=args, 
        filenames=train_files, 
        num_examples=train_size,
        shuffle=True)

    valid = create_loader(
        args=args, 
        filenames=valid_files, 
        num_examples=valid_size)

    test = create_loader(
        args=args, 
        filenames=test_files, 
        num_examples=test_size)

    vocab_size = len(tokenizer)

    return train, valid, test, vocab_size
