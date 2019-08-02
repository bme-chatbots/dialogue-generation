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
import json
import copy

from tqdm import tqdm
from itertools import (
    zip_longest, chain)

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
        '--max_history',
        type=str,
        default=4,
        help='Maximum number of turns in history.')
    parser.add_argument(
        '--force_new',
        action='store_true',
        help='If set, the dataset is recreated even if it exists.')


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
    groups = [iter(iterable)] * group_size

    return zip_longest(*groups, fillvalue=fillvalue)


def save_examples(args, data_path, tokenizer):
    """
    Creates numericalizes examples from a raw dailydialog
    file and serializes them.
    """
    name = basename(splitext(data_path)[0])

    special_ids = tokenizer.convert_tokens_to_ids([
        tokenizer.bos_token,
        SP1, SP2
    ])

    dialogues = generate_dialogues(
        args=args,
        data_path=data_path,
        tokenizer=tokenizer)

    # adding lookup indices to source and target
    # pairs for more efficient storage
    dialogues = generate_indices(
        dialogues=dialogues)

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

        examples, indices = zip(
            *list(generate_examples(
                dialogues=group,
                special_ids=special_ids)))

        indices = list(chain(*indices))

        dataset = {
            'examples': examples,
            'indices': indices
        }

        torch.save(dataset, filename)
        num_examples += len(examples)

    return filenames, num_examples


def merge_history(history, sp1_id, sp2_id):
    """
    Merges a list of history sentences into a single
    source example.
    """
    return list(chain(*[
            [sp2_id if idx % 2 == 0 else sp1_id]
            + seq for idx, seq
            in enumerate(history[::-1])
        ][::-1])) + [sp1_id]


def generate_examples(dialogues, special_ids):
    """
    Generates id examples from dialogues.
    """
    sos_id, sp1_id, sp2_id = special_ids

    for example in dialogues:
        if example is None:
            # reaching the last segment of the last
            # file so we can break from the loop
            break

        (history, target), indices = example

        # adding role tokens between segments
        source = merge_history(
            history, sp1_id, sp2_id)

        source = [sos_id] + source

        yield (source, target), indices


def generate_dialogues(args, data_path, tokenizer):
    """
    Generates dialogues from the raw dailydialog file.
    """
    for dialogue in tqdm(read_file(data_path)):
        encoded = [tokenizer.encode(u) for u in dialogue]

        for end_idx in range(1, len(encoded)):
            begin_idx = max(end_idx - args.max_history, 0)
            history = [ids for ids in encoded[begin_idx:end_idx]]
            target = encoded[end_idx]

            eos_id = tokenizer.convert_tokens_to_ids([
                tokenizer.eos_token])[0]
            target.append(eos_id)

            yield history, target


def generate_indices(dialogues):
    """
    Generates input - label indices for the dialogues.
    """
    for idx, (history, target) in enumerate(dialogues):
        # storing the length of the dialogues for
        # the bucket iterator
        indices = [
            (idx, target_idx,
                sum(len(s) for s in history) +
                len(target[:target_idx]))
            for target_idx
            in range(1, len(target))
        ]

        yield (history, target), indices


def read_file(data_path):
    """
    Reads the contents of a raw dailydialog file.
    """
    with open(data_path, 'r') as fh:
        for line in fh:
            dialogue = json.loads(
                line.strip())['dialogue']

            yield [ex['text'] for ex in dialogue]


def create_loader(args, filenames, num_examples,
                  shuffle=False):
    """
    Creates a generator that iterates through the
    dataset.
    """
    def loader():
        """
        Generator that loads examples from files
        lazily.
        """
        file_dataset = FileDataset(filenames)
        file_loader = DataLoader(
            file_dataset,
            collate_fn=lambda x: x[0])

        for examples, indices in file_loader:
            sampler = BucketSampler(
                indices, shuffle=shuffle)

            example_loader = DataLoader(
                DialogDataset(examples, indices),
                batch_size=args.batch_size,
                num_workers=4,
                sampler=sampler,
                pin_memory=True,
                collate_fn=padded_collate)

            yield from example_loader

    return loader


class FileDataset(Dataset):
    """
    Dataset that contains filenames for loading
    lazily.
    """

    def __init__(self, filenames):
        self.filenames = filenames

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        dataset = torch.load(filename)

        indices = dataset['indices']
        examples = dataset['examples']

        return examples, indices

    def __len__(self):
        return len(self.filenames)


class DialogDataset(Dataset):
    """
    Fetches utterances from a list of examples.
    """

    def __init__(self, examples, indices):
        self.examples = examples
        self.indices = indices

    def __getitem__(self, idx):
        example_idx, target_idx, _ = self.indices[idx]
        source, target = self.examples[example_idx]

        inputs = source + target[:target_idx]
        label = target[target_idx]

        # nested lists for convenient parameter
        # passing to collate_fn
        return [inputs, [label]]

    def __len__(self):
        return len(self.indices)


class BucketSampler(Sampler):
    """
    Iterator that creates batches from similar length
    sequences.
    """

    def __init__(self, data_source, bucket_size=1000,
                 shuffle=True):
        self.bucket_size = bucket_size
        self.shuffle = shuffle

        self.sorted = sorted(
            range(len(data_source)),
            key=lambda i: data_source[2])

    def __iter__(self):
        for idx in range(
                0, len(self.sorted), self.bucket_size):
            # divides the data into bucket size segments
            # and only these segment are shuffled
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
        tokenizer.add_special_tokens(
            {'sp1_token': SP1, 'sp2_token': SP2})

        (train_files, train_size), \
            (valid_files, valid_size), \
            (test_files, test_size) = transform(
                args, tokenizer)

        tokenizer.save_pretrained(args.data_dir)

        print('Saving metadata to {}'.format(metadata_path))
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
        print('Loading metadata from {}'.format(metadata_path))
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

    return (train, valid, test), tokenizer
