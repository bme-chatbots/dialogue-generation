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

from tqdm import tqdm
from itertools import (
    zip_longest, chain)

from math import ceil
from copy import deepcopy

from os.path import (
    exists, join,
    dirname, abspath,
    basename, splitext)

from torch.utils.data import (
    Dataset, DataLoader,
    Sampler)

from torch.utils.data.distributed import (
    DistributedSampler)

from pytorch_transformers import (
    XLNetTokenizer)

from collate import padded_collate


SP1 = '<sp1>'
SP2 = '<sp2>'
HST = '<hst>'
SRC = '<src>'  # unused currently
RSP = '<rsp>'


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
        default=3,
        help='Maximum number of turns in history.')


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

    train_path = join(args.data_dir, 'train.json')
    valid_path = join(args.data_dir, 'valid.json')
    test_path = join(args.data_dir, 'test.json')

    if not exists(train_path) or not \
            exists(valid_path) or not exists(test_path):
        print('Extracting dataset to {}'.format(
            args.data_dir))
        shutil.unpack_archive(download_path, args.data_dir)


def transform(args, tokenizer):
    """
    Transforms the dataset to numericalized format.
    """
    print('Transforming dataset')

    train_path = join(args.data_dir, 'train.json')
    train = save_examples(args, train_path, tokenizer)

    valid_path = join(args.data_dir, 'valid.json')
    valid = save_examples(args, valid_path, tokenizer)

    test_path = join(args.data_dir, 'test.json')
    test = save_examples(args, test_path, tokenizer)

    return train, valid, test


def group_elements(iterable, group_size, fillvalue=None):
    """
    Collect data into fixed-length chunks.
    """
    groups = [iter(iterable)] * group_size

    return zip_longest(*groups, fillvalue=fillvalue)


def save_examples(args, data_path, tokenizer):
    """
    Creates numericalizes examples from a raw dailydialog
    file and serializes them.
    """
    name = basename(splitext(data_path)[0])

    # during data preprocessing the history and target
    # utterances are saved only once
    dialogues = generate_dialogues(
        args=args,
        data_path=data_path,
        tokenizer=tokenizer)

    groups = group_elements(
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
                dialogues=group)))

        indices = list(chain(*indices))

        dataset = {
            'examples': examples,
            'indices': indices
        }

        torch.save(dataset, filename)
        num_examples += len(indices)

    return filenames, num_examples


def generate_examples(dialogues):
    """
    Generates id examples from dialogues.
    """
    for example in dialogues:
        if example is None:
            # reaching the last segment of the last
            # file so we can break from the loop
            break

        dialogue, indices = example

        yield dialogue, indices


def generate_dialogues(args, data_path, tokenizer):
    """
    Generates dialogues from the raw dailydialog file.
    """
    for dialogue_idx, dialogue in tqdm(
            enumerate(read_file(data_path))):
        dialogue = [tokenizer.encode(u) for u in dialogue]
        dialogue_indices = list(range(len(dialogue)))

        # generating indices list that indexes into
        # the dialogue and creates a history and a target
        # this way the text data only has to be stored once
        indices = []

        for end_idx in range(1, len(dialogue)):
            history_indices = \
                dialogue_indices[:end_idx][-args.max_history:]

            target_utterance = dialogue[end_idx]
            for begin_idx in history_indices:
                for label_idx in range(len(target_utterance)):
                    # adding 1 to lengths because role_id will
                    # be appended to every utterance in the
                    # `transform_history` function
                    size = sum(
                        len(dialogue[i]) + 1 for i
                        in history_indices) + \
                        label_idx + 1

                    indices.append((
                        dialogue_idx,
                        begin_idx, end_idx,
                        label_idx, size))

        yield dialogue, indices


def read_file(data_path):
    """
    Reads the contents of a raw dailydialog file.
    """
    with open(data_path, 'r') as fh:
        for line in fh:
            dialogue = json.loads(
                line.strip())['dialogue']

            yield [ex['text'] for ex in dialogue]


def create_loader(args, filenames, tokenizer,
                  distributed, shuffle=False):
    """
    Creates a generator that iterates through the
    dataset.
    """
    special_ids = tokenizer.convert_tokens_to_ids([
        SP1, SP2, tokenizer.bos_token, 
        tokenizer.eos_token, HST, RSP,
        tokenizer.mask_token
    ])

    # distributed training is used if the local
    # rank is not the default -1
    sampler_cls = DistributedSampler if \
        distributed else IndexSampler

    bucket_sampler_cls = create_sampler_cls(
        sampler_cls=sampler_cls)

    def load_examples():
        """
        Generator that loads examples from files
        lazily.
        """
        file_dataset = FileDataset(filenames)
        file_loader = DataLoader(
            file_dataset,
            collate_fn=lambda x: x[0])

        for dialogues, indices in file_loader:
            sampler = bucket_sampler_cls(
                indices, shuffle=shuffle)

            dialog_dataset = DialogDataset(
                dialogues=dialogues, indices=indices,
                special_ids=special_ids)

            example_loader = DataLoader(
                dialog_dataset,
                batch_size=args.batch_size,
                num_workers=4, sampler=sampler,
                pin_memory=True,
                collate_fn=padded_collate)

            yield from example_loader

    return load_examples


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

    def __init__(self, dialogues, indices, special_ids):
        self.dialogues = dialogues
        self.indices = indices

        self.sp1_id, self.sp2_id, self.sos_id, \
            self.eos_id, self.hst_id, self.rsp_id, \
            self.mask_id = special_ids
    
    def __getitem__(self, idx):
        dialogue_idx, begin_idx, end_idx, \
            target_idx, _ = self.indices[idx]
        dialogue = self.dialogues[dialogue_idx]

        # the whole dialogue is fetched and the
        # `idx` element of indices array creates
        # the example
        history = dialogue[begin_idx:end_idx]
        response = dialogue[end_idx]

        # the model only predict a single token
        # the rest of the response is added to the
        # inputs as "already generated"
        generated = response[:target_idx]
        label = response[target_idx]
        
        input_ids, token_type_ids = [], []
        # iterating on reversed history because the last
        # utterance is always from speaker2 thus it is
        # easier to assign the speaker id
        for idx, utr in enumerate(history[::-1]):
            role_id = self.sp2_id if idx % 2 == 0 \
                else self.sp1_id
            
            ids = [role_id] + utr

            input_ids.append(ids)

            # the first element of the history is the
            # source utterance which gets a different
            # role token than the other parts of history

            # NOTE only using hst and rsp types currently
            # type_id = src_id if idx == 0 else hst_id
            type_id = self.hst_id

            token_type_ids.append([type_id] * len(ids))

        # adding special ids to the dialogue history
        input_ids = [self.sos_id] + \
            list(chain(*input_ids[::-1])) + [self.sp1_id]
        # adding the token type id of the start id as well
        token_type_ids = [type_id] + \
            list(chain(*token_type_ids[::-1])) + \
            [self.rsp_id]

        input_ids = deepcopy(input_ids)
        token_type_ids = deepcopy(token_type_ids)

        # adding the first `target_idx` num of target
        # ids to inputs and the mask_id
        input_ids.extend(generated)
        input_ids.append(self.mask_id)

        # the token types are extended with the
        # type ids of the previous extension
        token_type_ids.extend(
            [self.rsp_id] * (len(generated) + 1))

        # returning nested lists for convenient
        # parameter passing to collate_fn
        return [input_ids, token_type_ids, [label]]

    def __len__(self):
        return len(self.indices)


class IndexSampler(Sampler):
    """
    Dummy class for sampling indices in range
    `len(data_source)`. The purpose of this class
    is to provide the same behaviour as 
    `DistributedSampler` in single world environment.
    """

    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        return iter(range(len(self.data_source)))

    def __len__(self):
        return len(self.data_source)


def create_sampler_cls(sampler_cls):
    """
    Creates a bucketized sampler class.
    """
    class BucketSampler(sampler_cls):
        """
        Bucketized sampler that yields exclusive groups
        of indices based on the sequence length.
        Similar sized examples are assigned to the same group.
        """

        def __init__(self, data_source, bucket_size=5000,
                     shuffle=True):
            super().__init__(data_source)
            self.bucket_size = bucket_size
            self.shuffle = shuffle

            # `data_source` here is the `indices` list
            # which contains tuples where the last element
            # is the size of the example
            self.sorted = sorted(
                list(super().__iter__()),
                key=lambda i: data_source[i][-1])

        def __iter__(self):
            # divides the data into bucket size segments
            # and only these segment are shuffled
            segments = [
                self.sorted[idx: idx + self.bucket_size]
                for idx in range(0, len(self.sorted),
                                 self.bucket_size)]

            # selecting seqgemnts in random order
            random.shuffle(segments)
            for segment in segments:

                if self.shuffle:
                    random.shuffle(segment)

                yield from segment

    return BucketSampler


def create_dataset(args, device, distributed):
    """
    Downloads the DailyDialog dataset, converts it
    to tokens and returns iterators over the train and
    test splits.
    """
    metadata_path = join(
        args.data_dir, 'metadata.json')

    if not exists(metadata_path):
        # if dataset does not exist then create it
        # downloading and tokenizing the raw files
        download(args)

        tokenizer = XLNetTokenizer.from_pretrained(
            'xlnet-base-cased')

        tokenizer.add_special_tokens({
            'sp1_token': SP1, 'sp2_token': SP2,
            'hst_token': HST, 'src_token': SRC,
            'rsp_token': RSP
        })

        transformed = transform(args, tokenizer)
        train, valid, test = transformed

        train_files, train_size = train
        valid_files, valid_size = valid
        test_files, test_size = test

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
        print('Loading metadata from {}'.format(
            metadata_path))
        with open(metadata_path, 'r') as fh:
            filenames = json.load(fh)

        train_files, train_size = filenames['train']
        valid_files, valid_size = filenames['valid']
        test_files, test_size = filenames['test']

        tokenizer = XLNetTokenizer.from_pretrained(
            args.data_dir)

    train_dataset = create_loader(
        args=args,
        filenames=train_files,
        tokenizer=tokenizer,
        distributed=distributed,
        shuffle=True)

    valid_dataset = create_loader(
        args=args,
        filenames=valid_files,
        distributed=distributed,
        tokenizer=tokenizer)

    test_dataset = create_loader(
        args=args,
        filenames=test_files,
        distributed=distributed,
        tokenizer=tokenizer)

    train = train_dataset, train_size
    valid = valid_dataset, valid_size
    test = test_dataset, test_size

    return (train, valid, test), tokenizer
