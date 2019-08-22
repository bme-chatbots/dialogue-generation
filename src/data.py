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
from collections import namedtuple

from torch.utils.data import (
    Dataset, DataLoader,
    Sampler)

from torch.utils.data.distributed import (
    DistributedSampler)

from pytorch_transformers import (
    XLNetTokenizer,
    GPT2Tokenizer)

from src.collate import COLLATE
from src.model import MODEL

from os.path import (
    exists, join,
    dirname, abspath,
    basename, splitext)


SP1 = '<sp1>'
SP2 = '<sp2>'
HST = '<hst>'
RSP = '<rsp>'
PAD = '<pad>'


def setup_data_args(parser):
    """
    Sets up the data arguments.
    """
    group = parser.add_argument_group('data')
    group.add_argument(
        '--data_name',
        type=str,
        default='dailydialog',
        help='Name of the dataset to use.')
    group.add_argument(
        '--data_dir',
        type=str,
        default=join(abspath(dirname(__file__)), '..', 'data'),
        help='Path of the data root directory.')
    group.add_argument(
        '--download_dir',
        type=str,
        default=join(abspath(dirname(__file__)), '..', 'data'),
        help='Path of the download directory.')
    group.add_argument(
        '--file_size',
        type=int,
        default=100000,
        help='Max number of examples in a single file.')
    group.add_argument(
        '--max_hist',
        type=str,
        default=4,
        help='Maximum number of turns in history.')


def group_elements(iterable, group_size):
    """
    Collect data into fixed-length chunks.
    """
    groups = [iter(iterable)] * group_size

    return zip_longest(*groups)


def save_examples(args, data_cls, data_path, tokenizer):
    """
    Creates numericalizes examples from a raw data
    file and serializes them.
    """
    name = basename(splitext(data_path)[0])
    data_dir = join(args.data_dir, args.data_name,
                    args.model_name)

    # during data preprocessing the history 
    # and target utterances are saved only once
    dialogs = generate_dialogs(
        args=args,
        data_cls=data_cls,
        data_path=data_path,
        tokenizer=tokenizer)

    groups = group_elements(
        iterable=dialogs,
        group_size=args.file_size)

    num_examples = 0
    filenames = []
    for idx, group in enumerate(groups):
        filename = '{}.{}.pt'.format(name, idx)
        filename = join(data_dir, filename)

        filenames.append(filename)

        examples, indices = zip(
            *list(generate_examples(
                dialogs=group)))

        indices = list(chain(*indices))

        dataset = {
            'examples': examples,
            'indices': indices
        }

        torch.save(dataset, filename)
        num_examples += len(indices)

    return filenames, num_examples


def generate_examples(dialogs):
    """
    Generates id examples from dialogs.
    """
    for example in dialogs:
        if example is None:
            # reaching the last segment of the last
            # file so we can break from the loop
            break

        dialog, indices = example

        yield dialog, indices


def generate_dialogs(args, data_cls, data_path, tokenizer):
    """
    Generates dialogs from the raw dailydialog file.
    """
    def generate_indices(dialog_idx, encoded_dialog):
        dialog_indices = list(range(len(encoded_dialog)))

        # starting from index 1 because there is always
        # at least 2 utterances in a dialog
        for end_idx in range(1, len(encoded_dialog)):
            hist_indices = dialog_indices[:end_idx]
            hist_indices = hist_indices[-args.max_hist:]

            target_utr = encoded_dialog[end_idx]

            for begin_idx in hist_indices:
                # adding 1 to length because role_id will
                # be appended to every utterance in the
                # `transform_dialog` function
                example_len = \
                    sum(len(encoded_dialog[i]) + 1
                        for i in hist_indices)

                # `len(target_utr) + 1` because
                # `eos_id` is not added yet and therefore it
                # is not included in target length
                target_len = len(target_utr) + 1
                example_len += target_len

                yield (dialog_idx, begin_idx,
                       end_idx, example_len)

    content = data_cls.read_file(data_path)

    for idx, dialog in tqdm(enumerate(content)):
        dialog = [tokenizer.encode(u) for u in dialog]

        # generating indices list that indexes into
        # the dialog and slices a history and a target
        # this way the text data only has to be stored once
        indices = list(generate_indices(
            dialog_idx=idx, encoded_dialog=dialog))

        yield dialog, indices


def transform_dialog(history, special_ids):
    """
    Transforms a dialog and creates `input_ids`
    and `token_type_ids` lists.
    """
    sp1_id, sp2_id, sos_id, _, hst_id, rsp_id = \
        special_ids

    input_ids, token_type_ids = [], []
    # iterating on reversed history because the last
    # utterance is always from speaker2 thus it is
    # easier to assign the speaker id
    for idx, utr in enumerate(history[::-1]):
        role_id = sp2_id if idx % 2 == 0 else sp1_id

        ids = [role_id] + utr
        input_ids.append(ids)

        type_id = hst_id
        token_type_ids.append([type_id] * len(ids))

    # adding the history conversations
    # reversing the order back to original
    input_ids = \
        list(chain(*input_ids[::-1]))
    token_type_ids = \
        list(chain(*token_type_ids[::-1]))

    # adding the initial start token
    input_ids.insert(0, sos_id)
    token_type_ids.insert(0, type_id)

    # adding the initial token of the response
    input_ids.append(sp1_id)
    token_type_ids.append(rsp_id)

    return input_ids, token_type_ids


class FileDataset(Dataset):
    """
    Dataset that contains filenames of dataset
    fragments and loads them lazily.
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
    The examples are produced from subsets of dialogs.
    """

    # base url to download the data
    url = ''

    # name of the dataset
    name = ''

    # name of the downloaded archive
    archive = ''

    # list of the extracted filenames
    files = []

    @classmethod
    def download(cls, args):
        """
        Downloads and extracts the daily dialog 
        dataset from parlai.
        """
        extract_dir = join(
            args.data_dir, args.data_name)
        download_dir = join(
            args.download_dir, args.data_name)

        os.makedirs(extract_dir, exist_ok=True)
        os.makedirs(download_dir, exist_ok=True)

        url = cls.url + cls.archive
        download_path = join(download_dir, cls.archive)

        if not exists(download_path):
            if args.local_rank in [-1, 0]:
                print('Downloading dataset to {}'.format(
                    download_path))

            with requests.Session() as session:
                response = session.get(
                    url, stream=True, timeout=5)

                # data is read in 2 ** 15 sized chunks
                # NOTE this could be tuned to reveal
                # data size in MBs
                loop = response.iter_content(2 ** 20)
                loop = tqdm(loop, unit='MB', unit_scale=True)

                with open(download_path, 'wb') as f:
                    for chunk in loop:
                        if chunk:
                            f.write(chunk)

        extracted = [
            join(extract_dir, f) for f in cls.files]

        if any(not exists(p) for p in extracted):
            print('Extracting dataset to {}'.format(
                extract_dir))
            shutil.unpack_archive(
                download_path, extract_dir)

        return extracted

    @classmethod
    def transform(cls, args, files, tokenizer):
        """
        Transforms the dataset into numericalized
        format and saves it in fragments.
        """
        print('Transforming dataset')

        return [
            save_examples(
                args=args, 
                data_cls=cls,
                data_path=data_path, 
                tokenizer=tokenizer)
            for data_path in files
        ]

    @classmethod
    def subclasses(cls):
        """
        Lists the available datasets.
        """
        def generate_subclasses(c):
            for s in c.__subclasses__():
                # recursively runs through the
                # subclasses of dialog datasets
                yield from generate_subclasses(s)
                yield s

        # filtering out possible duplicates
        subclasses = set(generate_subclasses(cls))

        return {s.name: s for s in subclasses}

    def __init__(self, dialogs, indices, special_ids):
        self.dialogs = dialogs
        self.indices = indices
        self.special_ids = special_ids

    def __getitem__(self, idx):
        dialog_idx, begin_idx, end_idx, seq_len = \
            self.indices[idx]
        dialog = self.dialogs[dialog_idx]

        # the whole dialog is fetched and the
        # `idx` element of indices array creates
        # the example
        eos_id = self.special_ids[3]
        rsp_id = self.special_ids[5]

        history = dialog[begin_idx:end_idx]
        target = dialog[end_idx] + [eos_id]

        inputs = transform_dialog(
            history=history,
            special_ids=self.special_ids)

        input_ids, token_type_ids = inputs

        input_ids.extend(target)
        token_type_ids.extend([rsp_id] * len(target))

        # returning nested lists for convenient
        # parameter passing to collate_fn
        return [
            input_ids, token_type_ids, 
            target, [seq_len]
        ]

    def __len__(self):
        return len(self.indices)


class IndexSampler(Sampler):
    """
    Dummy class for sampling indices in range
    `len(data_source)`. Provides same behavior as
    `DistributedSampler` in single world environment.
    """

    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        return iter(range(len(self.data_source)))

    def __len__(self):
        return len(self.data_source)


def create_loader(args, filenames, tokenizer,
                  data_cls, shuffle=False):
    """
    Creates a generator that iterates through the
    dataset.
    """
    special_ids = tokenizer.convert_tokens_to_ids([
        SP1, SP2, tokenizer.bos_token,
        tokenizer.eos_token, HST, RSP
    ])

    # distributed training is used if the local
    # rank is not the default -1
    sampler_cls = DistributedSampler if \
        args.distributed else IndexSampler

    bucket_sampler_cls = create_sampler_cls(
        sampler_cls=sampler_cls)

    collate_fn = COLLATE[args.model_name]

    def load_examples():
        """
        Generator that loads examples from files
        lazily.
        """
        file_dataset = FileDataset(filenames)
        file_loader = DataLoader(
            file_dataset,
            collate_fn=lambda x: x[0])

        for dialogs, indices in file_loader:
            sampler = bucket_sampler_cls(
                indices, shuffle=shuffle)

            dialog_dataset = data_cls(
                dialogs=dialogs,
                indices=indices,
                special_ids=special_ids)

            example_loader = DataLoader(
                dialog_dataset,
                batch_size=args.batch_size,
                num_workers=4,
                sampler=sampler,
                pin_memory=True,
                collate_fn=collate_fn)

            yield from example_loader

    return load_examples


TOKENIZER = {
    'xlnet': XLNetTokenizer, 
    'gpt2': GPT2Tokenizer
}


def create_tokenizer(args):
    """
    Creates the tokenizer for the model and saves
    it in the model data directory if it does not exist.
    """
    data_dir = join(args.data_dir, args.data_name,
                    args.model_name)
    tokenizer_path = join(data_dir, 'special_tokens_map.json')

    tokenizer_cls = TOKENIZER[args.model_name]
    model_cls = MODEL[args.model_name]

    if not exists(tokenizer_path):
        # TODO come up with better naming
        # for tokenizer `instance`
        instance = tokenizer_cls.from_pretrained(
            model_cls.config)

        # adding special tokens
        # TODO check compatibility with all tokenizers
        special_tokens = [SP1, SP2, HST, RSP]
        instance.add_special_tokens({
            'additional_special_tokens': special_tokens})

        if instance.pad_token is None:
            # GPT2 does not have a pad token
            instance.add_special_tokens({
                'pad_token': PAD})

        instance.save_pretrained(data_dir)

    else:
        instance = tokenizer_cls.from_pretrained(
            data_dir)

    return instance


def create_dataset(args):
    """
    Downloads the dataset, converts it to tokens and 
    returns iterators over the train and test splits.
    """
    data_dir = join(args.data_dir, args.data_name,
                    args.model_name)
    os.makedirs(data_dir, exist_ok=True)

    metadata_path = join(data_dir, 'metadata.json')

    tokenizer = create_tokenizer(args)
    data_cls = create_data_cls(args)

    if not exists(metadata_path):
        # if dataset does not exist then create it
        # downloading and tokenizing the raw files
        files = data_cls.download(args=args)

        train, valid, test = data_cls.transform(
            args=args, files=files, tokenizer=tokenizer)

        train_files, train_size = train
        valid_files, valid_size = valid
        test_files, test_size = test

        print('Saving metadata to {}'.format(
            metadata_path))
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

    train_dataset = create_loader(
        args=args, 
        filenames=train_files,
        tokenizer=tokenizer, 
        data_cls=data_cls,
        shuffle=True)

    valid_dataset = create_loader(
        args=args, 
        filenames=valid_files,
        tokenizer=tokenizer,
        data_cls=data_cls)

    test_dataset = create_loader(
        args=args, 
        filenames=test_files,
        tokenizer=tokenizer,
        data_cls=data_cls)

    train = train_dataset, train_size
    valid = valid_dataset, valid_size
    test = test_dataset, test_size

    return (train, valid, test), tokenizer


def create_sampler_cls(sampler_cls):
    """
    Creates a bucketized sampler class.
    """
    class BucketSampler(sampler_cls):
        """
        Bucketized sampler that yields exclusive groups
        of indices based on the sequence length.
        """

        def __init__(self, data_source, bucket_size=1000,
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
            def generate_indices(group):
                for idx in group:
                    if idx is not None:
                        yield idx

            groups = group_elements(
                iterable=self.sorted,
                group_size=self.bucket_size)

            # TODO shuffling groups should be
            # deterministic with regards to epoch
            groups = list(groups)
            random.shuffle(groups)

            for group in groups:
                indices = list(generate_indices(group))

                if self.shuffle:
                    indices = deepcopy(indices)
                    random.shuffle(indices)

                yield from indices

    return BucketSampler


def create_data_cls(args):
    """
    Creates a data class based on the provided
    data name and model name.
    """
    data_classes = DialogDataset.subclasses()
    data_cls = data_classes[args.data_name]

    return data_cls


class DailyDialog(DialogDataset):
    """
    The daily-dialog dataset from
    https://arxiv.org/pdf/1710.03957.pdf
    """

    url = 'http://parl.ai/downloads/dailydialog/'

    name = 'dailydialog'

    archive = 'dailydialog.tar.gz'

    files = ['train.json', 'valid.json', 'test.json']

    @classmethod
    def read_file(cls, data_path):
        """
        Reads the contents of a raw dailydialog file.
        """
        with open(data_path, 'r') as fh:
            for line in fh:
                dialog = json.loads(
                    line.strip())['dialogue']

                yield [ex['text'] for ex in dialog]


class PersonaChat(DialogDataset):
    """
    """

    name = 'persona'

    @classmethod
    def transform(cls, args, tokenizer):
        pass


class CornellMovies(DialogDataset):
    """
    """

    name = 'cornell'

    @classmethod
    def transform(cls, args, tokenizer):
        pass


class OpenSubtitles(DialogDataset):
    """
    """

    name = 'subtitles'

    @classmethod
    def transform(cls, args, tokenizer):
        pass
