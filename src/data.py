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
import hashlib
import shutil
import os
import random
import json
import tarfile

from tqdm import tqdm
from itertools import (
    zip_longest, chain)

from copy import deepcopy

from torch.utils.data import (
    Dataset, DataLoader,
    Sampler)

from torch.distributed import barrier

from torch.utils.data.distributed import (
    DistributedSampler)

from transformers import (
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
        '-d', '--data',
        type=str,
        default='dailydialog',
        choices=list(DialogDataset.subclasses()),
        help='Name of the dataset to use.')
    group.add_argument(
        '--data_dir',
        type=str,
        default=join(abspath(
            dirname(__file__)), '..', 'data'),
        help='Path of the data root directory.')
    group.add_argument(
        '--download_dir',
        type=str,
        default=join(abspath(
            dirname(__file__)), '..', 'data'),
        help='Path of the download directory.')
    group.add_argument(
        '--file_size',
        type=int,
        default=100000,
        help='Max number of examples in a single file.')
    group.add_argument(
        '--max_hist',
        type=int,
        default=2,
        help='Maximum number of turns in history.')
    group.add_argument(
        '--force_rebuild',
        action='store_true',
        help='Creates the dataset even if it exists.')
    group.add_argument(
        '--max_len',
        type=int,
        default=50,
        help='Maximum length of a sequence.')


def group_elements(iterable, group_size):
    """
    Collect data into fixed-length chunks.
    """
    groups = [iter(iterable)] * group_size

    return zip_longest(*groups)


def generate_num_elements(iterable, num_elements):
    """
    Convenience function for generating `num_elements`
    from an iterable.
    """
    for _, element in zip(range(num_elements), iterable):
        yield element


def hash_data(args):
    """
    Creates a unique identifier for the special tokens
    and tokenzier type.
    """
    string = '{}{}'.format(
        args.max_hist, args.file_size)
    encoded = string.encode()

    num = int(hashlib.sha1(encoded).hexdigest(), 16)

    return str(num)[-8:]


def save_examples(
        args, content, name, tokenizer, data_dir):
    """
    Creates numericalizes examples from a raw data
    file and serializes them.
    """
 
    # during data preprocessing the history
    # and target utterances are saved only once
    dialogs = generate_dialogs(
        args=args,
        content=content,
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
            *list(generate_examples(dialogs=group)))

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


def generate_dialogs(args, content, tokenizer):
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
                # `eos_id` is not added yet and therefore
                # it is not included in target length
                target_len = len(target_utr) + 1
                example_len += target_len

                yield (dialog_idx, begin_idx,
                       end_idx, example_len)

    for idx, dialog in tqdm(
            enumerate(content), 
            desc='transforming dataset',
            leave=False):
        dialog = [tokenizer.encode(u) for u in dialog]

        # generating indices list that indexes into
        # the dialog and slices a history and a target
        # this way the text data only has to be stored once
        indices = list(generate_indices(
            dialog_idx=idx, encoded_dialog=dialog))

        yield dialog, indices


def transform_dialog(history, special_ids, max_len):
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
        # truncating each sequence to max len
        ids = ids[:max_len]
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


def generate_files(filenames):
    """
    Geberates files from the filenames.
    """
    file_dataset = FileDataset(filenames)
    file_loader = DataLoader(
        file_dataset,
        collate_fn=lambda x: x[0])

    yield from file_loader

        
def create_loader(
        args, filenames, tokenizer, dataset_cls, 
        shuffle=False):
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

    collate_fn = COLLATE[args.model]

    def load_examples():
        """
        Generator that loads examples from files
        lazily.
        """
        files = generate_files(filenames)

        for dialogs, indices in files:
            sampler = bucket_sampler_cls(
                indices, shuffle=shuffle)

            dialog_dataset = dataset_cls(
                dialogs=dialogs,
                indices=indices,
                max_len=args.max_len,
                special_ids=special_ids)

            example_loader = DataLoader(
                dialog_dataset,
                batch_size=args.batch_size,
                num_workers=1,
                sampler=sampler,
                pin_memory=True,
                collate_fn=collate_fn)

            yield from example_loader

    return load_examples


TOKENIZER = {
    'xlnet-base-cased':     XLNetTokenizer,
    'xlnet-large-cased':    XLNetTokenizer,
    'distilgpt2':           GPT2Tokenizer,
    'gpt2':                 GPT2Tokenizer,
    'gpt2-medium':          GPT2Tokenizer,
    'gpt2-large':           GPT2Tokenizer,
    'gpt2-xl':              GPT2Tokenizer
}


def create_tokenizer(args):
    """
    Creates the tokenizer for the model and saves
    it in the model data directory if it does not exist.
    """
    data_dir = join(
        args.data_dir, args.data, args.model)

    tokenizer_path = join(
        data_dir, 'special_tokens_map.json')

    assert args.model in TOKENIZER, \
        'Available tokenizers: {} received `{}`'.format(
            ', '.join(TOKENIZER), args.model)

    tokenizer_cls = TOKENIZER[args.model]

    if not exists(tokenizer_path):
        # TODO come up with better naming
        # for tokenizer `instance`
        tokenizer = tokenizer_cls.from_pretrained(
            args.model)

        # adding special tokens
        # TODO check compatibility with all tokenizers
        special_tokens = [SP1, SP2, HST, RSP]
        tokenizer.add_special_tokens({
            'additional_special_tokens': special_tokens})

        if tokenizer.pad_token is None:
            # GPT2 does not have a pad token
            tokenizer.add_special_tokens({
                'pad_token': PAD})

        tokenizer.save_pretrained(data_dir)

    else:
        tokenizer = tokenizer_cls.from_pretrained(
            data_dir)

    return tokenizer


def create_dataset(args, master_process):
    """
    Downloads the dataset, converts it to tokens and 
    returns iterators over the train and test splits.
    """
    data_hash = hash_data(args)

    data_dir = join(
        args.data_dir, args.data, args.model,
        data_hash)

    os.makedirs(data_dir, exist_ok=True)

    metadata_path = join(data_dir, 'metadata.json')

    tokenizer = create_tokenizer(args)
    dataset_cls = create_data_cls(args)

    # only the master process will create the dataset
    if (not exists(metadata_path) \
            or args.force_rebuild) and master_process:
        # if dataset does not exist then create it
        # downloading and tokenizing the raw files
        files = dataset_cls.download(args=args)

        train, valid, test = dataset_cls.transform(
            args=args, files=files, 
            tokenizer=tokenizer, data_dir=data_dir)

        train_files, train_size = train
        valid_files, valid_size = valid
        test_files, test_size = test

        # save the location of the files in a metadata
        # json object and delete the file in case of
        # interrupt so it wont be left in corrupted state
        with open(metadata_path, 'w') as fh:
            try:
                json.dump({
                    'train': [train_files, train_size],
                    'valid': [valid_files, valid_size],
                    'test': [test_files, test_size],
                    'max_hist': args.max_hist,
                }, fh)
            except KeyboardInterrupt:
                shutil.rmtree(metadata_path)

    if args.distributed:
        # synchronizing processes before creating
        # data loaders
        barrier()

    with open(metadata_path, 'r') as fh:
        filenames = json.load(fh)

    train_files, train_size = filenames['train']
    valid_files, valid_size = filenames['valid']
    test_files, test_size = filenames['test']

    train_dataset = create_loader(
        args=args,
        filenames=train_files,
        tokenizer=tokenizer,
        dataset_cls=dataset_cls,
        shuffle=True)

    max_len = args.max_len

    valid_dataset = create_loader(
        args=args,
        filenames=valid_files,
        tokenizer=tokenizer,
        dataset_cls=dataset_cls)

    test_dataset = create_loader(
        args=args,
        filenames=test_files,
        tokenizer=tokenizer,
        dataset_cls=dataset_cls)

    train = train_dataset, train_size
    valid = valid_dataset, valid_size
    test = test_dataset, test_size

    return (train, valid, test), tokenizer, max_len


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
    data_cls = data_classes[args.data]

    return data_cls


def create_dummy_batch(
        args, ignore_idx):
    """
    Creates a dummy batch for OOM sync.
    """
    collate_fn = COLLATE[args.model]

    # each input starts with a role id
    # and the whole sequence starts with
    # a special sos token
    input_len = (args.max_len + 1) * \
        args.max_hist + 1

    # adding 1 for the final eos_token
    target_len = args.max_len + 1

    targets = [ignore_idx] * target_len
    input_ids = [0] * input_len + targets
    token_type_ids = [0] * len(input_ids)

    dummy_example = [
        input_ids, token_type_ids, 
        targets, [input_len]]

    return collate_fn(
        [dummy_example] * args.batch_size)


def download(download_path, url):
    """
    Downloads a file.
    """
    with requests.Session() as session:
        response = session.get(
            url, stream=True, timeout=5)

        # data is read in 2 ** 15 sized chunks
        # NOTE this could be tuned to reveal
        # data size in MBs
        loop = response.iter_content(2 ** 20)
        loop = tqdm(
            loop, leave=False, unit='MB', 
            unit_scale=True)

        with open(download_path, 'wb') as f:
            for chunk in loop:
                if chunk:
                    f.write(chunk)


class DialogDataset(Dataset):
    """
    Fetches utterances from a list of examples.
    The examples are produced from subsets of dialogs.
    """

    # base url to download the data
    url = ''

    # name of the dataset
    name = ''

    # TODO renames these attributes
    # name of the downloaded archive
    archives = []

    # list of the extracted filenames
    files = []

    @classmethod
    def download(cls, args):
        """
        Downloads and extracts the dataset.
        """
        extract_dir = join(
            args.data_dir, args.data)
        download_dir = join(
            args.download_dir, args.data)

        os.makedirs(extract_dir, exist_ok=True)
        os.makedirs(download_dir, exist_ok=True)

        extracted_files = []

        for archive in cls.archives:
            url = cls.url + archive

            download_path = join(
                download_dir, archive)

            if not exists(download_path):
                msg = 'Downloading dataset to {}'
                print(msg.format(download_path))

                download(download_path, url)

            extracted_files.extend(
                cls.extract(
                    download_path=download_path,
                    extract_dir=extract_dir))

        return extracted_files

    @classmethod
    def read_file(cls, data_path):
        raise NotImplementedError('Abstract method.')

    @classmethod
    def extract(cls, download_path, extract_dir):
        raise NotImplementedError('Abstract method.')

    @classmethod
    def generate_splits(cls, extracted_files):
        raise NotImplementedError('Abstract method.')

    @classmethod
    def transform(cls, args, files, tokenizer, data_dir):
        """
        Transforms the dataset into numericalized
        format and saves it in fragments.
        """
        for content, name in cls.generate_splits(files):
            yield save_examples(
                args=args,
                content=content,
                name=name,
                tokenizer=tokenizer,
                data_dir=data_dir)

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

    def __init__(
            self, dialogs, indices, 
            special_ids, max_len):
        self.dialogs = dialogs
        self.indices = indices
        self.max_len = max_len
        self.special_ids = special_ids

    def __getitem__(self, idx):
        dialog_idx, begin_idx, end_idx, input_len = \
            self.indices[idx]
        dialog = self.dialogs[dialog_idx]

        # the whole dialog is fetched and the
        # `idx` element of indices array creates
        # the example
        eos_id = self.special_ids[3]
        rsp_id = self.special_ids[5]

        history = dialog[begin_idx:end_idx]
        
        # truncating the target to max_len
        # each input sequence will be tuncated
        # in `transform_dialog` function
        target = dialog[end_idx][:self.max_len]
        target += [eos_id]

        inputs = transform_dialog(
            history=history,
            special_ids=self.special_ids,
            max_len=self.max_len)

        input_ids, token_type_ids = inputs

        input_ids.extend(target)
        token_type_ids.extend([rsp_id] * len(target))

        # returning nested lists for convenient
        # parameter passing to collate_fn
        return [
            input_ids, token_type_ids, 
            target, [input_len]
        ]

    def __len__(self):
        return len(self.indices)


class DailyDialog(DialogDataset):
    """
    The daily-dialog dataset from
    https://arxiv.org/pdf/1710.03957.pdf
    """

    url = 'http://parl.ai/downloads/dailydialog/'

    name = 'dailydialog'

    archives = ['dailydialog.tar.gz']

    files = ['train.json', 'valid.json', 'test.json']

    @classmethod
    def extract(cls, download_path, extract_dir):
        extracted_files = [
            join(extract_dir, f) for f in cls.files]

        if any(not exists(p) for p in extracted_files):
            shutil.unpack_archive(
                download_path, extract_dir)

        return extracted_files

    @classmethod
    def generate_splits(cls, extracted_files):
        """
        Creates splits from the extracted_files.
        """
        # daily dialog data is already split into
        # train valid and test split
        for f in extracted_files:
            yield (
                cls.read_file(f),
                basename(splitext(f)[0])
            )

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
    Persona chat dataset from
    https://arxiv.org/pdf/1801.07243.pdf
    """

    url = 'https://s3.amazonaws.com/datasets.huggingface.co/personachat/'

    name = 'personachat'

    archives = ['personachat_self_original.json']

    # no compression is used in this dataset
    files = None

    @classmethod
    def extract(cls, download_path, extract_dir):
        return [join(extract_dir, cls.archives[0])]

    @classmethod
    def read_file(cls, data_path):
        """
        Reads the contents of a raw file.
        """
        with open(data_path, 'r') as fh:
            return json.load(fh)

    @classmethod
    def generate_splits(cls, extracted_files):
        """
        Creates splits from the extracted_files.
        """
        content = cls.read_file(extracted_files[0])

        train = content['train']
        valid = content['valid']
        test = content['valid']

        def generate_uttrs(split):
            # generating only the utterances from
            # persona chat datafile
            for dialog in split:
                whole_dialog = dialog['utterances'][-1]
                history = whole_dialog['history']
                response = whole_dialog['candidates'][-1]

                # appending the response to the whole
                yield history + [response]

        # TODO there is no test data for personachat
        # create separate test data from validation
        return [
            (generate_uttrs(train), 'train'),
            (generate_uttrs(valid), 'valid'),
            (generate_uttrs(test), 'test')
        ]


class TopicalChat(DialogDataset):
    """
    Topical-Chat dataset from
    https://github.com/alexa/alexa-prize-topical-chat-dataset
    """

    name = 'topicalchat'

    url = 'https://raw.githubusercontent.com/alexa/alexa-prize-topical-chat-dataset/master/conversations/'

    archives = [
        'train.json',
        'valid_rare.json', 'valid_freq.json',
        'test_rare.json', 'test_freq.json'
    ]

    files = archives

    @classmethod
    def extract(cls, download_path, extract_dir):
        return [join(
            extract_dir,
            download_path.split('/')[-1])]

    @classmethod
    def read_file(cls, data_path):
        """
        Reads the contents of a raw file.
        """
        with open(data_path, 'r') as fh:
            return json.load(fh)

    @classmethod
    def generate_splits(cls, extracted_files):
        """
        Creates splits from the extracted_files.
        """
        files = [cls.read_file(f) for f in extracted_files]

        train, valid_rare, valid_freq, \
            test_rare, test_freq = files

        # NOTE currently we are not dealing with
        # frequent or rare elements so merging them
        valid = {**valid_rare, **valid_freq}
        test = {**test_rare, **test_freq}

        def generate_uttrs(split):
            """
            Generates data from dialog jsons.
            """
            for dialog in split:
                content = split[dialog]['content']

                # appending the response to the whole
                yield [turn['message'] for turn in content]

        return [
            (generate_uttrs(train), 'train'),
            (generate_uttrs(valid), 'valid'),
            (generate_uttrs(test), 'test')
        ]


class CornellMovies(DialogDataset):
    """
    Cornell movies dataset from
    https://arxiv.org/pdf/1106.3077.pdf
    """

    name = 'cornellmovies'

    @classmethod
    def transform(cls, args, tokenizer):
        pass


class OpenSubtitles(DialogDataset):
    """
    """

    url = 'http://opus.lingfil.uu.se/download.php?f=OpenSubtitles/'

    archives = ['en.tar.gz']

    files = ['train.json', 'valid.json', 'test.json']

    name = 'opensubtitles'


class CustomDataset(DialogDataset):
    """
    Example for defining a custom dataset.
    """

    name = 'custom_dataset'

    @classmethod
    def download(cls, args):
        # this method would normally download the
        # dataset but it is assumed that custom data
        # is already present
        return [
            (join(args.data_dir, split) + '.txt', split)
            for split in ['train', 'valid', 'test']
        ]

    @classmethod
    def read_file(cls, data_path):
        """
        Reads the contents of a raw file.
        """
        with open(data_path, 'r') as fh:
            for line in fh:
                yield fh.strip()

    @classmethod
    def generate_splits(cls, extracted_files):
        """
        Creates splits from the extracted_files.
        """
        def generate_uttrs(split):
            """
            Generates data from text data.
            """
            dialog = []
            for utterance in split:
                if utterance == '':
                    yield dialog
                    dialog = []
                else:
                    dialog.append(utterance)

        return [
            (generate_uttrs(file_path), name)
            for file_path, name in extracted_files
        ]
