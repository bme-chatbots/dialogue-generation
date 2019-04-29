"""

@author:    Patrik Purgai
@copyright: Copyright 2019, nmt
@license:   MIT
@email:     purgai.patrik@gmail.com
@date:      2019.04.04.
"""

import torch
import sys

from os.path import exists, join, dirname, abspath
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
from nltk.translate import bleu


START_TOKEN = '<sos>'
END_TOKEN = '<eos>'
PAD_TOKEN = '<pad>'


def setup_data_args(parser):
    """
    Sets up the data arguments.
    """
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Size of the batches during training.')
    parser.add_argument(
        '--data_dir',
        type=str,
        default=join(abspath(dirname(__file__)), '..', 'data'),
        help='Path of the data root directory.')
    parser.add_argument(
        '--vocab_size',
        type=int,
        default=3000,
        help='Maximum size of the vocabulary.')
    

def text2ids(text, field):
    """
    Converts a text to list of ids.
    """
    return field.numericalize(text)


def ids2text(ids, field):
    """
    Converts a list of ids to text.
    """
    return ' '.join([field.itos[i] for i in ids])


def create_datasets(args, device):
    """
    Creates the datasets.
    """
    fields_path = join(args.model_dir, 'fields.pt')
    if not exists(fields_path):
        src = Field(
            batch_first=True,
            lower=True)

        trg = Field(
            init_token=START_TOKEN, 
            eos_token=END_TOKEN,
            batch_first=True,
            lower=True,
            is_target=True)
    else:
        print('Loading fields from {}'.format(fields_path))
        fields = torch.load(fields_path)
        src = fields['src']
        trg = fields['trg']

    train, valid, test = Multi30k.splits(
        exts=('.en', '.de'),
        fields=(src, trg),
        root=args.data_dir)

    if not exists(fields_path):
        src.build_vocab(
            train, valid, 
            max_size=args.vocab_size)

        trg.build_vocab(
            train, valid, 
            max_size=args.vocab_size)

        print('Saving fields to {}'.format(fields_path))
        torch.save({'src': src, 'trg': trg}, fields_path)

    vocabs = src.vocab, trg.vocab

    iterators = BucketIterator.splits(
        (train, valid, test), 
        batch_sizes=[args.batch_size] * 3,
        sort_key=lambda x: len(x.src),
        device=device)

    return iterators, vocabs


def get_special_indices(vocabs):
    """
    Returns the special token indices from the vocab.
    """
    src_vocab, trg_vocab = vocabs

    start_index = trg_vocab.stoi[START_TOKEN]
    end_index = trg_vocab.stoi[END_TOKEN]

    src_pad_index = src_vocab.stoi[PAD_TOKEN]
    trg_pad_index = trg_vocab.stoi[PAD_TOKEN]

    return start_index, end_index, src_pad_index, \
        trg_pad_index
