"""
@author:    Patrik Purgai
@copyright: Copyright 2019, gpt2-chatbot
@license:   MIT
@email:     purgai.patrik@gmail.com
@date:      2019.09.30.
"""

import json
import os
import argparse
import requests

from tqdm import tqdm

from os.path import join, dirname, abspath, split, exists


URL = 'https://s3.amazonaws.com/datasets.huggingface.co/personachat/'

FILE_NAME = 'personachat_self_original.json'


def download(dump_path):
    """
    Downloads the personachat data file.
    """
    request = requests.get(URL + FILE_NAME, stream=True)

    with open(dump_path, 'wb') as fh:
        file_size = int(request.headers['content-length'])
        chunk_size = 1000

        with tqdm(
                desc='Downloading persona chat',
                total=file_size,
                unit_scale=True,
                unit='B',
                leave=False) as pbar:

            for chunk in request.iter_content(
                    chunk_size=chunk_size):
                fh.write(chunk)
                pbar.update(chunk_size)


def create_splits(dataset):
    """
    Creates splits from the downloaded dataset.
    """
    def generate_dialogs(split):
        # generating only the utterances from
        # persona chat data file
        for dialog in split:
            example = dialog['utterances'][-1]
            history = example['history']
            response = example['candidates'][-1]

            if history[0] == '__ SILENCE __':
                history = history[1:]

            # appending the response to the rest
            yield history + [response]

    return [
        generate_dialogs(dataset[split])
        for split in ['train', 'valid']
    ]


def save_jsonl(dump_path, dataset):
    """
    Saves the dataset to a file where every line of
    the file is a single json object containin a single
    training example.
    """
    with open(dump_path, 'w') as fh:
        for example in tqdm(
                dataset,
                desc='writing {}'.format(
                    split(dump_path)[-1]),
                leave=False):
            fh.write(json.dumps(
                {'dialog': example}) + '\n')


def create_config(args):
    """
    Creates a config file that lists the data files
    for the train and test set.
    """
    # config.json lists the path of the data files
    # relative to the directory that contains config.json
    config_path = join(args.dump_dir, 'config.json')
    with open(config_path, 'w') as fh:
        json.dump({
            'train': 'train.jsonl',
            'valid': 'valid.jsonl'
        }, fh)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dump_dir',
        default=join(
            abspath(dirname(__file__)), 'personachat'))
    parser.add_argument(
        '--force_download',
        action='store_true')

    args = parser.parse_args()
    os.makedirs(args.dump_dir, exist_ok=True)

    dump_path = join(args.dump_dir, FILE_NAME)

    if not exists(dump_path) or args.force_download:
        download(dump_path=dump_path)

    with open(dump_path, 'r') as fh:
        dataset = json.load(fh)

    train, valid = create_splits(dataset)

    save_jsonl(join(args.dump_dir, 'train.jsonl'), train)
    save_jsonl(join(args.dump_dir, 'valid.jsonl'), valid)

    create_config(args)


if __name__ == '__main__':
    main()

