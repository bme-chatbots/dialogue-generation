"""
@author:    Patrik Purgai
@copyright: Copyright 2020, dialogue-generation
@license:   MIT
@email:     purgai.patrik@gmail.com
@date:      2020.05.30.
"""

import os
import shutil
import tempfile
import argparse
import requests
import json
import random
import tqdm
import datasets
import csv

URL = "https://raw.githubusercontent.com/facebookresearch/opendialkg/master/data/"

DATA = "opendialkg.csv"
ENTITIES = "opendialkg_entities.txt"
RELATIONS = "opendialkg_relations.txt"
TRIPLES = "opendialkg_triples.txt"

FILES = [DATA, ENTITIES, RELATIONS, TRIPLES]

SPLITS = datasets.Split.TRAIN, datasets.Split.VALIDATION


def download(file_name, url):
    with requests.Session() as session:
        response = session.get(url, stream=True, timeout=5)
        loop = tqdm.tqdm(desc="Downloading", unit="B", unit_scale=True)

        with open(file_name, "wb") as fh:
            for chunk in response.iter_content(1024):
                if chunk:
                    loop.update(len(chunk))
                    fh.write(chunk)


def write_jsonl(dialogues, output_file, field):
    with open(output_file, "w") as fh:
        for dialogue in dialogues:
            print(json.dumps({field: dialogue}), file=fh)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        required=True,
        type=str,
        help="Directory name of the output .txt files.",
    )
    parser.add_argument(
        "--field",
        default="dialogue",
        type=str,
        help="Name of the key in the output JSON object for the dialogue.",
    )
    parser.add_argument(
        "--valid_split",
        default=0.1,
        type=float,
        help="Size of the validation split as a number between 0-1.",
    )
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
        help="Seed value for the split generation for deterministic behaviour.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="If provided then dataset is recreated regardless if it exists.",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if not all(
        os.path.exists(os.path.join(args.output_dir, str(split), f"{split}.jsonl")) for split in SPLITS
    ):
        with tempfile.TemporaryDirectory(dir=args.output_dir) as td:
            file_name = os.path.join(td, DATA)
            download(file_name, URL + DATA)

            dialogues = []

            with open(file_name, newline="") as fh:
                reader = csv.DictReader(fh)
                key = "message"

                for row in reader:
                    dialogue = [m[key] for m in json.loads(row["Messages"]) if key in m]
                    if len(dialogue) > 1:
                        dialogues.append(dialogue)

            indices = list(range(len(dialogues)))
            random.shuffle(indices)

            train_split = int(len(indices) * (1 - args.valid_split))
            splits = zip(SPLITS, (indices[:train_split], indices[train_split:]))

            for split, split_indices in splits:
                split_dir = os.path.join(args.output_dir, str(split))
                os.makedirs(split_dir, exist_ok=True)

                write_jsonl(
                    [dialogues[idx] for idx in split_indices],
                    os.path.join(split_dir, f"{split}.jsonl"),
                    args.field,
                )


if __name__ == "__main__":
    main()
