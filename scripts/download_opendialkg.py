"""
@author:    Patrik Purgai
@copyright: Copyright 2020, dialogue-kg
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
import tqdm
import csv

URL = "https://raw.githubusercontent.com/facebookresearch/opendialkg/master/data/"

DATA = "opendialkg.csv"
ENTITIES = "opendialkg_entities.txt"
RELATIONS = "opendialkg_relations.txt"
TRIPLES = "opendialkg_triples.txt"

FILES = [DATA, ENTITIES, RELATIONS, TRIPLES]


def download(file_name, url):
    with requests.Session() as session:
        response = session.get(url, stream=True, timeout=5)
        loop = tqdm.tqdm(desc="Downloading", unit="B", unit_scale=True)

        with open(file_name, "wb") as fh:
            for chunk in response.iter_content(1024):
                if chunk:
                    loop.update(len(chunk))
                    fh.write(chunk)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        required=True,
        type=str,
        help="Directory name of the output .txt files.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if not all(
        os.path.exists(os.path.join(args.output_dir, file_name)) for file_name in FILES
    ):
        for file_name in FILES:
            download(os.path.join(args.output_dir, file_name), f"{URL}{file_name}")


if __name__ == "__main__":
    main()
