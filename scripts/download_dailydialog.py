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
import tqdm

URL = "http://parl.ai/downloads/dailydialog/"
ARCHIVE = "dailydialog.tar.gz"
SPLITS = (
    ("train", datasets.Split.TRAIN),
    ("valid", datasets.Split.VALIDATION),
    ("test", datasets.Split.TEST),
)


def download(file_name, url):
    with requests.Session() as session:
        response = session.get(url, stream=True, timeout=5)
        loop = tqdm.tqdm(desc="Downloading", unit="B", unit_scale=True)

        with open(file_name, "wb") as fh:
            for chunk in response.iter_content(1024):
                if chunk:
                    loop.update(len(chunk))
                    fh.write(chunk)


def read_jsonl(file_name):
    with open(file_name) as fh:
        for line in fh:
            yield [turn["text"] for turn in json.loads(line)["dialogue"]]


def write_jsonl(input_file, output_file, field):
    with open(output_file, "w") as fh:
        for dialogue in read_jsonl(input_file):
            print(json.dumps({field: dialogue}), file=fh)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        required=True,
        type=str,
        help="Directory name of the output training files.",
    )
    parser.add_argument(
        "--field",
        default="dialogue",
        type=str,
        help="Name of the key in the output JSON object for the dialogue.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="If provided then dataset is recreated regardless if it exists.",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.force or not all(
        os.path.exists(os.path.join(args.output_dir, str(split))) for _, split in SPLITS
    ):
        with tempfile.TemporaryDirectory(dir=args.output_dir) as td:
            archive_path = os.path.join(td, ARCHIVE)
            download(archive_path, URL + ARCHIVE)
            shutil.unpack_archive(archive_path, td)

            for split_from, split_to in SPLITS:
                output_dir = os.path.join(args.output_dir, str(split_to))
                os.makedirs(output_dir, exist_ok=True)

                write_jsonl(
                    os.path.join(td, split_from + ".json"),
                    os.path.join(output_dir, "data.jsonl"),
                    args.field,
                )


if __name__ == "__main__":
    main()
