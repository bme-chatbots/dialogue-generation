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
import datasets

URL = "https://s3.amazonaws.com/datasets.huggingface.co/personachat/"
FILE_NAME = "personachat_self_original.json"
SPLITS = ("train", datasets.Split.TRAIN), ("valid", datasets.Split.VALIDATION)


def download(file_name, url):
    with requests.Session() as session:
        response = session.get(url, stream=True, timeout=5)
        loop = tqdm.tqdm(desc="Downloading", unit="B", unit_scale=True)

        with open(file_name, "wb") as fh:
            for chunk in response.iter_content(1024):
                if chunk:
                    loop.update(len(chunk))
                    fh.write(chunk)


def write_txt(dialogues, output_file, field):
    with open(output_file, "w") as fh:
        for dialogue in dialogues:
            last = dialogue["utterances"][-1]
            utterances = last["history"] + [last["candidates"][-1]]
            utterances = [m for m in utterances if m != "__SILENCE__"]
            print(json.dumps({field: utterances}), file=fh)


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
            file_name = os.path.join(td, FILE_NAME)
            download(file_name, URL + FILE_NAME)

            with open(file_name) as fh:
                persona_chat = json.load(fh)

            for split_from, split_to in SPLITS:
                output_dir = os.path.join(args.output_dir, str(split_to))
                os.makedirs(output_dir, exist_ok=True)

                write_txt(
                    persona_chat[split_from],
                    os.path.join(output_dir, "data.jsonl"),
                    args.field
                )


if __name__ == "__main__":
    main()
