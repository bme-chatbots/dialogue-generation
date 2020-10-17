"""
@author:    Patrik Purgai
@copyright: Copyright 2020, dialogue-kg
@license:   MIT
@email:     purgai.patrik@gmail.com
@date:      2020.05.30.
"""

import os
import argparse
import csv
import json
import random
import datasets

FILE_NAME = "opendialkg.csv"
SPLITS = datasets.Split.TRAIN, datasets.Split.VALIDATION


def write_jsonl(dialogues, output_file, field):
    with open(output_file, "w") as fh:
        for dialogue in dialogues:
            print(json.dumps({field: dialogue}), file=fh)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        required=True,
        type=str,
        help="Directory name of the input files.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        type=str,
        help="Directory name of the output files.",
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

    args = parser.parse_args()

    if not all(
        os.path.exists(os.path.join(args.output_dir, str(split), f"{split}.jsonl"))
        for split in SPLITS
    ):
        dialogues = []

        with open(os.path.join(args.input_dir, FILE_NAME), newline="") as fh:
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

        for split_name, split_indices in splits:
            split_dir = os.path.join(args.output_dir, str(split_name))
            os.makedirs(split_dir, exist_ok=True)

            write_jsonl(
                [dialogues[idx] for idx in split_indices],
                os.path.join(split_dir, f"{split_name}.jsonl"),
                args.field,
            )


if __name__ == "__main__":
    main()
