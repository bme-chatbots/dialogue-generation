"""
@author:    Patrik Purgai
@copyright: Copyright 2020, dialoue-generation
@license:   MIT
@email:     purgai.patrik@gmail.com
@date:      2020.05.30.
"""

import random
import os
import omegaconf
import torch
import glob
import transformers
import itertools
import datasets
import functools
import importlib

import pytorch_lightning as pl
import numpy as np


EXPERIMENT_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_DIR = os.path.join(EXPERIMENT_DIR, "..", "..")

# this implementation uses the token_type_id field of the gpt2 transformer
# for signaling the source of the current utterance
BOT = "<|bot|>"
USR = "<|user|>"

# other than the native eos_id the dialogue model uses sos_id and pad_id
# as special control
EOS = "<|endoftext|>"
SOS = "<|startoftext|>"
PAD = "<|padding|>"

SPECIAL_TOKENS = [EOS, PAD, SOS, BOT, USR]

SPLITS = TRAIN, VALID = datasets.Split.TRAIN, datasets.Split.VALIDATION

INPUT_TEXT, LABEL_TEXT = "input_text", "label_text"
INPUT_IDS, SPEAKER_IDS, LABELS = "input_ids", "speaker_ids", "labels"
ATTENTION_MASK = "attention_mask"

COLUMNS = [INPUT_IDS, LABELS, INPUT_TYPES, ATTENTION_MASK]


# custom modelcheckpoint is required to overwrite the formatting in the file name
# as default "=" symbol conflicts with hydra's argument parsing
class ModelCheckpoint(pl.callbacks.ModelCheckpoint):
    def format_checkpoint_name(self, *args, **kwargs):
        return super().format_checkpoint_name(*args, **kwargs).replace("=", ":")


class Seq2SeqModule(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams

        self.model = transformers.EncoderDecoderModel.from_encoder_decoder_pretrained(
            self.hparams.encoder_pretrained_name, self.hparams.decoder_pretrained_name
        )

        self.model.get_encoder.resize_token_embeddings(self.hparams.vocab_size)
        self.model.get_decoder.resize_token_embeddings(

    def forward(self, batch):
        output = self.model(
            input_ids=batch[INPUT_IDS],
            attention_mask=batch[ATTENTION_MASK],
            return_dict=True,
        )

        logits = output["logits"].view(-1, output["logits"].size(-1))
        label_ids = batch[LABEL_IDS].view(-1)

        loss = torch.nn.functional.cross_entropy(
            logits,
            label_ids,
            ignore_index=self.hparams.pad_id,
        )

        attention_mask = batch[ATTENTION_MASK].view(-1)
        accuracy = (
            (label_ids[attention_mask] == logits[attention_mask].argmax(-1))
            .float()
            .mean()
        )

        ppl = torch.exp(loss)

        return {"loss": loss, "accuracy": accuracy, "ppl": ppl}

    def training_step(self, batch, batch_idx):
        output = self(batch)

        result = pl.TrainResult(output["loss"])
        result.log("loss", output.pop("loss"))

        for name, value in output.items():
            result.log(name, value, prog_bar=True)

        return result

    def validation_step(self, batch, batch_idx):
        output = self(batch)

        result = pl.EvalResult(
            checkpoint_on=output["loss"], early_stop_on=output["loss"]
        )
        result.log("loss", output.pop("loss"))

        for name, value in output.items():
            result.log(name, value, prog_bar=True)

        return result

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)

        return optimizer


class Seq2SeqDataModule(pl.LightningDataModule):
    def __init__(self, config, tokenizer):
        super().__init__()

        self.config = config
        self.tokenizer = tokenizer
        self.dataset = None

        special_tokens = [PAD, BOT, USR]
        self.specials = self.tokenizer.convert_tokens_to_ids(special_tokens)

    def prepare_data(self):
        project_scripts_dir = os.path.join(PROJECT_DIR, "scripts")
        experiments_script_dir = os.path.join(EXPERIMENT_DIR, "scripts")

        if self.config.name == "opendialkg":
            download_script = os.path.join(
                project_scripts_dir, f"download_opendialkg.py"
            )

            os.system(f"python {download_script} --output_dir {self.config.build_dir}")

            prepare_script = os.path.join(
                experiments_script_dir, f"prepare_opendialkg.py"
            )

            params = [
                f"--input_dir {self.config.build_dir}",
                f"--output_dir {os.path.join(self.config.build_dir, 'seq2seq')}",
                f"--field {self.config.field}",
            ]

            os.system(f"python {prepare_script} {' '.join(params)}")

        # instantiating dataset for building the cache file on a single worker
        build_dataset(self.tokenizer, self.config)

    def setup(self, stage=None):
        self.dataset = build_dataset(self.tokenizer, self.config)

    def train_dataloader(self):
        return build_dataloader(self.dataset[TRAIN], self.config, self.specials)

    def val_dataloader(self):
        return build_dataloader(self.dataset[VALID], self.config, self.specials, False)


class BucketSampler(torch.utils.data.Sampler):
    def __init__(self, lengths, max_tokens, num_buckets, shuffle=True):
        super().__init__(lengths)

        self.shuffle = shuffle
        self.partitions = partition(lengths, max_tokens, num_buckets + 1)
        self.length = None

    def __iter__(self):
        def group_elements(iterable, group_size):
            return itertools.zip_longest(*([iter(iterable)] * group_size))

        def generate_batches():
            partition_indices = list(range(len(self.partitions)))
            if self.shuffle:
                random.shuffle(partition_indices)

            for idx in partition_indices:
                batch_size, partition = self.partitions[idx]
                example_indices = partition.copy()

                if self.shuffle:
                    random.shuffle(example_indices)

                for group in group_elements(example_indices, batch_size):
                    yield list(filter(lambda idx: idx is not None, group))

        batches = list(generate_batches())
        if self.shuffle:
            random.shuffle(batches)

        return iter(batches)

    def __len__(self) -> int:
        if self.length is None:
            self.length = len(list(iter(self)))

        return self.length


def build_dataloader(dataset, config, specials, shuffle=True):
    bucket_sampler = BucketSampler(
        dataset["lengths"]["length"],
        config.max_input_tokens,
        config.num_buckets,
        shuffle,
    )

    return torch.utils.data.DataLoader(
        dataset["examples"],
        pin_memory=True,
        num_workers=config.num_workers,
        collate_fn=functools.partial(collate, specials=specials),
        batch_sampler=bucket_sampler,
    )


def build_examples(dialogues, history_size, input_field):
    examples = itertools.chain(
        *(
            list(generate_examples(dialogue, history_size))
            for dialogue in dialogues[input_field]
        )
    )

    inputs, input_types, labels = zip(*examples)

    return {
        INPUT_TEXT: list(inputs),
        INPUT_TYPES: list(input_types),
        LABEL_TEXT: list(labels),
    }


def generate_examples(dialogue, history_size):
    # each dialogue is separated into slices with src size of history_size + 1 and
    # a target which is the upcoming utterance in the dialogue
    def generate_slices():
        for to_idx in range(2, len(dialogue) + 1):
            for from_idx in range(max(to_idx - history_size - 2, 0), to_idx - 1):
                yield dialogue[from_idx:to_idx]

    for example in generate_slices():
        inputs, labels = example[:-1], SOS + example[-1] + EOS
        # 0 value always marks the last utterance of the partner in the dialogue
        input_types = [
            [idx % 2] * len(utterance) for idx, utterance in enumerate(reversed(inputs))
        ][::-1]

        yield inputs, input_types, labels


def encode_examples(example, tokenizer, max_input_tokens, max_label_tokens):
    model_inputs = tokenizer(
        example[INPUT_TEXT],
        return_tensors="np",
        max_length=max_input_tokens,
        truncation=True,
    )

    labels_and_decoder_mask = tokenizer(
        example[LABEL_TEXT],
        return_tensors="np",
        truncation=True,
    )

    labels = labels_and_decoder_mask["input_ids"]

    model_inputs["labels"] = labels

    return model_inputs


def build_split(
    examples,
    tokenizer,
    max_input_tokens,
    max_label_tokens,
    cache_dir,
    split_name,
    history_size,
    field,
):
    examples = examples.map(
        functools.partial(build_examples, history_size=history_size, input_field=field),
        batched=True,
        remove_columns=examples.column_names,
        cache_file_name=os.path.join(cache_dir, f"{split_name}.raw.cache"),
    )

    examples = examples.map(
        lambda example: encode_examples(
            example,
            tokenizer,
            max_input_tokens,
            max_label_tokens,
        ),
        remove_columns=[INPUT_TEXT, LABEL_TEXT],
        cache_file_name=os.path.join(cache_dir, f"{split_name}.examples.cache"),
    )

    examples.set_format(type="np", columns=COLUMNS)

    lengths = examples.map(
        lambda example: {"length": example[INPUT_IDS][0].size},
        remove_columns=examples.column_names,
        cache_file_name=os.path.join(cache_dir, f"{split_name}.lengths.cache"),
    )

    return {"examples": examples, "lengths": lengths}


def build_dataset(tokenizer, config):
    data_files = {
        str(split): glob.glob(config.get(str(split)).data_pattern) for split in SPLITS
    }

    dataset = datasets.load_dataset("json", data_files=data_files)

    # splits contains a dictionary of dictionary of datasets with 2 keys `examples`
    # contains the dataset with the inputs and `lengths` contains the size of each
    # example which is used by the batch sampler
    splits = {
        split: build_split(
            examples=examples,
            tokenizer=tokenizer,
            max_input_tokens=config.max_input_tokens,
            max_label_tokens=config.max_label_tokens,
            cache_dir=config.cache_dir,
            split_name=split,
            history_size=config.history_size,
            field=config.field,
        )
        for split, examples in dataset.items()
    }

    return splits


def partition(lengths, max_tokens, num_buckets):
    points = np.arange(1 / num_buckets, 1.0, 1 / num_buckets)
    quantiles = np.quantile(np.array(lengths), points)

    indices = iter(sorted(range(len(lengths)), key=lambda idx: lengths[idx]))
    batch_sizes = [2 ** exp for exp in range(10, -1, -1)]

    partitions = []
    for q in quantiles.tolist() + [max(lengths)]:
        batch_size = itertools.dropwhile(lambda bs: max_tokens < bs * q, batch_sizes)
        partition = list(itertools.takewhile(lambda idx: lengths[idx] <= q, indices))
        partitions.append((next(batch_size), partition))

    return partitions


def collate(batch, specials):
    pad_id, bot_id, usr_id = specials

    batch_size = len(batch)
    max_len = max([e[INPUT_IDS][0].shape[0] for e in batch]) - 1

    input_ids = np.full((batch_size, max_len), pad_id, dtype=np.int64)
    label_ids = np.copy(input_ids)
    attention_mask = np.zeros_like(input_ids, dtype=np.int8)

    for idx, example in enumerate(batch):
        example_len = example[INPUT_IDS][0].shape[0] - 1
        input_ids[idx, :example_len] = example[INPUT_IDS][0][:-1]
        attention_mask[idx, :example_len] = example[ATTENTION_MASK][0][:-1]

        label_ids[idx, :example_len] = example[INPUT_IDS][0][1:]

    label_ids = torch.as_tensor(label_ids)
    label_ids[(label_ids == bot_id) | (label_ids == usr_id)] = pad_id

    return {
        INPUT_IDS: torch.as_tensor(input_ids),
        LABEL_IDS: label_ids,
        ATTENTION_MASK: torch.as_tensor(attention_mask, dtype=torch.bool),
    }

