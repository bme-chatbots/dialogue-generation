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
import logging
import dataclasses
import collections

import pytorch_lightning as pl
import numpy as np

EXPERIMENT_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_DIR = os.path.join(EXPERIMENT_DIR, "..", "..")

# this implementation uses the token_type_id field of the gpt2 transformer
# for signaling the source of the current utterance
SPEAKER_FROM = "<|speaker_from|>"
SPEAKER_TO = "<|speaker_to|>"

SPLITS = TRAIN, VALID = datasets.Split.TRAIN, datasets.Split.VALIDATION

INPUT_IDS = "input_ids"
ATTENTION_MASK = "attention_mask"
DECODER_INPUT_IDS = "decoder_input_ids"
DECODER_ATTENTION_MASK = "decoder_attention_mask"
LABELS = "labels"

LABEL_MASK_ID = -100


# custom modelcheckpoint is required to overwrite the formatting in the file name
# as default "=" symbol conflicts with hydra's argument parsing
class ModelCheckpoint(pl.callbacks.ModelCheckpoint):
    def format_checkpoint_name(self, *args, **kwargs):
        return super().format_checkpoint_name(*args, **kwargs).replace("=", ":")


class BARTModule(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams

        self.model = transformers.BartForConditionalGeneration.from_pretrained(
            self.hparams.pretrained_name
        )

        self.model.resize_token_embeddings(self.hparams.vocab_size)

    def forward(self, batch):
        output = self.model(
            input_ids=batch[INPUT_IDS],
            attention_mask=batch[ATTENTION_MASK],
            decoder_input_ids=batch[DECODER_INPUT_IDS],
            decoder_attention_mask=batch[DECODER_ATTENTION_MASK],
            use_cache=False,
            return_dict=True,
        )

        logits = output["logits"].view(-1, output["logits"].size(-1))
        labels = batch[LABELS].view(-1)

        loss = torch.nn.functional.cross_entropy(logits, labels)

        mask = batch[DECODER_ATTENTION_MASK].view(-1)
        accuracy = (
            (labels[mask] == logits[mask].argmax(-1))
            .float()
            .mean()
        )

        ppl = torch.exp(loss)

        return {"loss": loss, "accuracy": accuracy, "ppl": ppl}

    def training_step(self, batch, batch_idx):
        output = self(batch)

        loss = output.pop("loss")
        self.log("loss", loss)
        self.log_dict(output, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        output = self(batch)

        loss = output.pop("loss")
        self.log("loss", loss)
        self.log_dict(output, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)

        return optimizer


class BARTDataModule(pl.LightningDataModule):
    def __init__(self, config, tokenizer):
        super().__init__()

        self.config = config
        self.tokenizer = tokenizer
        self.dataset = None

    def prepare_data(self):
        def run_script(name):
            scripts_dir = os.path.join(PROJECT_DIR, "scripts")
            script = os.path.join(scripts_dir, f"download_{name}.py")
            params = f"--output_dir {self.config.build_dir} --field {self.config.field}"

            if self.config.rebuild:
                params += " --force"

            os.system(f"python {script} {params}")

        # TODO
        # upon the release of omegaconf 2.1 this can be changed to contain a list
        # by defining custom resolvers for the config interpolation
        run_script(self.config.name)

        # instantiating dataset for building the cache file on a single worker
        build_dataset(self.tokenizer, self.config)

    def setup(self, stage=None):
        self.dataset = build_dataset(self.tokenizer, self.config)

    def train_dataloader(self):
        return build_dataloader(self.dataset[TRAIN], self.config)

    def val_dataloader(self):
        return build_dataloader(self.dataset[VALID], self.config, False)


class BucketSampler(torch.utils.data.Sampler):
    def __init__(self, lengths, max_length, num_buckets, shuffle=True):
        super().__init__(lengths)

        self.shuffle = shuffle
        self.partitions = partition(lengths, max_length, num_buckets + 1)
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


def build_dataloader(dataset, config, shuffle=True):
    bucket_sampler = BucketSampler(
        dataset["lengths"]["length"],
        config.max_input_length,
        config.num_buckets,
        shuffle,
    )

    return torch.utils.data.DataLoader(
        dataset["examples"],
        pin_memory=True,
        num_workers=config.num_workers,
        collate_fn=collate,
        batch_sampler=bucket_sampler,
    )


def build_dataset(tokenizer, config):
    data_files = {
        str(split): glob.glob(config.get(str(split)).data_pattern) for split in SPLITS
    }

    dataset = datasets.load_dataset("json", data_files=data_files)

    # splits contains a dictionary of dictionary of datasets with 2 keys `examples`
    # contains the dataset with the inputs and `lengths` contains the size of each
    # example which is used by the batch sampler
    splits = {
        split_name: build_split(examples, tokenizer, split_name, config)
        for split_name, examples in dataset.items()
    }

    return splits


def build_split(examples, tokenizer, split_name, config):
    examples = examples.map(
        lambda example: build_examples(example, tokenizer, config),
        batched=True,
        remove_columns=examples.column_names,
        cache_file_name=os.path.join(config.cache_dir, f"{split_name}.examples.cache"),
    )

    lengths = examples.map(
        lambda example: {"length": len(example[INPUT_IDS])},
        remove_columns=examples.column_names,
        cache_file_name=os.path.join(config.cache_dir, f"{split_name}.lengths.cache"),
    )

    examples.set_format(type="np", columns=examples.column_names)

    return {"examples": examples, "lengths": lengths}


def build_examples(dialogues, tokenizer, config):
    def generate_examples():
        for dialogue in dialogues[config.field]:
            assert len(dialogue) > 1

            for idx in range(2, len(dialogue)):
                yield build_example(dialogue[:idx], dialogue[idx], tokenizer, config)

    examples = collections.defaultdict(list)
    for example in generate_examples():
        for key, value in example.items():
            examples[key].append(value)

    return examples


def build_example(input_texts, label_text, tokenizer, config):
    example = tokenizer(flatten_dialogue(input_texts), add_special_tokens=False)

    example = {
        key: value[-config.max_input_length :]
        for key, value in example.items()
    }

    decoder_dict = tokenizer(
        label_text, max_length=config.max_label_length, truncation=True
    )

    for key, value in decoder_dict.items():
        example[f"decoder_{key}"] = value

    return example


def flatten_dialogue(dialogue):
    # each dialogue is flattened into a single string where the utterances are
    # preceided by a special role token and finished with an EOS token
    reversed_dialogue = [
        get_speaker_token(idx) + utterance
        for idx, utterance in enumerate(reversed(dialogue))
    ]

    return "".join(reversed_dialogue[::-1])


def get_speaker_token(idx):
    # if the idx is even then SPEAKER_TO id is assigned to the idx
    return SPEAKER_TO if idx % 2 else SPEAKER_FROM


def partition(lengths, max_length, num_buckets):
    points = np.arange(1 / num_buckets, 1.0, 1 / num_buckets)
    quantiles = np.quantile(np.array(lengths), points)

    indices = iter(sorted(range(len(lengths)), key=lambda idx: lengths[idx]))
    batch_sizes = [2 ** exp for exp in range(10, -1, -1)]

    partitions = []
    for q in quantiles.tolist() + [max(lengths)]:
        batch_size = itertools.dropwhile(lambda bs: max_length < bs * q, batch_sizes)
        partition = list(itertools.takewhile(lambda idx: lengths[idx] <= q, indices))
        partitions.append((next(batch_size), partition))

    return partitions


def collate(examples):
    batch_size = len(examples)
    max_input_len = max([e[INPUT_IDS].shape[0] for e in examples])
    max_label_len = max([e[DECODER_INPUT_IDS].shape[0] for e in examples])

    input_ids = np.full((batch_size, max_input_len), 2, dtype=np.int64)
    attention_mask = np.zeros_like(input_ids, dtype=np.bool)

    encoder_batch = {INPUT_IDS: input_ids, ATTENTION_MASK: attention_mask}

    decoder_input_ids = np.full((batch_size, max_label_len - 1), 2, dtype=np.int64)
    decoder_attention_mask = np.zeros_like(decoder_input_ids, dtype=np.bool)

    decoder_batch = {
        DECODER_INPUT_IDS: decoder_input_ids,
        DECODER_ATTENTION_MASK: decoder_attention_mask,
    }

    labels = np.full_like(decoder_input_ids, LABEL_MASK_ID)

    for idx, example in enumerate(examples):
        input_len = example[INPUT_IDS].shape[0]
        for key in encoder_batch:
            encoder_batch[key][idx, :input_len] = example[key]

        label_len = example[DECODER_INPUT_IDS].shape[0] - 1
        for key in decoder_batch:
            decoder_batch[key][idx, :label_len] = example[key][:-1]

        labels[idx, :label_len] = example[DECODER_INPUT_IDS][1:]

    batch = {LABELS: labels, **encoder_batch, **decoder_batch}
    return {key: torch.as_tensor(value) for key, value in batch.items()}

