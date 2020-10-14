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

import pytorch_lightning as pl
import numpy as np


EXPERIMENT_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_DIR = os.path.join(EXPERIMENT_DIR, "..", "..")

# this implementation uses the token_type_id field of the gpt2 transformer
# for signaling the source of the current utterance
SPEAKER_FROM = "<|speaker_from|>"
SPEAKER_TO = "<|speaker_to|>"

# other than the native eos_id the dialogue model uses sos_id and pad_id
# as special control
EOS = "<|endoftext|>"
SOS = "<|startoftext|>"
PAD = "<|padding|>"

ENCODER, DECODER = "encoder", "decoder"

SPLITS = TRAIN, VALID = datasets.Split.TRAIN, datasets.Split.VALIDATION

INPUT_IDS, SPEAKER_IDS, LABELS = "input_ids", "speaker_ids", "labels"
INPUT_MASK, LABEL_MASK = "input_mask", "label_mask"
ATTENTION_MASK = "attention_mask"


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
        # self.model.get_decoder.resize_token_embeddings(

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
    def __init__(self, config, encoder_tokenizer, decoder_tokenizer):
        super().__init__()

        self.config = config
        self.tokenizer = {ENCODER: encoder_tokenizer, DECODER: decoder_tokenizer}
        self.dataset = None

        self.pad_id = {
            ENCODER: encoder_tokenizer.convert_tokens_to_ids(PAD),
            DECODER: decoder_tokenizer.convert_tokens_to_ids(PAD),
        }

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
        return build_dataloader(self.dataset[TRAIN], self.config, self.pad_id)

    def val_dataloader(self):
        return build_dataloader(self.dataset[VALID], self.config, self.pad_id, False)


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


def build_dataloader(dataset, config, pad_id, shuffle=True):
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
        collate_fn=functools.partial(collate, pad_id=pad_id),
        batch_sampler=bucket_sampler,
    )


def load_or_build_tokenizer(config, additional_tokens=None, special_tokens=None):
    try:
        tokenizer = load_tokenizer(config.pretrained_name, config.tokenizer_dir)

    except OSError:
        # tokenizer does not exist in the given path thereby creating it there
        logging.info(f"Tokenizer not found in {config.tokenizer_dir}")
        tokenizer = build_tokenizer(
            config.pretrained_name,
            config.tokenizer_dir,
            additional_tokens,
            special_tokens,
        )

    return tokenizer


def load_tokenizer(pretrained_name, tokenizer_dir):
    config = transformers.AutoConfig.from_pretrained(pretrained_name)

    # there is an issue about loading a tokenizer from local directory with the
    # AutoModel class as it requires the configuration file to identify the model
    # https://github.com/huggingface/transformers/issues/4197
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_dir, config=config)

    return tokenizer


def build_tokenizer(
    pretrained_name, tokenizer_dir, additional_tokens=None, special_tokens=None
):
    tokenizer = transformers.AutoTokenizer.from_pretrained(pretrained_name)
    if special_tokens is not None:
        tokenizer.add_special_tokens(special_tokens)

    if additional_tokens is not None:
        tokenizer.add_tokens(additional_tokens, special_tokens=True)

    tokenizer.save_pretrained(tokenizer_dir)

    logging.info(f"Saving tokenizer to {tokenizer_dir}")

    return tokenizer


def build_dataset(tokenizer, config):
    data_files = {
        str(split): glob.glob(config.get(str(split)).data_pattern) for split in SPLITS
    }

    dataset = datasets.load_dataset("json", data_files=data_files)

    # splits contains a dictionary of dictionary of datasets with 2 keys `examples`
    # contains the dataset with the inputs and `lengths` contains the size of each
    # example which is used by the batch sampler
    splits = {
        str(split_name): build_split(examples, tokenizer, str(split_name), config)
        for split_name, examples in dataset.items()
    }

    return splits


def build_split(examples, tokenizer, split_name, config):
    examples = examples.map(
        lambda example: build_examples(example, tokenizer, config),
        batched=True,
        remove_columns=[config.field],
        cache_file_name=os.path.join(config.cache_dir, f"{split_name}.examples.cache"),
    )

    examples.set_format(type="np", columns=examples.column_names)

    lengths = examples.map(
        lambda example: {"length": example[INPUT_IDS].size},
        remove_columns=examples.column_names,
        cache_file_name=os.path.join(config.cache_dir, f"{split_name}.lengths.cache"),
    )

    return {"examples": examples, "lengths": lengths}


def build_examples(examples, tokenizer, config):
    def generate_examples():
        for dialogue in examples[config.field]:
            assert len(dialogue) > 1

            for idx in range(2, len(dialogue)):
                yield build_example(dialogue[:idx], dialogue[idx], tokenizer, config)

    input_ids, speaker_ids, input_mask, labels, label_mask = zip(*generate_examples())

    return {
        INPUT_IDS: list(input_ids),
        SPEAKER_IDS: list(speaker_ids),
        INPUT_MASK: list(input_mask),
        LABELS: list(labels),
        LABEL_MASK: list(label_mask),
    }


def build_example(input_text, label_text, tokenizer, config):
    input_text = [
        get_speaker_token(idx) + utterance
        for idx, utterance in enumerate(reversed(input_text))
    ]

    example = tokenizer[ENCODER](input_text, add_special_tokens=False)

    speaker_tokens = [
        "".join([get_speaker_token(idx)] * len(utterance))
        for idx, utterance in enumerate(example[INPUT_IDS])
    ]

    speaker_dict = tokenizer[ENCODER](speaker_tokens, add_special_tokens=False)
    example[SPEAKER_IDS] = speaker_dict[INPUT_IDS]

    example = {
        key: list(itertools.chain(*value))[-config.max_input_tokens :]
        for key, value in example.items()
    }

    label_dict = tokenizer[DECODER](SOS + label_text + EOS, add_special_tokens=False)
    labels = label_dict[INPUT_IDS][: config.max_label_tokens]
    label_mask = label_dict[ATTENTION_MASK][: config.max_label_tokens]

    return (
        example[INPUT_IDS],
        example[SPEAKER_IDS],
        example[ATTENTION_MASK],
        labels,
        label_mask,
    )


def get_speaker_token(idx):
    # if the idx is even then SPEAKER_TO id is assigned to the idx
    return SPEAKER_TO if idx % 2 else SPEAKER_FROM


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


def collate(batch, pad_id):
    batch_size = len(batch)
    max_input_len = max([e[INPUT_IDS].shape[0] for e in batch])
    max_label_len = max([e[LABELS].shape[0] for e in batch])

    input_ids = np.full((batch_size, max_input_len), pad_id[ENCODER], dtype=np.int64)
    speaker_ids = np.copy(input_ids)
    input_mask = np.zeros_like(input_ids, dtype=np.int8)

    labels = np.full((batch_size, max_label_len), pad_id[DECODER], dtype=np.int64)
    label_mask = np.zeros_like(labels, dtype=np.int8)

    for idx, example in enumerate(batch):
        input_len = example[INPUT_IDS].shape[0]
        input_ids[idx, :input_len] = example[INPUT_IDS]
        speaker_ids[idx, :input_len] = example[SPEAKER_IDS]
        input_mask[idx, :input_len] = example[INPUT_MASK]

        label_len = example[LABELS].shape[0]
        labels[idx, :label_len] = example[LABELS]
        label_mask[idx, :label_len] = example[LABEL_MASK]

    return {
        INPUT_IDS: torch.as_tensor(input_ids),
        SPEAKER_IDS: torch.as_tensor(speaker_ids),
        INPUT_MASK: torch.as_tensor(input_mask, dtype=torch.bool),
        LABELS: torch.as_tensor(labels),
        LABEL_MASK: torch.as_tensor(label_mask, dtype=torch.bool),
    }
