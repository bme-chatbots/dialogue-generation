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

import pytorch_lightning as pl
import numpy as np

PROJECT_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), "..", "..")

# this implementation uses the token_type_id field of the gpt2 transformer
# for signaling the source of the current utterance
BOT = "<|bot|>"
USR = "<|user|>"

# other than the native eos_id the dialogue model uses sos_id and pad_id
# as special control
EOS = "<|endoftext|>"
PAD = "<|padding|>"

SPECIAL_TOKENS = [EOS, PAD, BOT, USR]

SPLITS = TRAIN, VALID = datasets.Split.TRAIN, datasets.Split.VALIDATION

COLUMNS = [INPUT_IDS, ATTENTION_MASK] = ["input_ids", "attention_mask"]

LABEL_IDS = "label_ids"


# custom modelcheckpoint is required to overwrite the formatting in the file name
# as default "=" symbol conflicts with hydra's argument parsing
class ModelCheckpoint(pl.callbacks.ModelCheckpoint):
    def format_checkpoint_name(self, *args, **kwargs):
        return super().format_checkpoint_name(*args, **kwargs).replace("=", ":")


class GPT2Module(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams

        self.model = transformers.GPT2LMHeadModel.from_pretrained(
            self.hparams.pretrained_name
        )

        self.model.resize_token_embeddings(self.hparams.vocab_size)

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


class GPT2DataModule(pl.LightningDataModule):
    def __init__(self, config, tokenizer):
        super().__init__()

        self.config = config
        self.tokenizer = tokenizer
        self.dataset = None

        special_tokens = [PAD, BOT, USR]
        self.specials = self.tokenizer.convert_tokens_to_ids(special_tokens)

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
        dataset["lengths"]["length"], config.max_tokens, config.num_buckets, shuffle
    )

    return torch.utils.data.DataLoader(
        dataset["examples"],
        pin_memory=True,
        num_workers=config.num_workers,
        collate_fn=functools.partial(collate, specials=specials),
        batch_sampler=bucket_sampler,
    )


def flatten_dialogue(dialogue, input_field, output_field):
    # each dialogue is flattened into a single string where the utterances are
    # preceided by a special role token and finished with an EOS token
    return {
        output_field: "".join(
            (BOT if idx % 2 else USR) + utterance + EOS
            for idx, utterance in enumerate(dialogue[input_field])
        )
    }


def build_split(examples, tokenizer, max_tokens, cache_dir, split_name, field):
    flat_field = f"flat_{field}"

    examples = examples.map(
        functools.partial(flatten_dialogue, input_field=field, output_field=flat_field),
        remove_columns=examples.column_names,
        cache_file_name=os.path.join(cache_dir, f"{split_name}.raw.cache"),
    )

    examples = examples.map(
        lambda example: tokenizer(
            example[flat_field],
            return_tensors="np",
            max_length=max_tokens,
            truncation=True,
        ),
        remove_columns=examples.column_names,
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
            max_tokens=config.max_tokens,
            cache_dir=config.cache_dir,
            split_name=split,
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
