"""
@author:    Patrik Purgai
@copyright: Copyright 2020, dialogue-generation
@license:   MIT
@email:     purgai.patrik@gmail.com
@date:      2020.05.30.
"""

import os
import torch
import hydra
import logging
import typing
import tqdm
import argparse
import omegaconf
import transformers
import dataclasses
import bart

import pytorch_lightning as pl

transformers.logging.set_verbosity(transformers.logging.WARNING)
pl._logger.handlers = []


@hydra.main(config_path="config", config_name="config")
def main(config: omegaconf.OmegaConf):
    logging.info("\n" + omegaconf.OmegaConf.to_yaml(config))

    if config.seed is not None:
        pl.trainer.seed_everything(config.seed)

    try:
        tokenizer = transformers.BartTokenizer.from_pretrained(config.tokenizer_dir)
        logging.info(f"Loading existing tokenizer from {config.tokenizer_dir}")

    except OSError:
        logging.info(f"Tokenizer not found in {config.tokenizer_dir}")

        tokenizer = transformers.BartTokenizer.from_pretrained(
            config.model.pretrained_name
        )

        tokenizer.add_tokens([bart.SPEAKER_FROM, bart.SPEAKER_TO], special_tokens=True)

        os.makedirs(config.tokenizer_dir, exist_ok=True)
        tokenizer.save_pretrained(config.tokenizer_dir)
        logging.info(f"Tokenizer saved to {config.tokenizer_dir}")

    if config.checkpoint_file is not None:
        trainer = pl.Trainer(
            **config.trainer, resume_from_checkpoint=config.checkpoint_file
        )

        model = bart.BARTModule.load_from_checkpoint(
            config.checkpoint_file, tokenizer=tokenizer
        )

    else:
        omegaconf.OmegaConf.set_struct(config, False)

        config.model.vocab_size = len(tokenizer)

        model = bart.BARTModule(argparse.Namespace(**config.model))

        model_checkpoint = bart.ModelCheckpoint(
            "{epoch}-{accuracy:.2f}", "loss", save_top_k=1, save_last=True
        )

        trainer = pl.Trainer(
            **config.trainer,
            checkpoint_callback=model_checkpoint,
            callbacks=[pl.callbacks.EarlyStopping("loss")],
        )

    data_module = bart.BARTDataModule(config.data, tokenizer)
    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    main()
