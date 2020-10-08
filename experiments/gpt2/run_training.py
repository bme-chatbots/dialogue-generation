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
import argparse
import omegaconf
import transformers
import gpt2

import pytorch_lightning as pl

transformers.logging.set_verbosity(transformers.logging.WARNING)
pl._logger.handlers = []


@hydra.main(config_path="config", config_name="config")
def main(config: omegaconf.OmegaConf):
    logging.info("\n" + omegaconf.OmegaConf.to_yaml(config, resolve=True))

    if config.seed is not None:
        pl.trainer.seed_everything(config.seed)

    try:
        tokenizer = transformers.GPT2TokenizerFast.from_pretrained(config.tokenizer_dir)
        logging.info(f"Loading existing tokenizer from {config.tokenizer_dir}")

    except OSError:
        logging.info(f"Tokenizer not found in {config.tokenizer_dir}")

        tokenizer = transformers.GPT2TokenizerFast.from_pretrained(
            config.model.pretrained_name
        )

        tokenizer.add_special_tokens({"pad_token": gpt2.PAD})
        tokenizer.add_tokens(gpt2.SPECIAL_TOKENS[2:], special_tokens=True)

        os.makedirs(config.tokenizer_dir, exist_ok=True)
        tokenizer.save_pretrained(config.tokenizer_dir)
        logging.info(f"Tokenizer saved to {config.tokenizer_dir}")

    if config.checkpoint_file is not None:
        trainer = pl.Trainer(
            **config.trainer, resume_from_checkpoint=config.checkpoint_file
        )

        model = gpt2.GPT2Module.load_from_checkpoint(
            config.checkpoint_file, **config.model, tokenizer=tokenizer
        )

    else:
        omegaconf.OmegaConf.set_struct(config, False)

        config.model.vocab_size = len(tokenizer)
        config.model.pad_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

        model = gpt2.GPT2Module(argparse.Namespace(**config.model))

        model_checkpoint = gpt2.ModelCheckpoint(
            "{epoch}-{accuracy:.2f}", save_top_k=1, save_last=True
        )

        early_stopping = pl.callbacks.EarlyStopping()

        callbacks = [pl.callbacks.LearningRateLogger(logging_interval="step")]

        trainer = pl.Trainer(
            **config.trainer,
            checkpoint_callback=model_checkpoint,
            early_stop_callback=early_stopping,
            callbacks=callbacks,
        )

    data_module = gpt2.GPT2DataModule(config.data, tokenizer)
    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    main()
