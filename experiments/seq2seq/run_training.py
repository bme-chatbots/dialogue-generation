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
import seq2seq

import pytorch_lightning as pl

transformers.logging.set_verbosity(transformers.logging.WARNING)
pl._logger.handlers = []


@hydra.main(config_path="config", config_name="config")
def main(config: omegaconf.OmegaConf):
    logging.info("\n" + omegaconf.OmegaConf.to_yaml(config))

    if config.seed is not None:
        pl.trainer.seed_everything(config.seed)

    try:
        encoder_tokenizer = load_tokenizer(
            config.encoder_pretrained_name, config.encoder_tokenizer_dir
        )

    try:
        encoder_tokenizer = transformers.GPT2TokenizerFast.from_pretrained(
            config.encoder_tokenizer_dir
        )

        logging.info(f"Loading existing tokenizer from {config.tokenizer_dir}")

    except OSError:
        logging.info(f"Tokenizer not found in {config.tokenizer_dir}")

        tokenizer = transformers.GPT2TokenizerFast.from_pretrained(
            config.model.pretrained_name
        )

        tokenizer.add_special_tokens({"pad_token": seq2seq.PAD})
        tokenizer.add_tokens(seq2seq.SPECIAL_TOKENS[2:], special_tokens=True)

        os.makedirs(config.tokenizer_dir, exist_ok=True)
        tokenizer.save_pretrained(config.tokenizer_dir)
        logging.info(f"Tokenizer saved to {config.tokenizer_dir}")

    data_module = seq2seq.Seq2SeqDataModule(config.data, tokenizer)
    data_module.prepare_data()
    data_module.setup()

    for batch in data_module.train_dataloader():
        print(batch)

    if config.checkpoint_file is not None:
        trainer = pl.Trainer(
            **config.trainer, resume_from_checkpoint=config.checkpoint_file
        )

        model = seq2seq.Seq2SeqModule.load_from_checkpoint(
            config.checkpoint_file, **config.model, tokenizer=tokenizer
        )

    else:
        omegaconf.OmegaConf.set_struct(config, False)

        config.model.vocab_size = len(tokenizer)
        config.model.pad_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

        model = seq2seq.Seq2SeqModule(argparse.Namespace(**config.model))

        model_checkpoint = seq2seq.ModelCheckpoint(
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

    data_module = gpt2.DialogDataModule(config.data, tokenizer)
    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    main()
