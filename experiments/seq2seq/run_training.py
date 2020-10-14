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
import seq2seq

import pytorch_lightning as pl

transformers.logging.set_verbosity(transformers.logging.WARNING)
pl._logger.handlers = []


@hydra.main(config_path="config", config_name="config")
def main(config: omegaconf.OmegaConf):
    logging.info("\n" + omegaconf.OmegaConf.to_yaml(config))

    if config.seed is not None:
        pl.trainer.seed_everything(config.seed)

    speaker_tokens = [seq2seq.SPEAKER_FROM, seq2seq.SPEAKER_TO]

    encoder_tokenizer = seq2seq.load_or_build_tokenizer(config.encoder, speaker_tokens)

    decoder_tokenizer = seq2seq.load_or_build_tokenizer(
        config.decoder, [seq2seq.SPEAKER_FROM, seq2seq.SOS], {"pad_token": seq2seq.PAD}
    )

    if config.checkpoint_file is not None:
        trainer = pl.Trainer(
            **config.trainer, resume_from_checkpoint=config.checkpoint_file
        )

        model = seq2seq.Seq2SeqModule.load_from_checkpoint(
            config.checkpoint_file, **config.model, tokenizer=tokenizer
        )

    else:
        omegaconf.OmegaConf.set_struct(config, False)

        config.model.encoder.vocab_size = len(encoder_tokenizer)
        config.model.decoder.vocab_size = len(decoder_tokenizer)

        config.model.pad_id = decoder_tokenizer.convert_tokens_to_ids(
            decoder_tokenizer.pad_token
        )

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
