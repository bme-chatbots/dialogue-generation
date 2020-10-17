"""
@author:    Patrik Purgai
@copyright: Copyright 2020, dialogue-generation
@license:   MIT
@email:     purgai.patrik@gmail.com
@date:      2020.05.30.
"""

import os
import pprint
import typing
import torch
import hydra
import logging
import omegaconf
import transformers
import bart

import pytorch_lightning as pl

transformers.logging.set_verbosity(transformers.logging.WARNING)
pl._logger.handlers = []


@hydra.main(config_path="config", config_name="config")
def main(config: omegaconf.OmegaConf):
    assert config.checkpoint_file is not None

    logging.info("\n" + omegaconf.OmegaConf.to_yaml(config, resolve=True))

    if config.seed is not None:
        pl.trainer.seed_everything(config.seed)

    tokenizer = transformers.BartTokenizer.from_pretrained(config.tokenizer_dir)
    lightning_model = bart.BARTModule.load_from_checkpoint(config.checkpoint_file)

    context = []

    while True:
        context.append(input("user >>> "))
        # with history size 0 only the last utterance will be used as input
        input_text = context[-(config.generation.history_size + 1) :]
        input_text = bart.build_example()

        input_ids = tokenizer(input_text, return_tensors="pt")["input_ids"]

        response_ids = lightning_model.model.generate(
            input_ids,
            **config.generation,
            pad_token_id=pad_token_id,
            bad_words_ids=role_token_ids,
        )

        output_ids = response_ids[0].tolist()[input_ids.size(-1) :]

        response = tokenizer.decode(output_ids, skip_special_tokens=True)
        print(f"bart >>> {response}")

        context.append(response)


if __name__ == "__main__":
    main()
