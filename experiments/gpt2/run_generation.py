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
import gpt2

import pytorch_lightning as pl

transformers.logging.set_verbosity(transformers.logging.WARNING)
pl._logger.handlers = []


# identical to the function in gpt2.DialogDataModule.build_dataset but ensures
# that the final utterance is encoded with the USR token
def prepare_dialogue(inputs: typing.List[str]) -> str:
    input_text = "".join(
        [
            (gpt2.BOT if i % 2 else gpt2.USR) + u + gpt2.EOS
            for i, u in enumerate(inputs[::-1])
        ][::-1]
    )

    # adding starting bot token to the ids
    return input_text + gpt2.BOT


@hydra.main(config_path="config", config_name="config")
def main(config: omegaconf.OmegaConf):
    assert config.checkpoint_file is not None

    logging.info("\n" + omegaconf.OmegaConf.to_yaml(config, resolve=True))

    if config.seed is not None:
        pl.trainer.seed_everything(config.seed)

    tokenizer = transformers.GPT2TokenizerFast.from_pretrained(config.tokenizer_dir)

    special_tokens = [tokenizer.pad_token, gpt2.BOT, gpt2.USR]
    special_token_ids = tokenizer.convert_tokens_to_ids(special_tokens)

    pad_token_id, *role_token_ids = special_token_ids
    role_token_ids = [[token_id] for token_id in role_token_ids]

    lightning_model = gpt2.GPT2Module.load_from_checkpoint(config.checkpoint_file)

    context = []

    while True:
        context.append(input("user >>> "))
        # with history size 0 only the last utterance will be used as input
        input_text = prepare_dialogue(context[-(config.generation.history_size + 1) :])
        input_ids = tokenizer(input_text, return_tensors="pt")["input_ids"]

        response_ids = lightning_model.model.generate(
            input_ids,
            **config.generation,
            pad_token_id=pad_token_id,
            bad_words_ids=role_token_ids,
        )

        output_ids = response_ids[0].tolist()[input_ids.size(-1) :]

        response = tokenizer.decode(output_ids, skip_special_tokens=True)
        print(f"gpt2 >>> {response}")

        context.append(response)


if __name__ == "__main__":
    main()
