"""

@author:    Patrik Purgai
@copyright: Copyright 2019, dialogue-generation
@license:   MIT
@email:     purgai.patrik@gmail.com
@date:      2019.04.04.
"""

# pylint: disable=no-member
# pylint: disable=not-callable

import torch
import random

from torch.nn.modules import (
    ModuleList)

from torch.nn import (
    Linear)

from pytorch_transformers import (
    XLNetLMHeadModel,
    XLNetConfig,
    XLNetModel)

from os.path import (
    exists, join)


def setup_model_args(parser):
    """
    Sets up the model arguments.
    """


def create_model(args, vocab_size, device):
    """
    Creates the classifier and encoder model.
    """
    config_path = join(args.model_dir, 'config.json')

    if not exists(config_path):
        config = XLNetConfig.from_pretrained(
            'xlnet-base-cased')

        generator = XLNetGenerator(config)
        generator.resize_token_embeddings(vocab_size)

        config.save_pretrained(args.model_dir)

    # TODO huggingface output bias layer is bugged
    # if the size of the embeddings is modified
    # reloading the model with new config 
    # fixes the problem
    config = XLNetConfig.from_pretrained(
        args.model_dir)

    generator = XLNetGenerator(config)
    
    generator = generator.to(device)

    return generator


def compute_size(model):
    """
    Computes the number of parameters of the model.
    """
    return sum(
        p.numel() for 
        p in model.parameters())


class XLNetGenerator(XLNetLMHeadModel):
    """
    A version of XLNetLMHead that has bias
    disabled in the projection layer.
    """

    def __init__(self, config):
        super().__init__(config)

        # TODO remove temporary solution that we
        # are only using the first 4 layers of
        # XLNet for faster testing speed
        self.transformer.layer = ModuleList([
            layer for layer 
            in self.transformer.layer[:4]
        ])

        self.lm_loss = Linear(
            config.d_model, config.n_token)

        self.apply(self.init_weights)
        self.tie_weights()
