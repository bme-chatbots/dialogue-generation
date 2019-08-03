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


def setup_model_args(parser):
    """
    Sets up the model arguments.
    """


def create_model(args, vocab_size, device):
    """
    Creates the classifier and encoder model.
    """
    config = XLNetConfig.from_pretrained(
        'xlnet-base-cased')

    generator = XLNetGenerator(config)
    generator.resize_token_embeddings(vocab_size)
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

        # self.transformer.layer = ModuleList([
        #     layer for layer 
        #     in self.transformer.layer[:2]
        # ])

        self.lm_loss = Linear(
            config.d_model, config.n_token, 
            bias=False)

        self.apply(self.init_weights)
        self.tie_weights()
