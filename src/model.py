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

from torch.nn.modules import Module
from torch.nn.functional import (
    log_softmax, softmax)

from torch.nn import (
    NLLLoss, LSTM, 
    Embedding, Linear,
    Softmax, Dropout)


from pytorch_transformers import XLNetModel


NEAR_INF = 1e20


def setup_model_args(parser):
    """
    Sets up the model arguments.
    """
    parser.add_argument(
        '--hidden_size',
        type=int,
        default=256,
        help='Hidden size of the model.')
    parser.add_argument(
        '--embedding_size',
        type=int,
        default=128,
        help='Embedding dimension for the tokens.')
    

def create_model(args, vocab_size, device):
    pass
