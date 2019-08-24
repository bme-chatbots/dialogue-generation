"""

@author:    Patrik Purgai
@copyright: Copyright 2019, dialogue-generation
@license:   MIT
@email:     purgai.patrik@gmail.com
@date:      2019.04.04.
"""

# pylint: disable=no-member
# pylint: disable=not-callable

import os
import torch
import random

from torch.nn.modules import (
    ModuleList)

from torch.nn import (
    Linear, Parameter)

from pytorch_transformers import (
    XLNetLMHeadModel, XLNetConfig,
    XLNetModel, GPT2Config,
    GPT2LMHeadModel)

from datetime import datetime
from collections import namedtuple

from os.path import (
    exists, join,
    dirname, abspath)


def setup_model_args(parser):
    """
    Sets up the model arguments.
    """
    group = parser.add_argument_group('model')
    group.add_argument(
        '--model_name',
        type=str,
        default='xlnet-base-cased',
        help='Name of the model.')
    group.add_argument(
        '--model_dir',
        type=str,
        default=join(abspath(dirname(__file__)), 
            '..', 'model'),
        help='Path of the model checkpoints.')


def create_model(args, vocab_size, device):
    """
    Creates the classifier and encoder model.
    """
    model_dir = join(args.model_dir, args.model_name)
    os.makedirs(model_dir, exist_ok=True)

    model_path = join(model_dir, 'pytorch_model.bin')

    assert args.model_name in MODEL, \
        'Available models: {} received `{}`'.format(
            ', '.join(MODEL), args.model_name)

    model_cls = MODEL[args.model_name]

    if not exists(model_path):
        generator = model_cls.from_pretrained(
            args.model_name)

        generator.resize_token_embeddings(vocab_size)
        generator.save_pretrained(model_dir)

    else:
        generator = model_cls.from_pretrained(
            model_dir)

    generator = generator.to(device)

    return generator


def compute_size(model):
    """
    Computes the number of parameters of the model.
    """
    return sum(
        p.numel() for 
        p in model.parameters())


def convert_to_float(tensor, half=False):
    """
    Converts the tensor to either float32
    or float16 based on the parameter.
    """
    if half:
        tensor = tensor.half()
    else:
        tensor = tensor.float()

    return tensor


class XLNetGenerator(XLNetLMHeadModel):
    """
    Generator model based on XLNet language model.
    """
    
    def resize_bias(self, new_num_tokens=None):
        """
        Fix that also copies the weights of bias
        in the output layer.
        """
        new_bias = torch.zeros(new_num_tokens)

        old_size = self.lm_loss.bias.size(0)

        new_bias[:old_size] = self.lm_loss.bias
        self.lm_loss.bias = Parameter(new_bias)

    def resize_token_embeddings(self, new_num_tokens=None):
        """
        Extends the resize fn by resizing the bias layer.
        """
        super().resize_token_embeddings(new_num_tokens)
        self.resize_bias(new_num_tokens)

    def forward(self, inputs, half=False):
        # converting the batch of inputs to torch tensor
        parameter = next(self.parameters())

        inputs = [
            torch.as_tensor(t).to(
                parameter.device, 
                non_blocking=True) 
            for t in inputs
        ]

        input_ids, token_type_ids, attn_mask, \
            perm_mask, target_map = inputs

        attn_mask = convert_to_float(
            tensor=attn_mask, half=half)

        perm_mask = convert_to_float(
            tensor=perm_mask, half=half)

        target_map = convert_to_float(
            tensor=target_map, half=half)

        outputs = super().forward(
            input_ids=input_ids.long(),
            token_type_ids=token_type_ids.long(),
            attention_mask=attn_mask,
            perm_mask=perm_mask,
            target_mapping=target_map)

        return outputs


class GPT2Generator(GPT2LMHeadModel):
    """
    Generator model based on GPT2 language model.
    """

    def forward(self, inputs, half=False):
        # converting the batch of inputs to torch tensor
        device = next(self.parameters()).device

        inputs = [
            torch.as_tensor(t).to(
                device, non_blocking=True) 
            for t in inputs
        ]

        input_ids, token_type_ids = inputs

        outputs = super().forward(
            input_ids=input_ids.long(),
            token_type_ids=token_type_ids.long())
        
        return outputs


MODEL = {
    'xlnet-base-cased':     XLNetGenerator, 
    'xlnet-large-cased':    XLNetGenerator, 
    'gpt2':                 GPT2Generator,
    'gpt2-medium':          GPT2Generator,
    'gpt2-large':           GPT2Generator
}
