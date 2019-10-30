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

from inspect import getfullargspec

from torch.nn.modules import (
    ModuleList, Module)

from torch.nn import (
    Linear, Parameter)

from transformers import (
    XLNetLMHeadModel, XLNetConfig,
    XLNetModel, GPT2Config,
    GPT2LMHeadModel)

from transformers.modeling_xlnet import XLNetLayer
from transformers.modeling_gpt2 import Block

from datetime import datetime
from collections import namedtuple

from torch.utils.checkpoint import checkpoint

from os.path import (
    exists, join,
    dirname, abspath)


def setup_model_args(parser):
    """
    Sets up the model arguments.
    """
    group = parser.add_argument_group('model')
    group.add_argument(
        '-m', '--model',
        type=str,
        default='xlnet-base-cased',
        choices=list(MODEL),
        help='Name of the model.')
    group.add_argument(
        '--grad_ckpt',
        action='store_true',
        help='Use checkpointed models.')
    group.add_argument(
        '--name',
        type=str,
        default=datetime.today().strftime(
            '%y.%m.%d-%H:%M:%S'),
        help='Name of the trained model instance.')
    group.add_argument(
        '--model_dir',
        type=str,
        default=join(abspath(dirname(__file__)), 
            '..', 'model'),
        help='Path of the model checkpoints.')


def create_model(args, model_dir, vocab_size):
    """
    Creates the classifier and encoder model.
    """
    pretrained_dir = join(args.model_dir, args.model)

    model_path = join(
        pretrained_dir, 'pytorch_model.bin')

    model_cls = MODEL[args.model](args)

    if not exists(model_path):
        model = model_cls.from_pretrained(
            args.model)

        model.resize_token_embeddings(vocab_size)
        model.save_pretrained(pretrained_dir)

    else:
        model = model_cls.from_pretrained(
            pretrained_dir)

    return model


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
    return tensor.half() if half else tensor.float()


class CkptGPT2Layer(Module):
    """
    Wrapper class to perform checkpointing of
    Block ( GPT2Layer ) modue.
    """

    def __init__(self, module):
        super().__init__()

        self.module = module

    def forward_wrapper(self, *args, **kwargs):
        """
        Converts the forward function's output
        to a tuple.
        """
        return tuple(
            self.module.forward(*args, **kwargs))

    def forward(
            self, x, layer_past=None, 
            attention_mask=None, head_mask=None):

        return checkpoint(
            self.forward_wrapper, x, layer_past,
            attention_mask, head_mask)


class CkptXLNetLayer(Module):
    """
    Wrapper class to perform checkpointing of
    XLNetLayer modue.
    """

    def __init__(self, module):
        super().__init__()
        
        self.module = module

    def forward(
            self, output_h, output_g, attn_mask_h,
            attn_mask_g, r, seg_mat, mems=None, 
            target_mapping=None, head_mask=None):
        # delegating the parameters of the original
        # XLNetLayer module to the checkpoint call

        return checkpoint(
            self.module.forward, output_h, output_g,
            attn_mask_h, attn_mask_g, r, seg_mat,
            mems, target_mapping, head_mask)


def create_xlnet_model(args):
    """
    Creates an XLNet generator model.
    """
    class XLNetGenerator(XLNetLMHeadModel):
        """
        Generator model based on XLNet language model.
        """

        def __init__(self, config):
            super().__init__(config)

            if args.grad_ckpt:
                self.transformer.layer = ModuleList([
                    CkptXLNetLayer(layer) for
                    layer in self.transformer.layer
                ])
        
        def resize_bias(self, new_num_tokens=None):
            """
            Fix that also copies the weights of bias
            in the output layer.
            """
            new_bias = torch.zeros(new_num_tokens)

            old_size = self.lm_loss.bias.size(0)

            new_bias[:old_size] = self.lm_loss.bias
            self.lm_loss.bias = Parameter(new_bias)

        def resize_token_embeddings(
                self, new_num_tokens=None):
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

    return XLNetGenerator


def create_gpt2_model(args):
    """
    Creates a gpt2 generator model.
    """
    class GPT2Generator(GPT2LMHeadModel):
        """
        Generator model based on GPT2 language model.
        """

        def __init__(self, config):
            super().__init__(config)

            if args.grad_ckpt:
                self.transformer.h = ModuleList([
                    CkptGPT2Layer(layer) for
                    layer in self.transformer.h
                ])

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

    return GPT2Generator


MODEL = {
    'xlnet-base-cased':     create_xlnet_model, 
    'xlnet-large-cased':    create_xlnet_model, 
    'distilgpt2':           create_gpt2_model,
    'gpt2':                 create_gpt2_model,
    'gpt2-medium':          create_gpt2_model,
    'gpt2-large':           create_gpt2_model
}
