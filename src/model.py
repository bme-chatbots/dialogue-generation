# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import torch
import argparse

from math import sqrt, pi

from torch.nn.modules import (
    Embedding, Linear, Dropout,
    LayerNorm)

from torch.nn.init import normal_

from torch.nn.functional import (
    softmax, linear)


def setup_model_args(parser):
    """
    Parses the model related arguments.
    """
    parser.add_argument(
        '--pretrained',
        default='124M',
        choices=['124M', '355M', '774M', '1558M'],
        help='Name of the pretrained model to use.')
    parser.add_argument(
        '--output_attentions',
        action='store_true',
        help='Flag whether to output attention values.')
    parser.add_argument(
        '--output_hidden_states',
        action='store_true',
        help='Flag whether to output hidden state values.')
    parser.add_argument(
        '--output_past',
        action='store_true',
        help='Output past values.')
    parser.add_argument(
        '--vocab_size',
        type=int,
        default=None,
        help='Size of the model vocabulary.')
    parser.add_argument(
        '--n_positions',
        type=int,
        default=1024,
        help='Size of the position embedding.')
    parser.add_argument(
        '--resid_pdrop',
        type=float,
        default=0.1,
        help='Probability for the residual dropout.')
    parser.add_argument(
        '--attn_pdrop',
        type=float,
        default=0.1,
        help='Probability for the attention dropout.')
    parser.add_argument(
        '--embd_pdrop',
        type=float,
        default=0.1,
        help='Probabilirt for the embedding dropout.')
    parser.add_argument(
        '--layer_norm_epsilon',
        type=float,
        default=1e-5,
        help='Epsilon value for layer normalization.')
    parser.add_argument(
        '--initializer_range',
        type=float,
        default=0.2,
        help='Initialization range dor the variables.')


def gelu(x):
    """
    Gaussian Error Linear Unit.
    This is a smoother version of the ReLU.
    """
    return x * 0.5 * (
        1 + torch.tanh(
            sqrt(2 / pi) *
            (x + 0.044715 * torch.pow(x, 3))))


# based on Huggingface/transformers implementation
class Conv1D(torch.nn.Module):
    """
    Basically works like a Linear layer but the
    weights are transposed.
    """

    def __init__(self, d_out, d_in):
        super().__init__()

        self.d_out = d_out

        weight = torch.empty(d_in, d_out)
        normal_(weight, std=0.02)

        self.weight = torch.nn.Parameter(weight)
        self.bias = torch.nn.Parameter(torch.zeros(d_out))

    def forward(self, inputs):
        size_out = inputs.size()[:-1] + (self.d_out, )

        out = torch.addmm(
            self.bias,
            inputs.view(-1, inputs.size(-1)),
            self.weight)

        out = out.view(*size_out)

        return out


# based on Huggingface/transformers implementation
class Attention(torch.nn.Module):
    """
    Self attention module for the GPT-2 model.
    """

    def __init__(
            self, n_embd, n_ctx, config, scale=False):
        super().__init__()

        self.output_attentions = config.output_attentions

        self.register_buffer(
            'bias', torch.tril(
                torch.ones(n_ctx, n_ctx)).view(
                    1, 1, n_ctx, n_ctx))

        self.n_head = config.n_head
        self.split_size = n_embd
        self.scale = scale

        self.c_attn = Conv1D(n_embd * 3, n_embd)
        self.c_proj = Conv1D(n_embd, n_embd)

        self.attn_dropout = Dropout(config.attn_pdrop)
        self.resid_dropout = Dropout(config.resid_pdrop)

    def _attn(self, query, key, value, attn_mask=None):
        scores = torch.matmul(query, key)

        if self.scale:
            scores = scores / value.size(-1) ** 0.5

        nd, ns = scores.size()[-2:]

        bias = self.bias[:, :, ns - nd : ns, :ns]
        scores = scores * bias - 1e4 * (1 - bias)

        if attn_mask is not None:
            scores = scores.masked_fill(
                attn_mask, float('-inf'))

        scores = softmax(scores, dim=-1)
        scores = self.attn_dropout(scores)

        outputs = [torch.matmul(scores, value)]

        if self.output_attentions:
            outputs.append(scores)

        return outputs

    def merge_heads(self, inputs):
        inputs = inputs.permute(0, 2, 1, 3).contiguous()

        new_shape = inputs.size()[:-2] + (
            inputs.size(-2) * inputs.size(-1), )

        return inputs.view(*new_shape)

    def split_heads(self, inputs, is_key=False):
        new_shape = inputs.size()[:-1] + \
            (self.n_head, inputs.size(-1) // self.n_head)

        inputs = inputs.view(*new_shape)

        if is_key:
            return inputs.permute(0, 2, 3, 1)
        else:
            return inputs.permute(0, 2, 1, 3)

    def forward(
            self, inputs, past=None, attn_mask=None):
        out = self.c_attn(inputs)

        query, key, value = out.split(
            self.split_size, dim=2)

        query = self.split_heads(query)
        key = self.split_heads(key, is_key=True)
        value = self.split_heads(value)

        if past is not None:
            past_key, past_value = \
                past[0].transpose(-2, -1), past[1]

            key = torch.cat((past_key, key), dim=-1)
            value = torch.cat((past_value, value), dim=-2)

        present = torch.stack(
            (key.transpose(-2, -1), value))

        attn_outputs = self._attn(
            query, key, value, attn_mask)

        attn_out = attn_outputs[0]

        attn_out = self.merge_heads(attn_out)
        attn_out = self.c_proj(attn_out)
        attn_out = self.resid_dropout(attn_out)

        return [attn_out, present] + attn_outputs[1:]


# based on Huggingface/transformers implementation
class MLP(torch.nn.Module):
    """
    Fully connected layer for GPT-2 layer.
    """

    def __init__(self, n_state, config):
        super().__init__()

        n_embd = config.n_embd
        self.c_fc = Conv1D(n_state, n_embd)
        self.c_proj = Conv1D(n_embd, n_state)
        self.act = gelu
        self.dropout = Dropout(config.resid_pdrop)

    def forward(self, inputs):
        out = self.act(self.c_fc(inputs))
        out = self.c_proj(out)

        return self.dropout(out)


# based on Huggingface/transformers implementation
class Block(torch.nn.Module):
    """
    Layer block for the GPT-2 model.
    """

    def __init__(self, n_ctx, config, scale=False):
        super().__init__()

        n_embd = config.n_embd

        self.ln_1 = LayerNorm(
            n_embd, eps=config.layer_norm_epsilon)

        self.ln_2 = LayerNorm(
            n_embd, eps=config.layer_norm_epsilon)

        self.attn = Attention(
            n_embd, n_ctx, config, scale)

        self.mlp = MLP(4 * n_embd, config)

    def forward(
            self, inputs, past=None, attn_mask=None):

        attn_outputs = self.attn(
            self.ln_1(inputs),
            past=past,
            attn_mask=attn_mask)

        attn_out = attn_outputs[0]

        out = inputs + attn_out
        ff_out = self.mlp(self.ln_2(out))
        out = out + ff_out

        outputs = [out] + attn_outputs[1:]
        return outputs


# based on Huggingface/transformers implementation
class GPT2(torch.nn.Module):
    """
    The GPT-2 base model.
    """

    def __init__(self, config):
        super().__init__()

        self.config = config

        self.output_hidden_states = \
            config.output_hidden_states

        self.output_attentions = config.output_attentions
        self.output_past = config.output_past

        self.wte = Embedding(
            config.vocab_size, config.n_embd)
        self.wpe = Embedding(
            config.n_positions, config.n_embd)

        self.drop = Dropout(config.embd_pdrop)

        self.h = torch.nn.ModuleList([
            Block(config.n_ctx, config, scale=True)
            for _ in range(config.n_layer)
        ])

        self.ln_f = LayerNorm(
            config.n_embd,
            eps=config.layer_norm_epsilon)

        self.init_weights()

    def expand_embeddings(self, size):
        """
        Increases the size of embedding by the given
        amount.
        """
        old_size, n_embd = self.wte.weight.size()

        new_wte = Embedding(old_size + size, n_embd)
        new_wte.to(self.wte.weight.device)

        self._init_weights(new_wte)

        new_wte.weight.data[:old_size, :] = \
            self.wte.weight.data

        self.wte = new_wte

    def init_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        Initialize the weights.
        """
        if isinstance(
                module, (Linear, Embedding, Conv1D)):
            module.weight.data.normal_(
                mean=0.0,
                std=self.config.initializer_range)

            if isinstance(
                module, (Linear, Conv1D)) and \
                    module.bias is not None:
                module.bias.data.zero_()

        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids, past=None,
                attn_mask=None, type_ids=None,
                position_ids=None):

        input_shape = input_ids.size()

        if type_ids is not None:
            type_ids = type_ids.view(
                -1, input_shape[-1])

        if position_ids is not None:
            position_ids = position_ids.view(
                -1, input_shape[-1])

        if past is None:
            past_length = 0
            past = [None] * len(self.h)

        else:
            past_length = past[0][0].size(-2)

        if position_ids is None:
            device = input_ids.device

            position_ids = torch.arange(
                past_length,
                input_shape[-1] + past_length,
                dtype=torch.long, device=device)

            position_ids = position_ids\
                .unsqueeze(0).view(-1, input_shape[-1])

        if attn_mask is not None:
            attn_mask = attn_mask.view(
                -1, input_shape[-1])

            attn_mask = attn_mask\
                .unsqueeze(1).unsqueeze(2)

        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)

        if type_ids is not None:
            type_embeds = self.wte(type_ids)
        else:
            type_embeds = 0

        hidden_states = inputs_embeds + \
            position_embeds + \
            type_embeds

        hidden_states = self.drop(hidden_states)

        output_shape = input_shape + \
            (hidden_states.size(-1), )

        presents = ()
        all_attentions = []
        all_hidden_states = ()
        for idx, (block, layer_past) in \
                enumerate(zip(self.h, past)):

            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + \
                    (hidden_states.view(*output_shape), )

            outputs = block(
                hidden_states,
                past=layer_past,
                attn_mask=attn_mask)

            hidden_states, present = outputs[:2]
            if self.output_past:
                presents = presents + (present,)

            if self.output_attentions:
                all_attentions.append(outputs[2])

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(*output_shape)

        hidden_states = linear(
            hidden_states, self.wte.weight)

        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + \
                (hidden_states, )

        outputs = (hidden_states, )

        if self.output_past:
            outputs = outputs + (presents, )

        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states, )

        if self.output_attentions:
            attention_output_shape = input_shape[:-1] + \
                (-1,) + all_attentions[0].shape[-2:]
            all_attentions = tuple(
                t.view(*attention_output_shape)
                for t in all_attentions)

            outputs = outputs + (all_attentions, )

        return outputs

