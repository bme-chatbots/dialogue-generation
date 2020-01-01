"""
@author:    Patrik Purgai
@copyright: Copyright 2019, dialogue-generation
@license:   MIT
@email:     purgai.patrik@gmail.com
@date:      2019.04.04.
"""

# pylint: disable=import-error
# pylint: disable=no-member
# pylint: disable=no-name-in-module

import sys
import argparse
import torch

from tabulate import tabulate

from os.path import (
    join, abspath, dirname)

# HACK to enable launching with
# python src/train.py
PROJECT_DIR = join(abspath(dirname(__file__)), '..')
if PROJECT_DIR not in sys.path:
    sys.path.append(PROJECT_DIR)

from torch.nn.functional import softmax

from src.model import (
    GPT2,
    setup_model_args)

from src.utils import (
    SPECIAL_TOKENS,
    set_random_seed,
    load_tokenizer,
    load_config)


def setup_interact_args():
    """
    Sets up the arguments for interaction.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_file',
        required=True,
        help='Path to the file of the model.')
    parser.add_argument(
        '--decoding',
        type=str,
        default='nucleus',
        choices=list(METHODS),
        help='Decoding method to use.')
    parser.add_argument(
        '--no_cuda',
        action='store_true',
        default=torch.cuda.is_available(),
        help='Device for training.')
    parser.add_argument(
        '--top_p',
        type=float,
        default=0.9,
        help='Top-p parameter for nucleus sampling.')
    parser.add_argument(
        '--top_k',
        type=int,
        default=100,
        help='Top-k parameter for topk sampling.')
    parser.add_argument(
        '--min_len',
        type=int,
        default=0,
        help='Minimum length of the response sentence.')
    parser.add_argument(
        '--max_len',
        type=int,
        default=50,
        help='Maximum length of the response sentence.')
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for interactive mode.')

    setup_model_args(parser)

    return parser.parse_args()


def create_response_generator(
        args, model, tokenizer, specials, device):
    """
    Creates a generator decoder function.
    """
    def encode_inputs(text):
        """
        Creates the input_ids and type_ids.
        """
        ids = tokenizer.encode(text)
        ids.append(specials.EOS)

        input_ids = torch.tensor(ids).long().to(device)
        type_ids = torch.zeros_like(input_ids)

        return input_ids, type_ids, None

    def decode_output(ids):
        """
        Creates output text from the output ids.
        """
        return tokenizer.decode(
            token_id for token_id in ids
            if token_id not in vars(specials).values())

    @torch.no_grad()
    def generate_response():
        """
        Generator function that yields responses.
        """
        while True:
            text = yield

            inputs = encode_inputs(text)
            output_ids, past = greedy_decode(
                inputs=inputs, model=model,
                specials=specials, args=args)

            yield decode_output(output_ids)

    return generate_response


def greedy_decode(inputs, model, specials, args):
    """
    Applies greedy decoding to the input.
    """
    def append_to_tensor(tensor, value):
        """
        Concats a value to the end of the tensor.
        """
        device = tensor.device

        return torch.tensor(
            tensor.tolist() + [value]).long().to(device)

    select_fn = METHODS[args.decoding]

    outputs = []

    input_ids, type_ids, past = inputs

    for idx in range(args.max_len):
        hiddens, past = model(
            input_ids=input_ids,
            type_ids=type_ids,
            past=past)

        logits = hiddens[-1]

        logits = logits.view(-1, logits.size(-1))

        # forcing no eos id because the sequence is
        # not min_len long yet
        force_no_eos_id = None if idx >= \
            args.min_len else specials.EOS

        logits = select_fn(args, logits, force_no_eos_id)

        probs = softmax(logits, dim=-1)
        pred = torch.multinomial(probs, 1)
        pred = pred.view(-1).item()

        # breaking after eos_id is seed
        if pred == specials.EOS:
            break

        input_ids = append_to_tensor(input_ids, pred)
        type_ids = append_to_tensor(type_ids, 1)

        outputs.append(pred)

    return outputs, past


# implementation is from Huggingface/transformers repo
def select_topk(args, logits, force_no_eos_id=None):
    """
    Applies topk sampling decoding.
    """
    if force_no_eos_id is not None:
        logits[:, force_no_eos_id] = float('-inf')

    indices_to_remove = logits < \
        torch.topk(logits, args.top_k, axis=-1)[0][
            ..., -1, None]

    logits[indices_to_remove] = float('-inf')

    return logits


# implementation is from Huggingface/transformers repo
def select_nucleus(args, logits, force_no_eos_id=None):
    """
    Applies nucleus decoding.
    """
    if force_no_eos_id is not None:
        logits[:, force_no_eos_id] = float('-inf')

    sorted_logits, sorted_indices = torch.sort(
        logits, dim=-1, descending=True)

    cumulative_probs = torch.cumsum(
        softmax(sorted_logits, dim=-1), dim=-1)

    sorted_indices_to_remove = \
        cumulative_probs > args.top_p

    sorted_indices_to_remove[..., 1:] = \
        sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    for idx in range(logits.size(0)):
        indices_to_remove = \
            sorted_indices[idx, sorted_indices_to_remove[idx]]

        logits[idx, indices_to_remove] = float('-inf')

    return logits


METHODS = {
    'nucleus': select_nucleus,
    'topk': select_topk
}


def main():
    args = setup_interact_args()

    args.output_past = True
    args.distributed = False

    args.cuda = not args.no_cuda and \
        torch.cuda.is_available()

    if args.seed is not None: set_random_seed(args)

    device = torch.device(
        'cuda' if args.cuda else 'cpu')

    tokenizer, specials = load_tokenizer(args)
    args.vocab_size = len(tokenizer.encoder)

    config = load_config(args)
    config.vocab_size += len(SPECIAL_TOKENS)

    model = GPT2(config)

    model_state = torch.load(
        args.model_file, map_location=device)
    model.load_state_dict(model_state)

    model.eval()
    model = model.to(device)

    response_generator = create_response_generator(
        args=args,
        model=model,
        tokenizer=tokenizer,
        specials=specials,
        device=device)

    generate_response = response_generator()
    next(generate_response)

    print('Type a sentence for response. ' +
          'CTRL + C to escape.')

    while True:
        try:
            print()
            text = input('> ')
            print()
            output = generate_response.send(text)
            next(generate_response)
            print('{}'.format(output))

        except KeyboardInterrupt:
            break


if __name__ == '__main__':
    main()
