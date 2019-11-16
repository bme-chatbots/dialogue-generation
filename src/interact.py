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
PROJECT_PATH = join(abspath(dirname(__file__)), '..')
if PROJECT_PATH not in sys.path:
    sys.path.append(PROJECT_PATH)

from src.data import (
    setup_data_args,
    create_tokenizer,
    transform_dialog,
    RSP, SP1, SP2, HST)

from src.model import (
    create_model,
    setup_model_args)

from src.train import set_random_seed

from src.collate import PREPARE, COLLATE

from torch.nn.functional import softmax


def setup_interact_args():
    """
    Sets up the arguments for interaction.
    """
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group('interact')
    group.add_argument(
        '--model_file',
        type=str,
        default=None,
        help='Path to the file of the model.')
    group.add_argument(
        '--ckpt_name',
        type=str,
        default='last',
        choices=['last', 'best'],
        help='Name of the checkpoint to load.')
    group.add_argument(
        '--method',
        type=str,
        default='topk',
        choices=list(METHODS),
        help='Decoding method to use.')
    group.add_argument(
        '--no_cuda',
        action='store_true',
        default=torch.cuda.is_available(),
        help='Device for training.')
    group.add_argument(
        '--top_p',
        type=float,
        default=0.9,
        help='Top-p parameter for nucleus sampling.')
    group.add_argument(
        '--top_k',
        type=int,
        default=100,
        help='Top-k parameter for topk sampling.')
    group.add_argument(
        '--min_len',
        type=int,
        default=0,
        help='Minimum length of the response sentence.')
    group.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for interactive mode.')

    setup_data_args(parser)
    setup_model_args(parser)

    return parser.parse_args()


def decode(args, model, inputs, tokenizer, 
           select_fn, device):
    """
    Applies decoding given a model and inputs.
    """
    input_ids, token_type_ids = inputs

    batch_size = len(input_ids)

    if 'xlnet' in args.model:
        mask_id = tokenizer.convert_tokens_to_ids(
            tokenizer.mask_token)

    rsp_id, eos_id = \
        tokenizer.convert_tokens_to_ids([
            RSP, tokenizer.eos_token
        ])

    preds = [[] for _ in range(batch_size)]
    finished = set()

    for idx in range(args.max_len):
        curr_input_ids = [
            i + p for i, p in 
            zip(input_ids, preds)]            
        
        curr_token_type_ids = [
            t + [rsp_id] * len(p) for t, p in
            zip(token_type_ids, preds)]

        if 'xlnet' in args.model:
            for i in range(batch_size):
                curr_input_ids[i].append(mask_id)
                curr_token_type_ids[i].append(rsp_id)

        inputs = PREPARE[args.model](
            input_ids=curr_input_ids,
            token_type_ids=curr_token_type_ids)

        # the first value of the output tuple from
        # the model is the next token logits tensor
        logits = model(inputs)[0]

        # TODO find a better solution
        if 'gpt2' in args.model:
            logits = logits[:, -1]

        logits = logits.view(-1, logits.size(-1))

        force_eos_id = None if idx >= args.min_len \
            else eos_id
            
        logits = select_fn(args, logits, force_eos_id)

        probs = softmax(logits, dim=-1)
        pred = torch.multinomial(probs, 1)
        pred = pred.view(-1)

        # breaking after eos_id is seen
        for i, p in enumerate(pred.tolist()):
            if p == eos_id:
                finished.add(i)
            elif i not in finished:
                preds[i].append(p)

        if len(finished) == batch_size:
            break

    return preds


def select_topk(args, logits, force_eos_id=None):
    """
    Applies topk sampling decoding.
    """        
    if force_eos_id is not None:
        logits[:, force_eos_id] = float('-inf')

    indices_to_remove = logits < \
        torch.topk(logits, args.top_k, axis=-1)[0][
            ..., -1, None]
    logits[indices_to_remove] = float('-inf')

    return logits


def select_nucleus(args, logits, force_eos_id=None):
    """
    Applies nucleus decoding.
    """
    if force_eos_id is not None:
        logits[:, force_eos_id] = float('-inf')

    sorted_logits, sorted_indices = torch.sort(
        logits, descending=True)

    cumulative_probs = torch.cumsum(
        softmax(sorted_logits, dim=-1), dim=-1)

    sorted_indices_to_remove = \
        cumulative_probs > args.top_p

    sorted_indices_to_remove[..., 1:] = \
        sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    indices_to_remove = \
        sorted_indices[sorted_indices_to_remove]
    logits[indices_to_remove] = float('-inf')

    return logits


METHODS = {
    'nucleus': select_nucleus,
    'topk': select_topk
}


def main():
    args = setup_interact_args()

    args.distributed = False

    args.cuda = not args.no_cuda and \
        torch.cuda.is_available()

    if args.seed is not None:
        set_random_seed(args)

    device = torch.device(
        'cuda' if args.cuda else 'cpu')
    
    assert args.name is not None, \
        '`--name` must be given'

    model_dir = join(
        args.model_dir, args.model, args.name)

    model_path = args.model_file if \
        args.model_file else \
        join(model_dir, args.ckpt_name + '.pt')
    
    state_dict = torch.load(
        model_path, map_location=device)

    del state_dict['optimizer']

    tokenizer = create_tokenizer(args)

    vocab_size = len(tokenizer)

    model = create_model(args, model_dir, vocab_size)
    model = model.to(device)
    
    try:
        model.load_state_dict(state_dict.pop('model'))
        model.eval()
    except RuntimeError as e:
        print(
            'The provided checkpoint has mismatching '
            'weights in the parameter dict.')

        print(
            'WARNING: If the model was trained with '
            '`--grad_ckpt` you also have to provide '
            'this argument for this script.')

        sys.exit()

    print()
    print(tabulate(state_dict.items(), tablefmt='presto'))
    print()

    history = []

    select_fn = METHODS[args.method]

    special_ids = tokenizer.convert_tokens_to_ids([
        SP1, SP2, tokenizer.bos_token,
        tokenizer.eos_token, HST, RSP,
    ])

    @torch.no_grad()
    def respond(text):
        """
        Responds to the given text.
        """
        history.append(tokenizer.encode(text))

        inputs = transform_dialog(
            history[-args.max_hist:],
            special_ids=special_ids,
            max_len=args.max_len)

        input_ids, type_ids = inputs
        inputs = [[input_ids], [type_ids]]
        
        preds = decode(
            args=args, model=model,
            inputs=inputs, tokenizer=tokenizer,
            select_fn=select_fn, device=device)[0]

        history.append(preds)

        # last token is the end token
        return tokenizer.decode(preds)

    print('Type a sentence for response. ' +
          'CTRL + C to escape.')

    while True:
        try:
            print()
            text = input('User: ')
            print()
            output = respond(text)
            print('Bot: {}'.format(output))

        except KeyboardInterrupt:
            break


if __name__ == '__main__':
    main()
