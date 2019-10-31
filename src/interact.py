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

import argparse
import torch

from os.path import join

from src.data import (
    setup_data_args,
    create_dataset,
    transform_dialog,
    RSP, SP1, SP2, HST)

from src.model import (
    create_model,
    setup_model_args)

from src.collate import PREPARE

from torch.nn.functional import softmax


def setup_interact_args():
    """
    Sets up the arguments for evaluation.
    """
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group('interact')
    group.add_argument(
        '--model_file',
        type=str,
        default=None,
        help='Path to the file of the model.')
    group.add_argument(
        '--method',
        type=str,
        default='nucleus',
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

    setup_data_args(parser)
    setup_model_args(parser)

    return parser.parse_args()


def decode(args, model, inputs, tokenizer, 
           select_fn, device):
    """
    Applies decoding given a model and inputs.
    """
    input_ids, token_type_ids = inputs

    mask_id, rsp_id, eos_id = \
        tokenizer.convert_tokens_to_ids([
            tokenizer.mask_token,
            RSP,
            tokenizer.eos_token
        ])

    preds = []

    for _ in range(args.max_len):
        curr_input_ids = input_ids + preds

        curr_token_type_ids = token_type_ids + \
            [rsp_id] * len(preds)

        if 'xlnet' in args.model:
            curr_input_ids.append(mask_id)
            curr_token_type_ids.append(rsp_id)

        inputs = PREPARE[args.model](
            input_ids=[curr_input_ids],
            token_type_ids=[curr_token_type_ids])

        # the first value of the output tuple from
        # the model is the next token logits tensor
        logits = model(inputs)[0]

        # TODO find a better solution
        if 'gpt2' in args.model:
            logits = logits[0][-1]

        logits = logits.squeeze()
        logits = select_fn(args, logits)
        probs = softmax(logits, dim=-1)
        pred = torch.multinomial(probs, 1)

        preds.append(pred.item())

        # breaking after eos_id is seen
        if preds[-1] == eos_id:
            break

    else:
        preds.append(eos_id)

    return preds


def select_topk(args, logits):
    """
    Applies topk sampling decoding.
    """
    indices_to_remove = logits < \
        torch.topk(logits, args.top_k)[0][..., -1, None]
    logits[indices_to_remove] = float('-inf')

    return logits


def select_nucleus(args, logits):
    """
    Applies nucleus decoding.
    """
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
    'nucleus': select_topk,
    'topk': select_nucleus
}


def main():
    args = setup_interact_args()
    args.distributed = False
    args.cuda = not args.no_cuda and \
        torch.cuda.is_available()

    device = torch.device(
        'cuda' if args.cuda else 'cpu')
    
    assert args.name is not None, \
        '`--name` must be given'

    model_dir = join(
        args.model_dir, args.model, args.name)

    model_path = args.model_file if \
        args.model_file else \
        join(model_dir, 'model.pt')
    
    state_dict = torch.load(
        model_path,
        map_location=device)

    _, tokenizer, _ = create_dataset(
        args=args, master_process=True)

    vocab_size = len(tokenizer)

    model = create_model(args, model_dir, vocab_size)
    model = model.to(device)
    
    try:
        model.load_state_dict(state_dict['model'])
        model.eval()
    except KeyError as e:
        print(
            'The provided checkpoint has mismatching '
            'weights in the parameter dict.'
        )

        print(
            'WARNING: If the model was trained with '
            '`--grad_ckpt` you also have to provide '
            'this argument for this script.'
        )

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
        
        preds = decode(
            args=args, model=model,
            inputs=inputs, tokenizer=tokenizer,
            select_fn=select_fn, device=device)

        history.append(preds)

        # last token is the end token
        return tokenizer.decode(preds[:-1])

    print('Type a sentence to translate. ' +
          'CTRL + C to escape.')

    while True:
        try:
            print()
            text = input()
            output = respond(text)
            print(output)
            print()

        except KeyboardInterrupt:
            break


if __name__ == '__main__':
    main()
