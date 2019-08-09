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

from data import (
    setup_data_args,
    create_dataset,
    transform_history,
    RSP, SP1, SP2,
    HST, SRC)

from model import (
    create_model,
    setup_model_args)

from collate import (
    prepare_inputs)

from numpy import inf

from torch.nn.functional import (
    softmax)


def setup_eval_args():
    """
    Sets up the arguments for evaluation.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_dir',
        type=str,
        default=None,
        help='Path of the model file.')
    parser.add_argument(
        '--cuda',
        type=bool,
        default=False,
        help='Device for evaluation.')
    parser.add_argument(
        '--method',
        type=str,
        default='nucleus',
        help='Decoding method to use.')
    parser.add_argument(
        '--max_len',
        type=int,
        default=100,
        help='Maximum length of the decoded sequence.')
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

    setup_data_args(parser)
    setup_model_args(parser)

    return parser.parse_args()


def decode(args, model, inputs, tokenizer, select_fn,
           device):
    """
    Applies decoding given a model and inputs.
    """
    def convert_to_tensor(ids):
        """
        Convenience function for converting int32
        ndarray to torch int64.
        """
        return torch.as_tensor(ids).long().to(device)

    input_ids, token_type_ids = inputs

    mask_id, rsp_id, eos_id = \
        tokenizer.convert_tokens_to_ids([
            tokenizer.mask_token,
            RSP,
            tokenizer.eos_token
        ])

    preds = []

    for _ in range(args.max_len):
        curr_input_ids = [input_ids + \
            preds + [mask_id]]
        
        curr_token_type_ids = [token_type_ids + \
            [rsp_id] * (len(preds) + 1)]

        inputs = prepare_inputs(
            input_ids=curr_input_ids,
            token_type_ids=curr_token_type_ids)

        padded_input_ids, padded_token_type_ids, \
            attn_mask, perm_mask, target_map = [
                convert_to_tensor(m) for m in inputs]
        
        # the first value of the output tuple from
        # the model is the next token logits tensor
        logits = model(
            input_ids=padded_input_ids,
            token_type_ids=padded_token_type_ids,
            attention_mask=attn_mask,
            perm_mask=perm_mask,
            target_mapping=target_map.float())[0]

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
    logits[indices_to_remove] = -inf

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
    logits[indices_to_remove] = -inf

    return logits


METHODS = {
    'nucleus': select_topk,
    'topk': select_nucleus
}


def main():
    args = setup_eval_args()
    device = torch.device('cuda' if args.cuda else 'cpu')

    state_dict = torch.load(
        join(args.model_dir, 'model.pt'),
        map_location=device)

    _, tokenizer = create_dataset(
        args, device, distributed=False)

    vocab_size = len(tokenizer)

    model = create_model(args, vocab_size, device)
    model.load_state_dict(state_dict['model'])
    model.eval()

    history = []

    select_fn = METHODS[args.method]

    special_ids = tokenizer.convert_tokens_to_ids([
        tokenizer.bos_token,
        SP1, SP2, HST, SRC, RSP
    ])

    @torch.no_grad()
    def respond(text):
        """
        Responds to the given text.
        """
        history.append(tokenizer.encode(text))

        inputs = transform_history(
            history[:args.max_history],
            special_ids=special_ids)
        
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
