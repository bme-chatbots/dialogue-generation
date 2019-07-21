"""

@author:    Patrik Purgai
@copyright: Copyright 2019, chatbot
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

from beam import setup_beam_args, beam_search

from data import (
    ids2text, 
    text2ids, 
    get_special_indices)

from model import create_model, setup_model_args


DEVICE = 'cpu'


def setup_eval_args():
    """Sets up the arguments for evaluation."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_dir', 
        type=str,
        default=None,
        help='Path of the model file.')

    setup_beam_args(parser)
    setup_model_args(parser)

    return parser.parse_args()


@torch.no_grad()
def respond(text, model, fields, vocabs, indices, beam_size):
    """Translates the given text with beam search."""
    src_field, trg_field = fields
    ids = text2ids(text, src_field)
    preds, _ = beam_search(
        model=model, 
        inputs=ids, 
        indices=indices,
        beam_size=beam_size, 
        device=DEVICE)
    output = ids2text(preds, trg_field)

    return output


def main():
    args = setup_eval_args()
    state_dict = torch.load(join(args.model_dir, 'model.pt'), 
        map_location=DEVICE)
    fields = torch.load(join(args.model_dir, 'fields.pt'), 
        map_location=DEVICE)

    src_field, trg_field = fields['src'], fields['trg']

    fields = src_field, trg_field
    vocabs = src_field.vocab, trg_field.vocab
    indices = get_special_indices(vocabs)
    
    model = create_model(args, vocabs, indices, DEVICE)
    model.load_state_dict(state_dict['model'])
    model.eval()

    print('Type a sentence. CTRL + C to escape.')

    while True:
        try:
            print()
            text = input()
            output = respond(
                text=text, model=model,
                fields=fields, vocabs=vocabs,
                indices=indices, beam_size=args.beam_size)
            print('{}'.format(output))
            print()
            
        except KeyboardInterrupt:
            break
    

if __name__ == '__main__':
    main()
