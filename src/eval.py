"""

@author:    Patrik Purgai
@copyright: Copyright 2019, supervised-translation
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
    merge_history)

from model import (
    create_model,
    decode_greedy,
    setup_model_args)


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

    setup_data_args(parser)
    setup_model_args(parser)

    return parser.parse_args()


def main():
    args = setup_eval_args()
    device = torch.device('cuda' if args.cuda else 'cpu')

    state_dict = torch.load(
        join(args.model_dir, 'model.pt'),
        map_location=device)

    _, tokenizer = create_dataset(args, device)

    vocab_size = len(tokenizer)

    model = create_model(args, vocab_size, device)
    model.load_state_dict(state_dict['model'])
    model.eval()

    history = []

    def prepare_inputs():
        """
        Merges the history into a single example.
        """
        return merge_history(history[:args.max_history])

    @torch.no_grad()
    def respond(text):
        """
        Responds to the given text.
        """
        history.append(tokenizer.encode(text))
        inputs = prepare_inputs()
        _, preds = decode_greedy(model, inputs)
        history.append(preds)

        return tokenizer.decode(preds)

    print('Type a sentence to translate. ' + \
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
