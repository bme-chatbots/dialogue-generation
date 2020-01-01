"""
@author:    Patrik Purgai
@copyright: Copyright 2019, gpt2-chatbot
@license:   MIT
@email:     purgai.patrik@gmail.com
@date:      2019.08.11.
"""

import torch
import json
import sys
import argparse

import tensorflow as tf
import numpy as np

assert '1.12' in tf.__version__, \
    'tensorflow:1.12.0 is required for testing'

from os.path import join, dirname, abspath

OPENAI_GPT2_DIR = join(
    abspath(dirname(__file__)), 'gpt-2', 'src')

if OPENAI_GPT2_DIR not in sys.path:
    sys.path.append(OPENAI_GPT2_DIR)


GPT2_CHATBOT_DIR = join(
    abspath(dirname(__file__)), '..')

if GPT2_CHATBOT_DIR not in sys.path:
    sys.path.insert(0, GPT2_CHATBOT_DIR)


from src.utils import (
    load_config,
    download_pretrained,
    load_weights,
    load_tokenizer,
    CACHE_DIR)

from src.model import GPT2, setup_model_args

from model import model, default_hparams

TEST_INPUT = 'This is a test input to test models.'


def main():
    parser = argparse.ArgumentParser()
    setup_model_args(parser)

    args = parser.parse_args()

    # setting up tokenizer and downloading model
    # these steps are mutual for openai and pytorch
    # gpt2 models 
    tokenizer, _ = load_tokenizer(args)
    args.vocab_size = len(tokenizer.encoder)

    download_pretrained(args.pretrained)
    config = load_config(args)

    # loading the pretrained tensorflow weights
    # as a state dict for the pytorch model
    state_dict = load_weights(args.pretrained)

    pytorch_model = GPT2(config)
    pytorch_model.load_state_dict(
        state_dict, strict=False)

    pytorch_model.eval()

    # loading the hparams object for openai GPT2
    model_dir = join(CACHE_DIR, args.pretrained)
    hparams_path = join(model_dir, 'hparams.json')

    hparams = default_hparams()
    with open(hparams_path, 'r') as fh:
        hparams.override_from_dict(json.load(fh))

    # encoding testing input ids
    input_ids = tokenizer.encode(TEST_INPUT)

    with tf.Session(graph=tf.Graph()) as sess:
        input_placeholder = tf.placeholder(
            tf.int32, [1, None])

        output = model(
            hparams=hparams,
            X=input_placeholder,
            past=None,
            reuse=tf.AUTO_REUSE)

        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(model_dir)
        saver.restore(sess, ckpt)

        out = sess.run(output, feed_dict={
            input_placeholder: [input_ids]
        })

        tf_output = out['logits'][0]

    with torch.no_grad():
        out = pytorch_model(
            torch.tensor(input_ids).long())[0]

        pt_output = torch.matmul(
            out,
            pytorch_model.wte.weight.t()
        ).numpy()

    print(np.allclose(pt_output, tf_output))


if __name__ == '__main__':
    main()

