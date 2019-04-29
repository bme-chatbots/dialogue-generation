"""

@author:    Patrik Purgai
@copyright: Copyright 2019, nmt
@license:   MIT
@email:     purgai.patrik@gmail.com
@date:      2019.04.04.
"""

# pylint: disable=not-callable
# pylint: disable=no-member

import torch
import math

from torch.nn.functional import log_softmax
from collections import namedtuple


NEAR_INF = 1e20
NEAR_INF_FP16 = 65504


Hypothesis = namedtuple('Hypothesis', ['hyp_id', 'timestep', 'score'])


def setup_beam_args(parser):
    """
    Sets up the parameters for the beam search.
    """
    parser.add_argument(
        '--beam_size',
        type=int,
        default=3,
        help='Number of beam segments.')


def neginf(dtype):
    """
    Represents the negative infinity for the dtype.
    """
    if dtype is torch.float16:
        return -NEAR_INF_FP16
    else:
        return -NEAR_INF


def create_beams(*args, batch_size, **kwargs):
    """
    Creates a beam for each sample in a batch.
    """
    return [Beam(*args, **kwargs) for _ in range(batch_size)]


def select_hidden_states(hidden_states, indices):
    """
    Prepares the hidden states for the next step.
    """
    hidden_states = tuple(
        hs.index_select(1, indices) for hs in hidden_states)
    return hidden_states


def beam_search(model, inputs, indices, beam_size, device,
                min_len=3, max_len=50):
    """
    Applies beam search decoding on the provided inputs.
    Implementation is based on `facebookresearch ParlAI`.
    """
    batch_size = inputs.size(0)
    _, trg_pad_index, start_index, end_index = indices 

    encoder_outputs, hidden_state = model.encoder(inputs)

    # a beam is created for each element of the batch
    beams = create_beams(
        beam_size=beam_size, pad_index=trg_pad_index, 
        start_index=start_index, end_index=end_index,
        min_len=min_len, device=device, batch_size=batch_size)

    # the decoder has beam_size * batch_size inputs
    decoder_input = torch.tensor(start_index).to(device)
    decoder_input = decoder_input.expand(
        batch_size * beam_size, 1)

    indices = torch.arange(batch_size).to(device)
    indices = indices.unsqueeze(1).repeat(
        1, beam_size).view(-1)

    # each encoder output is copied beam_size times, 
    # making `encoder_outputs` of size
    # [batch_size * beam_size, seq_len, encoder_hidden_dim]
    encoder_outputs = encoder_outputs.index_select(0, indices)
    hidden_state = select_hidden_states(hidden_state, indices)
    
    for _ in range(max_len):
        if all(beam.finished for beam in beams):
            break

        logits, hidden_state = model.decoder(
            inputs=decoder_input, 
            encoder_outputs=encoder_outputs, 
            hidden_state=hidden_state)

        logits = logits[:, -1:, :]
        scores = log_softmax(logits, dim=2)
        scores = scores.view(batch_size, beam_size, -1)

        # each beam receives the corresponding score
        # output, to calculate the best candidates
        for index, beam in enumerate(beams):
            if not beam.finished:
                beam.step(scores[index])

        # prepares the indices, which select the hidden states
        # of the best scoring outputs
        indices = torch.cat([
            beam_size * i + b.hyp_ids[-1] 
            for i, b in enumerate(beams)
        ])

        hidden_state = select_hidden_states(hidden_state, indices)
        decoder_input = torch.index_select(decoder_input, 0, indices)

        prev_output = torch.cat([b.token_ids[-1] for b in beams])
        prev_output = prev_output.unsqueeze(-1)
        decoder_input = torch.cat([decoder_input, prev_output], dim=-1)

    top_preds, top_scores = [], []
    for beam in beams:
        preds, scores = beam.get_result()
        top_preds.append(preds)
        top_scores.append(scores)

    top_preds = torch.cat(top_preds, dim=-1)
    top_scores = torch.cat(top_scores, dim=-1)

    return top_preds, top_scores 


class Beam:

    def __init__(self, beam_size, pad_index, start_index,
                 end_index, min_len, device):
        """
        A beam that contains `beam_size` decoding candidates.
        Each beam operates on a single input sequence from the batch.
        """
        self.beam_size = beam_size
        self.pad_index = pad_index
        self.start_index = start_index
        self.end_index = end_index
        self.min_len = min_len
        self.device = device
        self.finished = False

        # scores of each candidate of the beam
        self.scores = torch.zeros(beam_size).to(device)
        # `self.scores` values for each time step
        self.all_scores = [torch.zeros(beam_size).to(device)]
        # ids of the previous candidates
        self.hyp_ids = []
        self.finished_hyps = []

        # output token ids of the hyps at each step
        self.token_ids = [
            torch.tensor(start_index)
                .expand(beam_size).to(device)]

    def step(self, scores):
        """
        Advances the beam a step forward by selecting the
        best candidates for the 
        """
        if self.finished:
            return

        # `scores` is the softmax output of the decoder
        vocab_size = scores.size(-1)
        current_length = len(self.all_scores) - 1

        if current_length < self.min_len:
            # penalizing end token before reaching minimum length
            for hyp_idx in range(scores.size(0)):
                scores[hyp_idx][self.end_index] = neginf(scores.dtype)

        if len(self.hyp_ids) == 0:
            # the scores for the first step is simply the
            # first candidate
            beam_scores = scores[0]

        else:
            prev_scores = self.scores.unsqueeze(1)
            prev_scores = prev_scores.expand_as(scores)
            beam_scores = scores + prev_scores

            for index in range(self.token_ids[-1].size(0)):
                if self.token_ids[-1][index] == self.end_index:
                    beam_scores[index] = neginf(scores.dtype)

        # flatten beam scores is vocab_size * beam_size
        flatten_beam_scores = beam_scores.view(-1)
        top_scores, top_idxs = torch.topk(
            flatten_beam_scores, self.beam_size, dim=-1)

        self.scores = top_scores
        self.all_scores.append(self.scores)

        # selecting the id of the best hyp at the current step
        self.hyp_ids.append(top_idxs / vocab_size)
        self.token_ids.append(top_idxs % vocab_size)

        for hyp_id in range(self.beam_size):
            if self.token_ids[-1][hyp_id] == self.end_index:
                time_step = len(self.token_ids)
                self.finished_hyps.append(
                    Hypothesis(
                        hyp_id=hyp_id, 
                        time_step=time_step,
                        score=self.scores[hyp_id] / (time_step + 1)))

                if len(self.finished_hyps) == self.beam_size:
                    self.finished = True
                    break


    def get_result(self):
        """"""
        best_hyp = sorted(self.finished_hyps, key=lambda x: x.score)[0]
        
        token_ids, scores = [], []
        for ts in range(best_hyp.timestep, 0, -1):
            pass
            
