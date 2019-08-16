"""
@author:    Patrik Purgai
@copyright: Copyright 2019, dialogue-generation
@license:   MIT
@email:     purgai.patrik@gmail.com
@date:      2019.06.25.
"""

# distutils: language=c++

cimport numpy as np
import numpy as np

import cython

from libcpp.vector cimport vector

from libc.stdint cimport int32_t


def xlnet_padded_collate(vector[vector[vector[int]]] example):
    """
    Collate function for merging a list of examples into
    a batch tensor.
    """
    # unzipping the examples lists
    cdef vector[vector[int]] input_ids, token_type_ids, targets
    cdef Py_ssize_t btc_idx, btc_size = example.size()

    for btc_idx in range(btc_size):
        input_ids.push_back(example[btc_idx][0])
        token_type_ids.push_back(example[btc_idx][1])
        targets.push_back(example[btc_idx][2])
    
    # input_tensors contains the attention_mask and input_ids
    input_tensors = create_xlnet_batch(input_ids, token_type_ids)
    target_tensor = batchify(targets, -1)

    return input_tensors, target_tensor


def gpt2_padded_collate(vector[vector[vector[int]]] example):
    """
    Collate function for merging a list of examples into
    a batch tensor.
    """
    # unzipping the examples lists
    cdef vector[vector[int]] input_ids, token_type_ids, targets
    cdef Py_ssize_t btc_idx, btc_size = example.size()
    cdef vector[int] target_lengths

    for btc_idx in range(btc_size):
        input_ids.push_back(example[btc_idx][0])
        token_type_ids.push_back(example[btc_idx][1])
        targets.push_back(example[btc_idx][2])
        target_lengths.push_back(example[btc_idx][2].size())
    
    # input_tensors contains the attention_mask and input_ids
    cdef np.ndarray[np.int32_t, ndim=3] tensors = \
        create_gpt2_train_batch(
            input_ids, token_type_ids, target_lengths)

    cdef np.ndarray[np.int32_t, ndim=3] input_tensors = \
        tensors[:2, :, :-1]

    cdef np.ndarray[np.int32_t, ndim=2] target_tensor = \
        tensors[2, :, 1:]

    return input_tensors, target_tensor


COLLATE = {
    'xlnet': xlnet_padded_collate,
    'gpt2': gpt2_padded_collate
}


def prepare_xlnet_inputs(input_ids, token_type_ids):
    """
    Creates padded arrays from the provided inputs
    for the XLNet model during inference.
    """
    return create_xlnet_batch(input_ids, token_type_ids)


def prepare_gpt2_inputs(input_ids, token_type_ids):
    """
    Creates padded arrays from the provided inputs
    for the GPT2 model during inference.
    """
    return create_gpt2_eval_batch(input_ids, token_type_ids)


PERPARE = {
    'xlnet': prepare_xlnet_inputs,
    'gpt2': prepare_gpt2_inputs
}


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef batchify(vector[vector[int]] ids, const int32_t pad):
    """
    Creates a batch from a list of utterances.
    """
    cdef Py_ssize_t btc_size = ids.size()
    cdef Py_ssize_t btc_idx
    cdef vector[int] utr
    cdef int utr_len, max_len = 0

    # computing the sequence length for the batch
    for btc_idx in range(btc_size):
        utr = ids[btc_idx]
        utr_len = utr.size()
        if utr_len > max_len:
            max_len = utr_len

    cdef np.ndarray[np.int32_t, ndim=2] tensor = \
        np.empty([btc_size, max_len], dtype=np.int32)

    cdef int32_t [:, :] tensor_view = tensor

    cdef Py_ssize_t utr_size, idx, diff_size, \
        diff_idx, pad_idx

    with nogil:
        for btc_idx in range(btc_size):
            utr_size = ids[btc_idx].size()
            diff_size = max_len - utr_size

            for idx in range(utr_size):
                tensor_view[btc_idx, idx] = \
                    ids[btc_idx][idx]

            for diff_idx in range(diff_size):
                pad_idx = utr_size + diff_idx 
                # 5 is the hard coded pad idx
                tensor[btc_idx, pad_idx] = pad
    
    return tensor


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef create_xlnet_batch(
        vector[vector[int]] input_ids, 
        vector[vector[int]] token_type_ids):
    """
    Creates a batch from a list of utterances.
    """
    cdef Py_ssize_t btc_size = input_ids.size()
    cdef Py_ssize_t btc_idx
    cdef vector[int] utr
    cdef int utr_len, max_len = 0

    # computing the sequence length for the batch
    for btc_idx in range(btc_size):
        utr = input_ids[btc_idx]
        utr_len = utr.size()
        if utr_len > max_len:
            max_len = utr_len

    cdef np.ndarray[np.int32_t, ndim=2] input_id_tensor = \
        np.empty([btc_size, max_len], dtype=np.int32)

    cdef int32_t [:, :] input_id_view = input_id_tensor

    # input_ids contain the history (the source utterance as well)
    # and the first n number of target tokens (the `labels` are
    # the n + 1 th tokens in the target utterance) the id `6` is
    # the mask token which will be predicted by the model and `5`
    # is the padding token

    # [
    #  [    1, 32001, 11781, ...,   32000,    9,    35,     6],
    #   ...
    #  [    1, 32001,    35, ...,   32000,   39,     6,     5]
    # ]

    cdef np.ndarray[np.int32_t, ndim=2] token_type_id_tensor = \
        np.empty([btc_size, max_len], dtype=np.int32)

    cdef int32_t [:, :] token_type_id_view = token_type_id_tensor

    # token_type_ids has the same size as input_ids and its
    # purpose is to mark the role_id of each token in the input
    # in this case the role is the id of the person that the
    # utterance comes from (speaker1 or speaker2) but this is
    # a subject to change in the near future

    # [
    #  [32001, 32001, 32001, ..., 32000, 32000, 32000, 32000],
    #   ...
    #  [32001, 32001, 32001, ..., 32000, 32000, 32000, 5]
    # ]

    cdef np.ndarray[np.int32_t, ndim=2] attn_mask = \
        np.ones([btc_size, max_len], dtype=np.int32)

    cdef int32_t [:, :] attn_mask_view = attn_mask

    # attn_mask also has the same dimensions as input_ids and
    # its purpose is to be a mask over the non-padding token ids
    # in the input, which is used mainly for attention computation
    # the mask is 1 at non-padding tokens and 0 at padding tokens

    # [
    #  [1, 1, 1, ..., 1, 1, 1, 1],
    #  [1, 1, 1, ..., 1, 1, 1, 0],
    # ]

    cdef np.ndarray[np.int32_t, ndim=3] perm_mask = \
        np.zeros([btc_size, max_len, max_len], dtype=np.int32)

    cdef int32_t [:, :, :] perm_mask_view = perm_mask

    # the value of this tensor at [btc_idx, i, j] marks whether
    # the value at the i index of the sequence should attend to
    # the value at the j index of the sequence in terms of
    # attention computation thus it is 1 everywhere except at
    # mask and pad token locations

    # [
    #  [
    #   [0, 0, 0, ..., 0, 0, 0, 1],
    #   [0, 0, 0, ..., 0, 0, 0, 1],
    #   [0, 0, 0, ..., 0, 0, 0, 1],
    #    ...,
    #   [0, 0, 0, ..., 0, 0, 0, 1],
    #   [0, 0, 0, ..., 0, 0, 0, 1],
    #   [0, 0, 0, ..., 0, 0, 0, 1]
    #  ],
    #  [
    #   [0, 0, 0, ..., 0, 0, 1, 1],
    #   [0, 0, 0, ..., 0, 0, 1, 1],
    #   [0, 0, 0, ..., 0, 0, 1, 1],
    #    ...,
    #   [0, 0, 0, ..., 0, 0, 1, 1],
    #   [0, 0, 0, ..., 0, 0, 1, 1],
    #   [0, 0, 0, ..., 0, 0, 1, 1]
    #  ],
    # ]

    cdef np.ndarray[np.int32_t, ndim=3] target_map = \
        np.zeros([btc_size, 1, max_len], dtype=np.int32)

    cdef int32_t [:, :, :] target_map_view = target_map

    # this tensor marks the location and order of the
    # expected output or target location
    # the size of the 2nd dimension of this tensor is the 
    # number of outputs which is 1 in this case and the 
    # 3rd dim is max_length which is 0 everywhere except
    # at the location of the mask token

    # [
    #  [
    #   [0, 0, 0, ..., 0, 0, 0, 1]
    #  ],
    #  [
    #   [0, 0, 0, ..., 0, 0, 1, 0]
    #  ]
    # ]

    cdef Py_ssize_t utr_size, idx, diff_size, \
        diff_idx, pad_idx

    with nogil:
        for btc_idx in range(btc_size):
            utr_size = input_ids[btc_idx].size()
            diff_size = max_len - utr_size

            target_map_view[btc_idx, 0, utr_size - 1] = 1

            for idx in range(utr_size):
                input_id_view[btc_idx, idx] = \
                    input_ids[btc_idx][idx]
                token_type_id_view[btc_idx, idx] = \
                    token_type_ids[btc_idx][idx]

            for idx in range(max_len):
                perm_mask_view[btc_idx, idx, utr_size - 1] = 1

            for diff_idx in range(diff_size):
                pad_idx = utr_size + diff_idx 
                # 5 is the hard coded pad idx
                input_id_view[btc_idx, pad_idx] = 5
                token_type_id_view[btc_idx, pad_idx] = 5
                attn_mask_view[btc_idx, pad_idx] = 0

                for idx in range(max_len):
                    perm_mask_view[btc_idx, idx, pad_idx] = 1
    
    return input_id_tensor, token_type_id_tensor, \
        attn_mask, perm_mask, target_map


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef create_gpt2_train_batch(
        vector[vector[int]] input_ids, 
        vector[vector[int]] token_type_ids,
        vector[int] target_lengths):
    """
    Creates a batch from a list of utterances.
    """
    cdef Py_ssize_t btc_size = input_ids.size()
    cdef Py_ssize_t btc_idx
    cdef vector[int] utr
    cdef int utr_len, max_len = 0

    # computing the sequence length for the batch
    for btc_idx in range(btc_size):
        utr = input_ids[btc_idx]
        utr_len = utr.size()
        if utr_len > max_len:
            max_len = utr_len

    cdef np.ndarray[np.int32_t, ndim=3] input_tensor = \
        np.empty([3, btc_size, max_len], dtype=np.int32)

    cdef int32_t [:, :, :] input_view = input_tensor

    cdef Py_ssize_t utr_size, idx, diff_size, \
        trg_size, diff_idx, pad_idx

    with nogil:
        for btc_idx in range(btc_size):
            utr_size = input_ids[btc_idx].size()
            trg_size = target_lengths[btc_idx]
            diff_size = max_len - utr_size

            for idx in range(utr_size):
                input_view[0, btc_idx, idx] = \
                    input_ids[btc_idx][idx]
                input_view[1, btc_idx, idx] = \
                    token_type_ids[btc_idx][idx]

                if idx < utr_size - trg_size:
                    input_view[2, btc_idx, idx] = 50261
                else:
                    input_view[2, btc_idx, idx] = \
                        input_ids[btc_idx][idx]

            for diff_idx in range(diff_size):
                pad_idx = utr_size + diff_idx 
                # -1 is the hard coded pad idx
                input_view[0, btc_idx, pad_idx] = 50261
                input_view[1, btc_idx, pad_idx] = 50261
                input_view[2, btc_idx, pad_idx] = 50261
    
    return input_tensor


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef create_gpt2_eval_batch(
        vector[vector[int]] input_ids, 
        vector[vector[int]] token_type_ids):
    """
    Creates a batch from a list of utterances.
    """
    cdef Py_ssize_t btc_size = input_ids.size()
    cdef Py_ssize_t btc_idx
    cdef vector[int] utr
    cdef int utr_len, max_len = 0

    # computing the sequence length for the batch
    for btc_idx in range(btc_size):
        utr = input_ids[btc_idx]
        utr_len = utr.size()
        if utr_len > max_len:
            max_len = utr_len

    cdef np.ndarray[np.int32_t, ndim=3] input_tensor = \
        np.empty([2, btc_size, max_len], dtype=np.int32)

    cdef int32_t [:, :, :] input_view = input_tensor

    cdef Py_ssize_t utr_size, idx, diff_size, \
        trg_size, diff_idx, pad_idx

    with nogil:
        for btc_idx in range(btc_size):
            utr_size = input_ids[btc_idx].size()
            diff_size = max_len - utr_size

            for idx in range(utr_size):
                input_view[0, btc_idx, idx] = \
                    input_ids[btc_idx][idx]
                input_view[1, btc_idx, idx] = \
                    token_type_ids[btc_idx][idx]

            for diff_idx in range(diff_size):
                pad_idx = utr_size + diff_idx 
                # 50261 is the hard coded pad idx for
                # the GPT-2 model
                input_view[0, btc_idx, pad_idx] = 50261
                input_view[1, btc_idx, pad_idx] = 50261
    
    return input_tensor
