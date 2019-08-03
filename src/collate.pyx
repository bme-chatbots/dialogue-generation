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


def padded_collate(vector[vector[vector[int]]] example):
    """
    Collate function for merging a list of examples into
    a batch tensor.
    """
    # unzipping the examples lists
    cdef vector[vector[int]] input_ids, token_type_ids
    cdef vector[int] labels
    cdef Py_ssize_t btc_idx, btc_size = example.size()

    for btc_idx in range(btc_size):
        input_ids.push_back(example[btc_idx][0])
        token_type_ids.push_back(example[btc_idx][1])
        labels.push_back(example[btc_idx][2][0])
    
    # input_tensors contains the attention_mask and input_ids
    input_tensors = batchify(input_ids, token_type_ids)

    return input_tensors, labels


def pad_inputs(input_ids, token_type_ids):
    """
    Creates padded arrays from the provided inputs
    for the XLNet model during inference.
    """
    return batchify(input_ids, token_type_ids)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef batchify(
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

    # inputs contains the input_ids and attention_mask
    # inputs[0] -> input_ids inputs[1] -> attention_mask
    cdef np.ndarray[np.int32_t, ndim=2] input_id_tensor = \
        np.empty([btc_size, max_len], dtype=np.int32)

    cdef np.ndarray[np.int32_t, ndim=2] token_type_id_tensor = \
        np.empty([btc_size, max_len], dtype=np.int32)

    cdef np.ndarray[np.int32_t, ndim=2] attn_mask = \
        np.ones([btc_size, max_len], dtype=np.int32)

    cdef np.ndarray[np.int32_t, ndim=3] perm_mask = \
        np.zeros([btc_size, max_len, max_len], dtype=np.int32)

    cdef np.ndarray[np.int32_t, ndim=3] target_map = \
        np.zeros([btc_size, 1, max_len], dtype=np.int32)

    cdef Py_ssize_t utr_size, idx, diff_size, \
        diff_idx, pad_idx

    # attention_mask values are according to 
    # pytorch-transformers 1 -> UNMASKED and 0 -> MASKED
    for btc_idx in range(btc_size):
        utr_size = input_ids[btc_idx].size()
        diff_size = max_len - utr_size

        target_map[btc_idx, 0, utr_size - 1] = 1

        for idx in range(utr_size):
            # input_id
            input_id_tensor[btc_idx, idx] = \
                input_ids[btc_idx][idx]
            token_type_id_tensor[btc_idx, idx] = \
                token_type_ids[btc_idx][idx]

        for idx in range(max_len):
            perm_mask[btc_idx, idx, utr_size - 1] = 1

        for diff_idx in range(diff_size):
            pad_idx = utr_size + diff_idx 
            # 5 is the hard coded pad idx
            input_id_tensor[btc_idx, pad_idx] = 5
            token_type_id_tensor[btc_idx, pad_idx] = 5
            attn_mask[btc_idx, pad_idx] = 0
 
    return input_id_tensor, token_type_id_tensor, \
        attn_mask, perm_mask, target_map
