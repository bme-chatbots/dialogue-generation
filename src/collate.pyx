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
    cdef vector[vector[int]] source, target
    cdef Py_ssize_t btc_idx, btc_size = example.size()

    for btc_idx in range(btc_size):
        source.push_back(example[btc_idx][0])
        target.push_back(example[btc_idx][1])
    
    cdef np.ndarray[np.int32_t, ndim=3] source_tensor = \
        batchify(source)

    cdef np.ndarray[np.int32_t, ndim=3] target_tensor = \
        batchify(target)

    return source_tensor, target_tensor


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef batchify(vector[vector[int]] ids):
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

    cdef np.ndarray[np.int32_t, ndim=3] tensor = \
        np.empty([2, btc_size, max_len], dtype=np.int32)

    cdef Py_ssize_t utr_size, tok_idx, diff_size, \
        diff_idx, pad_idx

    # attention mask is 0 at non-pad values
    # and 1 at pad values because it helps
    # the use of masked_fill() method

    for btc_idx in range(btc_size):
        utr_size = ids[btc_idx].size()
        diff_size = max_len - utr_size
        for tok_idx in range(utr_size):
            tensor[0, btc_idx, tok_idx] = \
                ids[btc_idx][tok_idx]
            tensor[1, btc_idx, tok_idx] = 0
        for diff_idx in range(diff_size):
            pad_idx = utr_size + diff_idx 
            # 0 is the hard coded pad idx
            tensor[0, btc_idx, pad_idx] = 0
            tensor[1, btc_idx, pad_idx] = 1

    return tensor
