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


cdef inline int int_max(int a, int b) nogil:
    """
    Simple max function.
    """
    return a if a >= b else b


cdef inline int max_len(vector[vector[int]] mat) nogil:
    """
    Finds the length of the longest element.
    """
    cdef Py_ssize_t idx, max_size = 0, size = mat.size()

    for idx in range(size):
        max_size = int_max(mat[idx].size(), max_size)

    return max_size


@cython.boundscheck(False)
@cython.nonecheck(False)
def xlnet_padded_collate(vector[vector[vector[int]]] examples):
    """
    Collate function for merging a list of examples into
    a batch tensor.
    """
    cdef vector[vector[int]] input_ids, token_type_ids, target_ids
    cdef Py_ssize_t btc_idx, btc_size = examples.size()

    for btc_idx in range(btc_size):
        input_ids.push_back(examples[btc_idx][0])
        token_type_ids.push_back(examples[btc_idx][1])
        target_ids.push_back(examples[btc_idx][2])
    
    # input_tensors contains the attention_mask and input_ids
    input_tensors, target_tensor = create_xlnet_train_batch(
        input_ids, token_type_ids, target_ids)

    return input_tensors, target_tensor


@cython.boundscheck(False)
@cython.nonecheck(False)
def gpt2_padded_collate(vector[vector[vector[int]]] examples):
    """
    Collate function for merging a list of examples into
    a batch tensor.
    """
    cdef vector[vector[int]] input_ids, token_type_ids
    cdef Py_ssize_t btc_idx, btc_size = examples.size()
    cdef vector[int] target_lengths

    for btc_idx in range(btc_size):
        input_ids.push_back(examples[btc_idx][0])
        token_type_ids.push_back(examples[btc_idx][1])
        target_lengths.push_back(examples[btc_idx][2].size())
    
    # `tensor` contains the input tensors for the model
    cdef np.ndarray[np.int32_t, ndim=3] tensors = \
        create_gpt2_train_batch(
            input_ids, token_type_ids, target_lengths)

    # the inputs of the model are the first N elements
    # of the sequence thus the id_tensor is shifted backward
    cdef np.ndarray[np.int32_t, ndim=2] input_id_tensor = \
        tensors[0, :, :-1]

    cdef np.ndarray[np.int32_t, ndim=2] token_type_id_tensor = \
        tensors[1, :, :-1]

    # the target tensor contains the target outputs for
    # the sequence thus the values are shifted forward
    cdef np.ndarray[np.int32_t, ndim=2] target_tensor = \
        tensors[2, :, 1:]

    return (input_id_tensor, token_type_id_tensor), target_tensor


COLLATE = {
    'xlnet-base-cased':     xlnet_padded_collate,
    'xlnet-large-cased':    xlnet_padded_collate,
    'distilgpt2':           gpt2_padded_collate,
    'gpt2':                 gpt2_padded_collate,
    'gpt2-medium':          gpt2_padded_collate,
    'gpt2-large':           gpt2_padded_collate,
    'gpt2-xl':              gpt2_padded_collate
}


def prepare_xlnet_inputs(input_ids, token_type_ids):
    """
    Creates padded arrays from the provided inputs
    for the XLNet model during inference.
    """
    return create_xlnet_eval_batch(input_ids, token_type_ids)


def prepare_gpt2_inputs(input_ids, token_type_ids):
    """
    Creates padded arrays from the provided inputs
    for the GPT2 model during inference.
    """
    return create_gpt2_eval_batch(input_ids, token_type_ids)


PREPARE = {
    'xlnet-base-cased':     prepare_xlnet_inputs,
    'xlnet-large-cased':    prepare_xlnet_inputs,
    'distilgpt2':           prepare_gpt2_inputs,
    'gpt2':                 prepare_gpt2_inputs,
    'gpt2-medium':          prepare_gpt2_inputs,
    'gpt2-large':           prepare_gpt2_inputs,
    'gpt2-xl':              prepare_gpt2_inputs
}


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef create_xlnet_train_batch(
        vector[vector[int]] input_ids, 
        vector[vector[int]] token_type_ids,
        vector[vector[int]] target_ids):
    """
    Creates a batch from a list of utterances.
    """
    cdef Py_ssize_t btc_size = input_ids.size()
    cdef Py_ssize_t btc_idx
    cdef vector[int] utr
    cdef int trg_size, seq_size

    cdef int max_seq = max_len(input_ids)
    cdef int max_trg = max_len(target_ids)

    cdef np.ndarray[np.int32_t, ndim=2] input_id_tensor = \
        np.empty([btc_size, max_seq], dtype=np.int32)
    cdef int32_t [:, :] input_id_view = input_id_tensor

    cdef np.ndarray[np.int32_t, ndim=2] token_type_id_tensor = \
        np.empty([btc_size, max_seq], dtype=np.int32)
    cdef int32_t [:, :] token_type_id_view = token_type_id_tensor

    cdef np.ndarray[np.int32_t, ndim=2] attn_mask = \
        np.ones([btc_size, max_seq], dtype=np.int32)
    cdef int32_t [:, :] attn_mask_view = attn_mask

    cdef np.ndarray[np.int32_t, ndim=3] perm_mask = \
        np.zeros([btc_size, max_seq, max_seq], dtype=np.int32)
    cdef int32_t [:, :, :] perm_mask_view = perm_mask

    cdef np.ndarray[np.int32_t, ndim=3] target_map = \
        np.zeros([btc_size, max_trg, max_seq], dtype=np.int32)
    cdef int32_t [:, :, :] target_map_view = target_map

    cdef np.ndarray[np.int32_t, ndim=2] target_tensor = \
        np.zeros([btc_size, max_trg], dtype=np.int32)
    cdef int32_t [:, :] target_view = target_tensor

    cdef Py_ssize_t utr_size, idx, diff_size, \
        trg_diff_size, diff_idx, pad_idx, utr_idx, trg_idx, \
        mask_size

    with nogil:
        for btc_idx in range(btc_size):
            utr_size = input_ids[btc_idx].size()

            trg_size = target_ids[btc_idx].size()
            trg_diff_size = max_trg - trg_size

            seq_size = utr_size - trg_size
            diff_size = max_seq - utr_size

            for idx in range(trg_size):
                target_view[btc_idx, idx] = \
                target_ids[btc_idx][idx]

            for diff_idx in range(trg_diff_size):
                pad_idx = trg_size + diff_idx
                target_view[btc_idx, pad_idx] = 5

            for idx in range(utr_size):
                input_id_view[btc_idx, idx] = \
                    input_ids[btc_idx][idx]
                token_type_id_view[btc_idx, idx] = \
                    token_type_ids[btc_idx][idx]

            for utr_idx in range(utr_size):
                mask_size = utr_idx + 1 - seq_size
                mask_size = int_max(mask_size, 0)
                mask_size = utr_size - seq_size - mask_size
                for idx in range(mask_size + diff_size + 1):
                    perm_mask_view[
                        btc_idx, utr_idx, 
                        max_seq - idx - 1] = 1

            for trg_idx in range(trg_size):
                target_map_view[
                    btc_idx, trg_idx, seq_size + trg_idx] = 1

            for diff_idx in range(diff_size):
                pad_idx = utr_size + diff_idx 
                # 5 is the hard coded pad idx
                input_id_view[btc_idx, pad_idx] = 5
                token_type_id_view[btc_idx, pad_idx] = 5
                attn_mask_view[btc_idx, pad_idx] = 0

                for idx in range(max_seq):
                    perm_mask_view[btc_idx, pad_idx, idx] = 1
    
    return (input_id_tensor, token_type_id_tensor,
        attn_mask, perm_mask, target_map), target_tensor


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef create_xlnet_eval_batch(
        vector[vector[int]] input_ids, 
        vector[vector[int]] token_type_ids):
    """
    Creates a batch from a list of utterances.
    """
    cdef Py_ssize_t btc_size = input_ids.size()
    cdef Py_ssize_t btc_idx
    cdef int max_seq = max_len(input_ids)

    cdef np.ndarray[np.int32_t, ndim=2] input_id_tensor = \
        np.empty([btc_size, max_seq], dtype=np.int32)
    cdef int32_t [:, :] input_id_view = input_id_tensor

    cdef np.ndarray[np.int32_t, ndim=2] token_type_id_tensor = \
        np.empty([btc_size, max_seq], dtype=np.int32)
    cdef int32_t [:, :] token_type_id_view = token_type_id_tensor

    cdef np.ndarray[np.int32_t, ndim=2] attn_mask = \
        np.ones([btc_size, max_seq], dtype=np.int32)
    cdef int32_t [:, :] attn_mask_view = attn_mask

    cdef np.ndarray[np.int32_t, ndim=3] perm_mask = \
        np.zeros([btc_size, max_seq, max_seq], dtype=np.int32)
    cdef int32_t [:, :, :] perm_mask_view = perm_mask

    cdef np.ndarray[np.int32_t, ndim=3] target_map = \
        np.zeros([btc_size, 1, max_seq], dtype=np.int32)
    cdef int32_t [:, :, :] target_map_view = target_map

    cdef Py_ssize_t utr_size, idx, diff_size, \
        diff_idx, pad_idx

    with nogil:
        for btc_idx in range(btc_size):
            utr_size = input_ids[btc_idx].size()
            diff_size = max_seq - utr_size

            target_map_view[btc_idx, 0, utr_size - 1] = 1

            for idx in range(utr_size):
                input_id_view[btc_idx, idx] = \
                    input_ids[btc_idx][idx]
                token_type_id_view[btc_idx, idx] = \
                    token_type_ids[btc_idx][idx]

            for idx in range(max_seq):
                perm_mask_view[btc_idx, idx, utr_size - 1] = 1

            for diff_idx in range(diff_size):
                pad_idx = utr_size + diff_idx 
                # 5 is the hard coded pad idx
                input_id_view[btc_idx, pad_idx] = 5
                token_type_id_view[btc_idx, pad_idx] = 5
                attn_mask_view[btc_idx, pad_idx] = 0

                for idx in range(max_seq):
                    perm_mask_view[btc_idx, idx, pad_idx] = 1
    
    return input_id_tensor, token_type_id_tensor, \
        attn_mask, perm_mask, target_map



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef create_gpt2_train_batch(
        vector[vector[int]] input_ids, 
        vector[vector[int]] token_type_ids,
        vector[int] target_lens):
    """
    Creates a batch from a list of utterances.
    """
    cdef Py_ssize_t btc_size = input_ids.size()
    cdef Py_ssize_t btc_idx
    cdef vector[int] utr

    cdef int max_seq = max_len(input_ids)

    cdef np.ndarray[np.int32_t, ndim=3] input_tensor = \
        np.empty([3, btc_size, max_seq], dtype=np.int32)

    cdef int32_t [:, :, :] input_view = input_tensor

    cdef Py_ssize_t utr_size, idx, diff_size, \
        trg_size, diff_idx, pad_idx

    with nogil:
        for btc_idx in range(btc_size):
            utr_size = input_ids[btc_idx].size()
            trg_size = target_lens[btc_idx]
            diff_size = max_seq - utr_size

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
                # 50261 is the hard coded pad idx for
                # the GPT-2 model
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

    cdef int max_seq = max_len(input_ids)

    cdef np.ndarray[np.int32_t, ndim=3] input_tensor = \
        np.empty([2, btc_size, max_seq], dtype=np.int32)

    cdef int32_t [:, :, :] input_view = input_tensor

    cdef Py_ssize_t utr_size, idx, diff_size, \
        diff_idx, pad_idx

    with nogil:
        for btc_idx in range(btc_size):
            utr_size = input_ids[btc_idx].size()
            diff_size = max_seq - utr_size

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
