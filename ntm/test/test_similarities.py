import pytest

import theano
import theano.tensor as T
import numpy as np


def test_cosine_similarity():
    from ntm.similarities import cosine_similarity

    key_var = T.matrix('key')
    memory_var = T.tensor3('memory')
    cosine_similarity_fn = theano.function([key_var, memory_var], \
        cosine_similarity(key_var, memory_var, eps=1e-6))

    test_key = np.random.rand(16, 20)
    test_memory = np.random.rand(16, 128, 20)

    test_output = cosine_similarity_fn(test_key, test_memory)
    test_output_manual = np.zeros_like(test_output)

    for i in range(16):
            for k in range(128):
                test_output_manual[i, k] = np.dot(test_key[i], test_memory[i, k]) / \
                    np.sqrt(np.sum(test_key[i] * test_key[i]) * np.sum(test_memory[i, k] * \
                    test_memory[i, k]) + 1e-6)

    assert np.allclose(test_output, test_output_manual)
