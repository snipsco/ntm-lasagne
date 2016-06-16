import pytest

import theano
import theano.tensor as T
import numpy as np

import lasagne.nonlinearities
from lasagne.theano_extensions import padding


def test_content_addressing():
    from ntm.similarities import cosine_similarity
    beta_var, key_var, memory_var = T.tensor3s('beta', 'key', 'memory')

    beta_var = T.addbroadcast(beta_var, 2)
    betaK = beta_var * cosine_similarity(key_var, memory_var)
    w_c = lasagne.nonlinearities.softmax(betaK.reshape((16 * 4, 128)))
    w_c = w_c.reshape(betaK.shape)

    content_addressing_fn = theano.function([beta_var, key_var, memory_var], w_c)

    beta = np.random.rand(16, 4, 1)
    key = np.random.rand(16, 4, 20)
    memory = np.random.rand(16, 128, 20)

    weights = content_addressing_fn(beta, key, memory)
    weights_manual = np.zeros_like(weights)

    def softmax(x):
        y = np.exp(x.T - np.max(x, axis=1))
        z = y / np.sum(y, axis=0)
        return z.T

    betaK_manual = np.zeros((16, 4, 128))
    for i in range(16):
        for j in range(4):
            for k in range(128):
                betaK_manual[i, j, k] = beta[i, j, 0] * np.dot(key[i, j], \
                    memory[i, k]) / np.sqrt(np.sum(key[i, j] * key[i, j]) * \
                    np.sum(memory[i, k] * memory[i, k]) + 1e-6)
    for i in range(16):
        weights_manual[i] = softmax(betaK_manual[i])

    assert weights.shape == (16, 4, 128)
    assert np.allclose(np.sum(weights, axis=2), np.ones((16, 4)))
    assert np.allclose(weights, weights_manual)


def test_convolutional_shift():
    weights_var, shift_var = T.tensor3s('weights', 'shift')
    num_shifts = 3

    weights_reshaped = weights_var.reshape((16 * 4, 128))
    weights_reshaped = weights_reshaped.dimshuffle(0, 'x', 'x', 1)
    shift_reshaped = shift_var.reshape((16 * 4, num_shifts))
    shift_reshaped = shift_reshaped.dimshuffle(0, 'x', 'x', 1)
    pad = (num_shifts // 2, (num_shifts - 1) // 2)
    weights_padded = padding.pad(weights_reshaped, [pad], batch_ndim=3)
    convolution = T.nnet.conv2d(weights_padded, shift_reshaped,
        input_shape=(16 * 4, 1, 1, 128 + pad[0] + pad[1]),
        filter_shape=(16 * 4, 1, 1, num_shifts),
        subsample=(1, 1),
        border_mode='valid')
    w_tilde = convolution[T.arange(16 * 4), T.arange(16 * 4), 0, :]
    w_tilde = w_tilde.reshape((16, 4, 128))

    convolutional_shift_fn = theano.function([weights_var, shift_var], w_tilde)

    weights = np.random.rand(16, 4, 128)
    shift = np.random.rand(16, 4, 3)

    weight_tilde = convolutional_shift_fn(weights, shift)
    weight_tilde_manual = np.zeros_like(weight_tilde)

    for i in range(16):
        for j in range(4):
            for k in range(128):
                # Filters in T.nnet.conv2d are reversed
                if (k - 1) >= 0:
                    weight_tilde_manual[i, j, k] += shift[i, j, 2] * weights[i, j, k - 1]
                weight_tilde_manual[i, j, k] += shift[i, j, 1] * weights[i, j, k]
                if (k + 1) < 128:
                    weight_tilde_manual[i, j, k] += shift[i, j, 0] * weights[i, j, k + 1]

    assert weight_tilde.shape == (16, 4, 128)
    assert np.allclose(weight_tilde, weight_tilde_manual)


def test_sharpening():
    weight_var, gamma_var = T.tensor3s('weight', 'gamma')

    gamma_var = T.addbroadcast(gamma_var, 2)
    w = T.pow(weight_var + 1e-6, gamma_var)
    w /= T.sum(w, axis=2).dimshuffle(0, 1, 'x')

    sharpening_fn = theano.function([weight_var, gamma_var], w)

    weights = np.random.rand(16, 4, 128)
    gamma = np.random.rand(16, 4, 1)

    weight_t = sharpening_fn(weights, gamma)
    weight_t_manual = np.zeros_like(weight_t)

    for i in range(16):
        for j in range(4):
            for k in range(128):
                weight_t_manual[i, j, k] = np.power(weights[i, j, k] + 1e-6, gamma[i, j])
            weight_t_manual[i, j] /= np.sum(weight_t_manual[i, j])

    assert weight_t.shape == (16, 4, 128)
    assert np.allclose(weight_t, weight_t_manual)