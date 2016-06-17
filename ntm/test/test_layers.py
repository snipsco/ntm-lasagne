import pytest

import theano
import theano.tensor as T
import numpy as np

from lasagne.layers import InputLayer, ReshapeLayer, DenseLayer
from lasagne.layers import get_output, get_all_param_values, set_all_param_values
from ntm.layers import NTMLayer
from ntm.heads import WriteHead, ReadHead
from ntm.controllers import DenseController
from ntm.memory import Memory


def model(input_var, batch_size=1):
    l_input = InputLayer((batch_size, None, 8), input_var=input_var)
    batch_size_var, seqlen, _ = l_input.input_var.shape

    # Neural Turing Machine Layer
    memory = Memory((128, 20), name='memory')
    controller = DenseController(l_input, memory_shape=(128, 20),
        num_units=100, num_reads=1, name='controller')
    heads = [
        WriteHead(controller, num_shifts=3, memory_shape=(128, 20), name='write'),
        ReadHead(controller, num_shifts=3, memory_shape=(128, 20), name='read')
    ]
    l_ntm = NTMLayer(l_input, memory=memory, controller=controller, heads=heads)

    # Output Layer
    l_output_reshape = ReshapeLayer(l_ntm, (-1, 100))
    l_output_dense = DenseLayer(l_output_reshape, num_units=8, name='dense')
    l_output = ReshapeLayer(l_output_dense, (batch_size_var if batch_size \
        is None else batch_size, seqlen, 8))

    return l_output


def test_batch_size():
    input_var01, input_var16 = T.tensor3s('input01', 'input16')
    l_output01 = model(input_var01, batch_size=1)
    l_output16 = model(input_var16, batch_size=16)

    # Share the parameters for both models
    params01 = get_all_param_values(l_output01)
    set_all_param_values(l_output16, params01)

    posterior_fn01 = theano.function([input_var01], get_output(l_output01))
    posterior_fn16 = theano.function([input_var16], get_output(l_output16))

    example_input = np.random.rand(16, 30, 8)
    example_output16 = posterior_fn16(example_input)
    example_output01 = np.zeros_like(example_output16)

    for i in range(16):
        example_output01[i] = posterior_fn01(example_input[i][np.newaxis, :, :])

    assert example_output16.shape == (16, 30, 8)
    assert np.allclose(example_output16, example_output01, atol=1e-3)


def test_batch_size_none():
    input_var = T.tensor3('input')
    l_output = model(input_var, batch_size=None)
    posterior_fn = theano.function([input_var], get_output(l_output))

    example_input = np.random.rand(16, 30, 8)
    example_output = posterior_fn(example_input)

    assert example_output.shape == (16, 30, 8)
