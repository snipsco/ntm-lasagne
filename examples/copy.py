import theano
import theano.tensor as T
import numpy as np
import random

import matplotlib.pyplot as plt

from lasagne.layers import InputLayer, DenseLayer, ReshapeLayer
import lasagne.layers
import lasagne.nonlinearities
import lasagne.updates
import lasagne.objectives

from ntm.ntm import NTMLayer
from ntm.memory import Memory
from ntm.controllers import DenseController
from ntm.heads import WriteHead, ReadHead

def make_example(size, length):
    sequence = np.random.binomial(1, 0.5, (length, size)).astype(np.uint8)
    example_input = np.zeros((1, 2 * length + 1, size + 1))
    example_output = np.zeros((1, 2 * length + 1, size + 1))

    example_input[0, :length, :size] = sequence
    example_output[0, length + 1:, :size] = sequence
    example_input[0, length, -1] = 1

    return example_input, example_output

input_var = T.dtensor3('input')
target = T.dtensor3('target')

# Network parameters
n = 20
num_units = 100
l = 8
memory_shape = (128, 20)

# Network definition
l_input = InputLayer((1, None, l + 1), input_var=input_var)
_, seqlen, _ = l_input.input_var.shape
# Neural Turing Machine layer
memory = Memory(memory_shape, name='memory', learn_init=True)
controller = DenseController(l_input, num_units=num_units, num_reads=1 * memory_shape[1], 
    nonlinearity=lasagne.nonlinearities.sigmoid, learn_init=True,
    name='controller')
heads = [
    WriteHead([controller, memory], shifts=(-1, 1), name='write', learn_init=False),
    ReadHead([controller, memory], shifts=(-1, 1), name='read', learn_init=False)
]
l_ntm = NTMLayer(l_input, memory=memory, controller=controller, \
      heads=heads, grad_clipping=10.)
l_shp = ReshapeLayer(l_ntm, (-1, num_units))
l_dense = DenseLayer(l_shp, num_units=l + 1, nonlinearity=lasagne.nonlinearities.sigmoid, \
    name='dense')
l_out = ReshapeLayer(l_dense, (1, seqlen, l + 1))

# Loss function
pred = lasagne.layers.get_output(l_out)
loss = T.mean(lasagne.objectives.binary_crossentropy(pred, target))
# Gradient descent updates
params = lasagne.layers.get_all_params(l_out, trainable=True)
updates = lasagne.updates.rmsprop(loss, params, \
    learning_rate=1e-4, rho=0.9, epsilon=1e-6)

train_fn = theano.function([input_var, target], loss, updates=updates)

# Training
max_sequences = 500000
max_length = 6
for batch in range(max_sequences):
    if batch == 200000:
        max_length = 20
    length = random.randint(1, max_length)
    i, o = make_example(8, length)
    score = train_fn(i, o)
    if np.isnan(score):
        break
    if batch % 500 == 0:
        print 'Batch #%d: %.6f' % (batch, score)
