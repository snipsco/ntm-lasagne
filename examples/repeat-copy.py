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
from ntm.updates import graves_rmsprop

from utils.generators import RepeatCopyTask
from utils.visualization import Dashboard

try:
    from palettable.cubehelix import jim_special_16
    default_cmap = jim_special_16.mpl_colormap
except ImportError:
    default_cmap = 'bone'


input_var, target_var = T.dtensor3s('input', 'target')

# Parameters - General
size = 8
batch_size = 1
# Parameters - NTM
num_units = 100
memory_shape = (128, 20)

# Input Layer
l_input = InputLayer((batch_size, None, size + 2), input_var=input_var)
_, seqlen, _ = l_input.input_var.shape

# Neural Turing Machine Layer
memory = Memory(memory_shape, name='memory', learn_init=False)
controller = DenseController(l_input, num_units=num_units, num_reads=1 * memory_shape[1], 
    nonlinearity=lasagne.nonlinearities.rectify,
    name='controller')
heads = [
    WriteHead(controller, num_shifts=3, memory_size=memory_shape, name='write', learn_init=False,
        W_hid_to_sign=None, nonlinearity_key=lasagne.nonlinearities.rectify, W_hid_to_sign_add=None,
        nonlinearity_add=lasagne.nonlinearities.rectify, p=0.),
    ReadHead(controller, num_shifts=3, memory_size=memory_shape, name='read', learn_init=False,
        W_hid_to_sign=None, nonlinearity_key=lasagne.nonlinearities.rectify, p=0.)
]
l_ntm = NTMLayer(l_input, memory=memory, controller=controller, \
      heads=heads)

# Output Layer
l_shp = ReshapeLayer(l_ntm, (-1, num_units))
l_dense = DenseLayer(l_shp, num_units=size + 2, nonlinearity=lasagne.nonlinearities.sigmoid, \
    name='dense')
l_out = ReshapeLayer(l_dense, (batch_size, seqlen, size + 2))


pred = T.clip(lasagne.layers.get_output(l_out), 1e-10, 1. - 1e-10)
loss = T.mean(lasagne.objectives.binary_crossentropy(pred, target_var))

params = lasagne.layers.get_all_params(l_out, trainable=True)
updates = graves_rmsprop(loss, params, beta=1e-3)

train_fn = theano.function([input_var, target_var], loss, updates=updates)
ntm_fn = theano.function([input_var], pred)
ntm_layer_fn = theano.function([input_var], lasagne.layers.get_output(l_ntm, deterministic=True, get_details=True))

# Training
generator = RepeatCopyTask(batch_size=batch_size, max_iter=500000, size=size, min_length=3, \
    max_length=5, max_repeats=5, unary=True)

try:
    scores, all_scores = [], []
    for i, (example_input, example_output) in generator:
        score = train_fn(example_input, example_output)
        scores.append(score)
        all_scores.append(score)
        if i % 500 == 0:
            print 'Batch #%d: %.6f' % (i, np.mean(scores))
            scores = []
except KeyboardInterrupt:
    pass

# Visualization
def marker(generator):
    def marker_(params):
        num_repeats_length = params['repeats'] if generator.unary else 1
        return params['length'] + num_repeats_length
    return marker_
markers = [
    {
        'location': marker(generator),
        'style': {'color': 'red', 'ls': '-'}
    }
]

dashboard = Dashboard(generator=generator, ntm_fn=ntm_fn, ntm_layer_fn=ntm_layer_fn, \
    memory_shape=memory_shape, markers=markers, cmap=default_cmap)
