import theano
import theano.tensor as T
import numpy as np
import random

import matplotlib.pyplot as plt
import pandas as pd

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

input_var = T.dtensor3('input')
target = T.dtensor3('target')

size = 8
num_units = 100
memory_shape = (128, 20)

l_input = InputLayer((1, None, size + 2), input_var=input_var)
_, seqlen, _ = l_input.input_var.shape
# Neural Turing Machine Layer
memory = Memory(memory_shape, name='memory', learn_init=False)
controller = DenseController(l_input, num_units=num_units, num_reads=1 * memory_shape[1], 
    nonlinearity=lasagne.nonlinearities.rectify,
    name='controller')
heads = [
    WriteHead(controller, num_shifts=3, memory_size=memory_shape, name='write', learn_init=False,
        W_hid_to_sign=None, nonlinearity_key=lasagne.nonlinearities.tanh, W_hid_to_sign_add=None,
        nonlinearity_add=lasagne.nonlinearities.tanh, p=0.),
    ReadHead(controller, num_shifts=3, memory_size=memory_shape, name='read', learn_init=False,
        W_hid_to_sign=None, nonlinearity_key=lasagne.nonlinearities.tanh, p=0.5)
]
l_ntm = NTMLayer(l_input, memory=memory, controller=controller, \
      heads=heads)
l_shp = ReshapeLayer(l_ntm, (-1, num_units))
l_dense = DenseLayer(l_shp, num_units=size + 2, nonlinearity=lasagne.nonlinearities.sigmoid, \
    name='dense')
l_out = ReshapeLayer(l_dense, (1, seqlen, size + 2))

pred = T.clip(lasagne.layers.get_output(l_out), 1e-10, 1. - 1e-10)
test_pred = T.clip(lasagne.layers.get_output(l_out, deterministic=True), 1e-10, 1. - 1e-10)
loss = T.mean(lasagne.objectives.binary_crossentropy(pred, target))

params = lasagne.layers.get_all_params(l_out, trainable=True)
updates = graves_rmsprop(loss, params, beta=1e-4)

train_fn = theano.function([input_var, target], loss, updates=updates)
ntm_fn = theano.function([input_var], test_pred)
ntm_layer_fn = theano.function([input_var], lasagne.layers.get_output(l_ntm, deterministic=True, get_details=True))

generator = RepeatCopyTask(max_iter=500000, size=size, min_length=3, \
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
from palettable.cubehelix import jim_special_16
dashboard = Dashboard(generator=generator, ntm_fn=ntm_fn, ntm_layer_fn=ntm_layer_fn, \
    memory_shape=memory_shape, markers=markers, cmap=jim_special_16.mpl_colormap)
