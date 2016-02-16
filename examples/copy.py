import theano
import theano.tensor as T
import numpy as np
import random
import os
from datetime import datetime

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

from utils.generators import CopyTask
from utils.visualization import Dashboard


# Save model snapshots
snapshot_directory = 'snapshots/copy/{0:%y}{0:%m}{0:%d}-{0:%H}{0:%M}{0:%S}'\
                     '-copy'.format(datetime.now())
os.mkdir(snapshot_directory)
print 'Snapshots directory: %s' % (snapshot_directory,)

input_var = T.dtensor3('input')
target = T.dtensor3('target')

size = 8
num_units = 100
memory_shape = (128, 20)
batch_size = 1

l_input = InputLayer((batch_size, None, size + 1), input_var=input_var)
_, seqlen, _ = l_input.input_var.shape
# Neural Turing Machine Layer
memory = Memory(memory_shape, name='memory', memory_init=lasagne.init.Constant(1e-6), learn_init=False)
controller = DenseController(l_input, num_units=num_units, num_reads=1 * memory_shape[1], 
    nonlinearity=lasagne.nonlinearities.rectify,
    name='controller')
heads = [
    WriteHead(controller, num_shifts=3, memory_size=memory_shape, name='write', learn_init=False,
        W_hid_to_sign=None, nonlinearity_key=lasagne.nonlinearities.tanh, W_hid_to_sign_add=None,
        nonlinearity_add=lasagne.nonlinearities.tanh),
    ReadHead(controller, num_shifts=3, memory_size=memory_shape, name='read',
        learn_init=False, W_hid_to_sign=None, nonlinearity_key=lasagne.nonlinearities.tanh)
]
l_ntm = NTMLayer(l_input, memory=memory, controller=controller, \
      heads=heads)
l_shp = ReshapeLayer(l_ntm, (-1, num_units))
l_dense = DenseLayer(l_shp, num_units=size + 1, nonlinearity=lasagne.nonlinearities.sigmoid, \
    name='dense')
l_output = ReshapeLayer(l_dense, (batch_size, seqlen, size + 1))

pred = T.clip(lasagne.layers.get_output(l_output), 1e-10, 1. - 1e-10)
loss = T.mean(lasagne.objectives.binary_crossentropy(pred, target))

params = lasagne.layers.get_all_params(l_output, trainable=True)
updates = graves_rmsprop(loss, params, beta=1e-3)

train_fn = theano.function([input_var, target], loss, updates=updates)
ntm_fn = theano.function([input_var], pred)
ntm_layer_fn = theano.function([input_var], lasagne.layers.get_output(l_ntm, get_details=True))

generator = CopyTask(batch_size=batch_size, max_iter=500000, size=size, max_length=5)

try:
    scores, all_scores = [], []
    best_score = -1.
    for i, (example_input, example_output) in generator:
        score = train_fn(example_input, example_output)
        scores.append(score)
        all_scores.append(score)
        if i % 200 == 0:
            with open(os.path.join(snapshot_directory, 'model_%d.npy' % i), 'w') as f:
                np.save(f, lasagne.layers.get_all_param_values(l_output))
        if i % 500 == 0:
            mean_scores = np.mean(scores)
            if (best_score < 0) or (mean_scores < best_score):
                best_score = mean_scores
                with open(os.path.join(snapshot_directory, 'model_best.npy'), 'w') as f:
                    np.save(f, lasagne.layers.get_all_param_values(l_output))
            print 'Batch #%d: %.6f' % (i, mean_scores)
            scores = []
except KeyboardInterrupt:
    pass

markers = [
    {
        'location': (lambda params: params['length']),
        'style': {'color': 'red'}
    }
]
from palettable.cubehelix import jim_special_16
dashboard = Dashboard(generator=generator, ntm_fn=ntm_fn, ntm_layer_fn=ntm_layer_fn, \
    memory_shape=memory_shape, markers=markers, cmap=jim_special_16.mpl_colormap)
