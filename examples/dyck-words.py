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

from utils.generators import DyckWordsTask
from utils.visualization import Dashboard

try:
    from palettable.cubehelix import jim_special_16
    default_cmap = jim_special_16.mpl_colormap
except ImportError:
    default_cmap = 'bone'


# Save model snapshots
snapshot_directory = 'snapshots/dyck-words/{0:%y}{0:%m}{0:%d}-{0:%H}{0:%M}{0:%S}'\
                     '-dyck-words'.format(datetime.now())
os.mkdir(snapshot_directory)
print 'Snapshots directory: %s' % (snapshot_directory,)

print np.random.get_state()

input_var = T.dtensor3('input')
target_var = T.dmatrix('target')

num_units = 100
memory_shape = (128, 20)
batch_size = 1

# Input Layer
l_input = InputLayer((batch_size, None, 1), input_var=input_var)
_, seqlen, _ = l_input.input_var.shape

# Neural Turing Machine Layer
memory = Memory(memory_shape, name='memory', memory_init=lasagne.init.Constant(1e-6), learn_init=False)
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
      heads=heads, only_return_final=True)

# Output Layer
l_output = DenseLayer(l_ntm, num_units=1, nonlinearity=lasagne.nonlinearities.sigmoid, \
    name='dense')


pred = T.clip(lasagne.layers.get_output(l_output), 1e-10, 1. - 1e-10)
loss = T.mean(lasagne.objectives.binary_crossentropy(pred, target_var))

params = lasagne.layers.get_all_params(l_out, trainable=True)
updates = graves_rmsprop(loss, params, beta=1e-3)

train_fn = theano.function([input_var, target_var], loss, updates=updates)
ntm_fn = theano.function([input_var], pred)
ntm_layer_fn = theano.function([input_var], lasagne.layers.get_output(l_ntm, deterministic=True, get_details=True))

# Training
generator = DyckWordsTask(batch_size=batch_size, max_iter=5000000, max_length=5)

try:
    scores, all_scores = [], []
    best_score = -1.
    for i, (example_input, example_output) in generator:
        score = train_fn(example_input, example_output)
        scores.append(score)
        all_scores.append(score)
        if i % 500 == 0:
            mean_scores = np.mean(scores)
            if (best_score < 0) or (mean_scores < best_score):
                best_score = mean_scores
                with open(os.path.join(snapshot_directory, 'model_best.npy'), 'w') as f:
                    np.save(f, lasagne.layers.get_all_param_values(l_output))
            if i % 2000 == 0:
                with open(os.path.join(snapshot_directory, 'model_%d.npy' % i), 'w') as f:
                    np.save(f, lasagne.layers.get_all_param_values(l_output))
            print 'Batch #%d: %.6f' % (i, mean_scores)
            scores = []
except KeyboardInterrupt:
    with open(os.path.join(snapshot_directory, 'learning_curve.npy'), 'w') as f:
        np.save(f, all_scores)
    pass

# Visualization
def marker1(params):
    return params['num_items'] * (params['item_length'] + 1)
def marker2(params):
    return (params['num_items'] + 1) * (params['item_length'] + 1)
markers = [
    {
        'location': marker1,
        'style': {'color': 'red', 'ls': '-'}
    },
    {
        'location': marker2,
        'style': {'color': 'green', 'ls': '-'}
    }
]

dashboard = Dashboard(generator=generator, ntm_fn=ntm_fn, ntm_layer_fn=ntm_layer_fn, \
    memory_shape=memory_shape, markers=markers, cmap=default_cmap)
