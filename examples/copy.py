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
from ntm.tasks import copy
from ntm.updates import graves_rmsprop

input_var = T.dtensor3('input')
target = T.dtensor3('target')

n = 20
num_units = 100
l = 8
memory_shape = (128, 20)

l_input = InputLayer((1, None, l + 1), input_var=input_var)
_, seqlen, _ = l_input.input_var.shape
# Neural Turing Machine Layer
memory = Memory(memory_shape, name='memory', learn_init=False)
controller = DenseController(l_input, num_units=num_units, num_reads=1 * memory_shape[1], 
    nonlinearity=lasagne.nonlinearities.rectify,
    name='controller')
heads = [
    WriteHead(controller, num_shifts=3, memory_size=memory_shape, name='write', learn_init=False),
    ReadHead(controller, num_shifts=3, memory_size=memory_shape, name='read', learn_init=False)
]
l_ntm = NTMLayer(l_input, memory=memory, controller=controller, \
      heads=heads)
l_shp = ReshapeLayer(l_ntm, (-1, num_units))
l_dense = DenseLayer(l_shp, num_units=l + 1, nonlinearity=lasagne.nonlinearities.sigmoid, \
    name='dense')
l_out = ReshapeLayer(l_dense, (1, seqlen, l + 1))

pred = T.clip(lasagne.layers.get_output(l_out), 1e-10, 1. - 1e-10)
loss = T.mean(lasagne.objectives.binary_crossentropy(pred, target))

params = lasagne.layers.get_all_params(l_out, trainable=True)
updates = graves_rmsprop(loss, params, beta=1e-3)

train_fn = theano.function([input_var, target], loss, updates=updates)
ntm_fn = theano.function([input_var], pred)

try:
    max_sequences = 500000
    max_length = 5
    scores = []
    all_scores = []
    for batch in range(max_sequences):
        length = random.randint(1, max_length)
        i, o = copy(8, length)
        score = train_fn(i, o)
        scores.append(score)
        all_scores.append(score)
        if batch % 500 == 0:
            print 'Batch #%d: %.6f' % (batch, np.mean(scores))
            scores = []
except KeyboardInterrupt:
    pass

def learning_curve():
    sc = pd.Series(all_scores)
    ma = pd.rolling_mean(sc, window=500)

    ax = plt.subplot(1, 1, 1)
    ax.plot(sc.index, sc, color='lightgray')
    ax.plot(ma.index, ma, color='red')
    ax.set_yscale('log')
    ax.set_xlim(sc.index.min(), sc.index.max())
    plt.show()

def viz(n):
    i, o = copy(8, n)
    plt.subplot2grid((3, 1), (0, 0))
    plt.imshow(i[0].T, interpolation='nearest', cmap='bone')
    
    plt.subplot2grid((3, 1), (1, 0))
    plt.imshow(o[0].T, interpolation='nearest', cmap='bone')

    plt.subplot2grid((3, 1), (2, 0))
    plt.imshow(ntm_fn(i)[0].T, interpolation='nearest', cmap='bone')
    plt.show()
