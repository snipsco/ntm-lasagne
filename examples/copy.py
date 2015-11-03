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
    WriteHead(controller, num_shifts=3, memory_size=memory_shape, name='write', learn_init=False,
        W_hid_to_sign=None, nonlinearity_key=lasagne.nonlinearities.tanh, W_hid_to_sign_add=None,
        nonlinearity_add=lasagne.nonlinearities.tanh),
    ReadHead(controller, num_shifts=3, memory_size=memory_shape, name='read',
        learn_init=False, W_hid_to_sign=None, nonlinearity_key=lasagne.nonlinearities.tanh)
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
ntm_layer_fn = theano.function([input_var], lasagne.layers.get_output(l_ntm, get_details=True))

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

def viz(length):
    example_input, example_output = copy(l, length)
    example_prediction = ntm_fn(example_input)
    example_ntm = ntm_layer_fn(example_input)

    subplot_shape = (3, 3)
    ax1 = plt.subplot2grid(subplot_shape, (0, 2))
    ax1.imshow(example_input[0].T, interpolation='nearest', cmap='bone')
    ax1.set_title('Input')
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)

    ax2 = plt.subplot2grid(subplot_shape, (1, 2))
    ax2.imshow(example_output[0].T, interpolation='nearest', cmap='bone')
    ax2.set_title('Output')
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)

    ax3 = plt.subplot2grid(subplot_shape, (2, 2))
    ax3.imshow(example_prediction[0].T, interpolation='nearest', cmap='bone')
    ax3.set_title('Prediction')
    ax3.get_xaxis().set_visible(False)
    ax3.get_yaxis().set_visible(False)

    ax4 = plt.subplot2grid(subplot_shape, (0, 1), rowspan=3)
    ax4.imshow(example_ntm[3][0].T, interpolation='nearest', cmap='bone')
    ax4.set_title('Read Weights')
    ax4.get_xaxis().set_visible(False)
    ax4.plot([length - 0.5, length - 0.5], [0, 127], color='red')
    ax4.set_xlim([-0.5, 2 * length + 0.5])
    ax4.set_ylim([-0.5, 127.5])

    ax5 = plt.subplot2grid(subplot_shape, (0, 0), rowspan=3)
    ax5.imshow(example_ntm[2][0].T, interpolation='nearest', cmap='bone')
    ax5.set_title('Write Weights')
    ax5.get_xaxis().set_visible(False)
    ax5.plot([length - 0.5, length - 0.5], [0, 127], color='red')
    ax5.set_xlim([-0.5, 2 * length + 0.5])
    ax5.set_ylim([-0.5, 127.5])

    plt.show()
