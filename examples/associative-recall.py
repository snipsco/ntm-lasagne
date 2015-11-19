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
from ntm.tasks import associative_recall, copy
from ntm.updates import graves_rmsprop

input_var = T.dtensor3('input')
target = T.dtensor3('target')

n = 20
num_units = 100
l = 8
memory_shape = (128, 20)

l_input = InputLayer((1, None, l + 2), input_var=input_var)
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
l_dense = DenseLayer(l_shp, num_units=l + 2, nonlinearity=lasagne.nonlinearities.sigmoid, \
    name='dense')
l_out = ReshapeLayer(l_dense, (1, seqlen, l + 2))

pred = T.clip(lasagne.layers.get_output(l_out), 1e-10, 1. - 1e-10)
loss = T.mean(lasagne.objectives.binary_crossentropy(pred, target))

params = lasagne.layers.get_all_params(l_out, trainable=True)
# updates = graves_rmsprop(loss, params, beta=1e-4)
updates = lasagne.updates.adam(loss, params, learning_rate=1e-4)

train_fn = theano.function([input_var, target], loss, updates=updates)
ntm_fn = theano.function([input_var], pred)
ntm_layer_fn = theano.function([input_var], lasagne.layers.get_output(l_ntm, get_details=True))

try:
    max_sequences = 5000000
    min_length = 1
    max_length = 5
    min_item = 3
    max_item = 3
    scores = []
    all_scores = []
    do_copy = True
    for batch in range(max_sequences):
        prev_params = [p.get_value() for p in params]
        length = random.randint(min_length, max_length)
        if do_copy:
            i, o = copy(9, length)
            i[:,:,-2] = 0.
            o[:,:,-2] = 0.
        else:
            item = random.randint(min_item, max_item)
            i, o = associative_recall(8, item, length)            
        score = train_fn(i, o)
        scores.append(score)
        all_scores.append(score)
        if np.any([np.any(np.isnan(p.get_value())) for p in params]):
            break
        if batch % 500 == 0:
            copy_string = '[Copy] ' if do_copy else ''
            print '%sBatch #%d: %.6f' % (copy_string, batch, np.mean(scores))
            if np.mean(scores) < 5e-5 or batch > 200000:
                do_copy = False
                min_length = 2
                max_length = 6
            scores = []
except KeyboardInterrupt:
    pass

# def viz(item_length, num_items, cmap='bone'):
#     example_input, example_output = associative_recall(8, item_length, num_items)
#     example_prediction = ntm_fn(example_input)
#     example_ntm = ntm_layer_fn(example_input)

#     subplot_shape = (3, 3)

#     ax1 = plt.subplot2grid(subplot_shape, (0, 2))
#     ax1.imshow(example_input[0].T, interpolation='nearest', cmap=cmap)
#     ax1.set_title('Input')
#     ax1.get_xaxis().set_visible(False)

#     ax2 = plt.subplot2grid(subplot_shape, (1, 2))
#     ax2.imshow(example_output[0].T, interpolation='nearest', cmap=cmap)
#     ax2.set_title('Output')
#     ax2.get_xaxis().set_visible(False)

#     ax3 = plt.subplot2grid(subplot_shape, (2, 2))
#     ax3.imshow(example_prediction[0].T, interpolation='nearest', cmap=cmap)
#     ax3.set_title('Prediction')
#     ax3.get_xaxis().set_visible(False)

#     ax4 = plt.subplot2grid(subplot_shape, (0, 1), rowspan=3)
#     ax4.imshow(example_ntm[4][0].T, interpolation='nearest', cmap=cmap)
#     ax4.set_title('Read Weights')
#     ax4.get_xaxis().set_visible(False)
#     ax4.plot([num_items * (item_length + 1) - 0.5, num_items * (item_length + 1) - 0.5], [0, 127], color='green')
#     ax4.plot([num_items * (item_length + 1) + item_length + 1.5, num_items * (item_length + 1) + item_length + 1.5], [0, 127], color='red')
#     ax4.set_xlim([-0.5, num_items * (item_length + 1) + 2 * item_length + 1.5])
#     ax4.set_ylim([-0.5, 127.5])
#     print example_ntm[3][0,-1,:].size
#     print example_ntm[3][0].shape
#     print example_ntm[3][0,-1,:]

#     ax5 = plt.subplot2grid(subplot_shape, (0, 0), rowspan=3)
#     ax5.imshow(example_ntm[3][0].T, interpolation='nearest', cmap=cmap)
#     ax5.set_title('Write Weights')
#     ax5.get_xaxis().set_visible(False)
#     ax5.plot([num_items * (item_length + 1) - 0.5, num_items * (item_length + 1) - 0.5], [0, 127], color='green')
#     ax5.plot([num_items * (item_length + 1) + item_length + 1.5, num_items * (item_length + 1) + item_length + 1.5], [0, 127], color='red')
#     ax5.set_xlim([-0.5, num_items * (item_length + 1) + 2 * item_length + 1.5])
#     ax5.set_ylim([-0.5, 127.5])

#     plt.show()