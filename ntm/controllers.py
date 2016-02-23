import theano
import theano.tensor as T
import numpy as np

from lasagne.layers import Layer, MergeLayer, DenseLayer, InputLayer
from lasagne.layers.recurrent import Gate, LSTMLayer
import lasagne.nonlinearities
import lasagne.init
import lasagne.utils

from collections import OrderedDict


class Controller(Layer):
    """
    docstring for Controller
    """
    def __init__(self, incoming, memory_shape, num_units, num_reads,
                 hid_init=lasagne.init.GlorotUniform(),
                 learn_init=False,
                 **kwargs):
        super(Controller, self).__init__(incoming, **kwargs)
        self.hid_init = self.add_param(hid_init, (1, num_units),
            name='hid_init', regularizable=False, trainable=learn_init)
        self.memory_shape = memory_shape
        self.num_units = num_units
        self.num_reads = num_reads

    def step(self, input, reads, hidden, state, *args, **kwargs):
        """
        Returns (output, state) where
            - 'output' is the true hidden state returned by the controller
            - 'state' is the augmented hidden state (eg. state + cell for LSTM)
        """
        raise NotImplementedError

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.num_units)


class DenseController(Controller):
    """
    docstring for DenseController
    """
    def __init__(self, incoming, memory_shape, num_units, num_reads,
                 W_in_to_hid=lasagne.init.GlorotUniform(),
                 b_in_to_hid=lasagne.init.Constant(0.),
                 W_reads_to_hid=lasagne.init.GlorotUniform(),
                 b_reads_to_hid=lasagne.init.Constant(0.),
                 nonlinearity=lasagne.nonlinearities.rectify,
                 hid_init=lasagne.init.GlorotUniform(),
                 learn_init=False,
                 **kwargs):
        super(DenseController, self).__init__(incoming, memory_shape, num_units,
                                              num_reads, hid_init, learn_init,
                                              **kwargs)
        self.nonlinearity = (lasagne.nonlinearities.identity if 
                             nonlinearity is None else nonlinearity)

        def add_weight_and_bias_params(input_dim, W, b, name):
            return (self.add_param(W, (input_dim, self.num_units), 
                name='W_{}'.format(name)),
                self.add_param(b, (self.num_units,),
                name='b_{}'.format(name)) if b is not None else None)
        num_inputs = int(np.prod(self.input_shape[2:]))
        # Inputs / Hidden parameters
        self.W_in_to_hid, self.b_in_to_hid = add_weight_and_bias_params(num_inputs,
            W_in_to_hid, b_in_to_hid, name='in_to_hid')
        # Read vectors / Hidden parameters
        self.W_reads_to_hid, self.b_reads_to_hid = add_weight_and_bias_params(self.num_reads * self.memory_shape[1],
            W_reads_to_hid, b_reads_to_hid, name='reads_to_hid')

    def step(self, input, reads, *args):
        if input.ndim > 2:
            input = input.flatten(2)
        if reads.ndim > 2:
            reads = reads.flatten(2)

        activation = T.dot(input, self.W_in_to_hid) + \
                     T.dot(reads, self.W_reads_to_hid)
        if self.b_in_to_hid is not None:
            activation = activation + self.b_in_to_hid.dimshuffle('x', 0)
        if self.b_reads_to_hid is not None:
            activation = activation + self.b_reads_to_hid.dimshuffle('x', 0)
        state = self.nonlinearity(activation)
        return state, state

    @property
    def outputs_info(self):
        ones_vector = T.ones((self.input_shape[0], 1))
        hid_init = T.dot(ones_vector, self.hid_init)
        hid_init = T.unbroadcast(hid_init, 0)
        return [hid_init, hid_init]
