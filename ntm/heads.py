import theano
import theano.tensor as T

import lasagne.init
from lasagne.layers import Layer, MergeLayer DenseLayer
import lasagne.nonlinearities
import lasagne.layer.helper as helper

import .similarities


class Head(MergeLayer):
    """
    docstring for Head
    Inherit from MergeLayer since the head gets inputs
    [h_t, w_tm1, M_t] where h_t is the output of the 
    controller, w_tm1 the weights at time t-1 and M_t
    the memory
    """
    def __init__(self, incomings, shifts=(-1, 1),
                 W_hid_to_key=lasagne.init.GlorotUniform(),
                 b_hid_to_key=lasagne.init.Constant(0.)
                 W_hid_to_beta=lasagne.init.GlorotUniform(),
                 b_hid_to_beta=lasagne.init.Constant(0.),
                 W_hid_to_gate=lasagne.init.GlorotUniform(),
                 b_hid_to_gate=lasagne.init.Constant(0.),
                 W_hid_to_shift=lasagne.init.GlorotUniform(),
                 b_hid_to_shift=lasagne.init.Constant(0.),
                 W_hid_to_gamma=lasagne.init.GlorotUniform(),
                 b_hid_to_gamma=lasagne.init.Constant(0.),
                 weights_init=lasagne.init.GlorotUniform(),
                 learn_init=True,
                 **kwargs):

        self.ctrl_output, self.weight_tm1, self.memory = incomings
        self.memory_size = lasagne.helper.get_output_shape(self.memory)
        self.learn_init = learn_init
        basename = kwargs.get('name', 'head')
    
        self.key = DenseLayer(self.ctrl_output, num_units=self.memory_size[0],
            W=W_hid_to_key, b=b_hid_to_key, nonlinearity=None,
            name=basename + '.key')
        
        self.beta = DenseLayer(self.ctrl_output, num_units=1,
            W=W_hid_to_beta, b=b_hid_to_beta, nonlinearity=None,
            name=basename + '.beta')

        self.gate = DenseLayer(self.ctrl_output, num_units=1,
            W=W_hid_to_gate, b=b_hid_to_gate, nonlinearity=None,
            name=basename + '.gate')

        if len(shifts) != 2:
            raise ValueError('`shifts` must be of length 2 (`%s`.shifts ' +
                             'has length %d)' % (basename, len(shifts)))
        if shifts[0] > shifts[1]:
            raise ValueError('`shifts` must be an interval (`%s`.shifts ' +
                             'has value %s)' % (basename, shifts))
        self.shifts = (int(shifts[0]), int(shifts[1]))
        num_shifts = self.shifts[1] - self.shifts[0] + 1
        self.shift = DenseLayer(self.ctrl_output, num_units=num_shifts,
            W=W_hid_to_shift, b_hid_to_shift, nonlinearity=None,
            name=basename + '.shift')

        self.gamma = DenseLayer(self.ctrl_output, num_units=1,
            W=W_hid_to_gamma, b=b_hid_to_gamma, nonlinearity=None,
            name=basename + '.gamma')

        self.weights_init = self.add_param(
            weights_init, (1,) + self.memory_size[1:],
            name=basename + '.weights_init', trainable=learn_init, regularizable=False)

        self.weights = T.dvector(basename + '.weights')

        super(Head, self).__init__(incomings, **kwargs)

    def get_output_for(self, inputs, **kwargs):
        h_t, w_tm1, M_t = inputs
        k_t = helper.get_output(self.key, h_t, **kwargs)
        beta_t = helper.get_output(self.beta, h_t, **kwargs)
        g_t = helper.get_output(self.gate, h_t, **kwargs)
        s_t = helper.get_output(self.shift, h_t, **kwargs)
        gamma_t = helper.get_output(self.gamma, h_t, **kwargs)

        # Content Adressing (3.3.1)
        betaK = beta_t * similarities.cosine_similarity(k_t, M_t)
        w_c = lasagne.nonlinearities(betaK)

        # Interpolation (3.3.2)
        w_g = g_t * w_c + (1. - g_t) * w_tm1

        # Convolutional Shift (3.3.2)
        # TODO: w_tilde = ...
        w_tilde = w_g

        # Sharpening (3.3.2)
        w = w_tilde ** gamma_t
        w /= T.sum(w) + 1e-9

        return w


class WriteHead(Head):
    """
    docstring for WriteHead
    """
    def __init__(self, incomings, shifts=(-1, 1),
                 W_hid_to_key=lasagne.init.GlorotUniform(),
                 b_hid_to_key=lasagne.init.Constant(0.)
                 W_hid_to_beta=lasagne.init.GlorotUniform(),
                 b_hid_to_beta=lasagne.init.Constant(0.),
                 W_hid_to_gate=lasagne.init.GlorotUniform(),
                 b_hid_to_gate=lasagne.init.Constant(0.),
                 W_hid_to_shift=lasagne.init.GlorotUniform(),
                 b_hid_to_shift=lasagne.init.Constant(0.),
                 W_hid_to_gamma=lasagne.init.GlorotUniform(),
                 b_hid_to_gamma=lasagne.init.Constant(0.),
                 W_hid_to_erase=lasagne.init.GlorotUniform(),
                 b_hid_to_erase=lasagne.init.Constant(0.)
                 W_hid_to_add=lasagne.init.GlorotUniform(),
                 b_hid_to_add=lasagne.init.Constant(0.),
                 **kwargs):

        super(WriteHead, self).__init__(incomings, shifts,
            W_hid_to_key=W_hid_to_key, b_hid_to_key=b_hid_to_key,
            W_hid_to_beta=W_hid_to_beta, b_hid_to_beta=b_hid_to_beta,
            W_hid_to_gate=W_hid_to_gate, b_hid_to_gate=b_hid_to_gate,
            W_hid_to_shift=W_hid_to_shift, b_hid_to_shift=b_hid_to_shift,
            W_hid_to_gamma=W_hid_to_gamma, b_hid_to_gamma=b_hid_to_gamma)
    
        self.erase = DenseLayer(self.ctrl_output, num_units=self.memory_size[0],
            W=W_hid_to_erase, b=b_hid_to_erase, nonlinearity=None,
            name=basename + '.erase')

        self.add = DenseLayer(self.ctrl_output, num_units=self.memory_size[0],
            W=W_hid_to_add, b=b_hid_to_add, nonlinearity=None,
            name=basename + '.add')


class ReadHead(Head):
    """
    docstring for ReadHead
    """
    def __init__(self):
        super(ReadHead, self).__init__()


class HeadCollection(object):
    """docstring for HeadCollection"""
    def __init__(self, arg):
        super(HeadCollection, self).__init__()
        self.arg = arg
