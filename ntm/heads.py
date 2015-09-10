import theano
import theano.tensor as T

import lasagne.init
from lasagne.layers import Layer, MergeLayer, DenseLayer, InputLayer
import lasagne.nonlinearities
import lasagne.layer.helper as helper

import .similarities


class Head(MergeLayer):
    """
    docstring for HeadLayer
    [h_t, M_t, w_tm1]
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

        self.ctrl_layer, self.memory_layer = incomings
        self.memory_size = lasagne.helper.get_output_shape(self.memory_layer)
        basename = kwargs.get('name', 'head')
        incomings.append(InputLayer((self.ctrl_layer.output_shape[0], \
            self.memory_size[0]), name=basename + '.recurrent'))
        super(Head, self).__init__(incomings, **kwargs)

        self.learn_init = learn_init
    
        self.key = DenseLayer(self.ctrl_layer, num_units=self.memory_size[0],
            W=W_hid_to_key, b=b_hid_to_key, nonlinearity=None,
            name=basename + '.key')
        self.W_hid_to_key, self.b_hid_to_key = self.key.W, self.key.b
        
        self.beta = DenseLayer(self.ctrl_layer, num_units=1,
            W=W_hid_to_beta, b=b_hid_to_beta, nonlinearity=None,
            name=basename + '.beta')
        self.W_hid_to_beta, self.b_hid_to_beta = self.beta.W, self.beta.b

        self.gate = DenseLayer(self.ctrl_layer, num_units=1,
            W=W_hid_to_gate, b=b_hid_to_gate, nonlinearity=None,
            name=basename + '.gate')
        self.W_hid_to_gate, self.b_hid_to_gate = self.gate.W, self.gate.b

        if len(shifts) != 2:
            raise ValueError('`shifts` must be of length 2 (`%s`.shifts ' +
                             'has length %d)' % (basename, len(shifts)))
        if shifts[0] > shifts[1]:
            raise ValueError('`shifts` must be an interval (`%s`.shifts ' +
                             'has value %s)' % (basename, shifts))
        self.shifts = (int(shifts[0]), int(shifts[1]))
        num_shifts = self.shifts[1] - self.shifts[0] + 1
        self.shift = DenseLayer(self.ctrl_layer, num_units=num_shifts,
            W=W_hid_to_shift, b_hid_to_shift, nonlinearity=None,
            name=basename + '.shift')
        self.W_hid_to_shift, self.b_hid_to_shift = self.shift.W, self.shift.b

        self.gamma = DenseLayer(self.ctrl_layer, num_units=1,
            W=W_hid_to_gamma, b=b_hid_to_gamma, nonlinearity=None,
            name=basename + '.gamma')
        self.W_hid_to_gamma, self.b_hid_to_gamma = self.gamma.W, self.gamma.b

        self.weights_init = self.add_param(
            weights_init, (1,) + self.memory_size[1:],
            name=basename + '.weights_init', trainable=learn_init, regularizable=False)


    def get_output_for(self, inputs, **kwargs):
        h_t, M_t, w_tm1 = inputs
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
                 weights_init=lasagne.init.GlorotUniform(),
                 learn_init=True,
                 **kwargs):
        super(WriteHead, self).__init__(incomings, shifts,
            W_hid_to_key=W_hid_to_key, b_hid_to_key=b_hid_to_key,
            W_hid_to_beta=W_hid_to_beta, b_hid_to_beta=b_hid_to_beta,
            W_hid_to_gate=W_hid_to_gate, b_hid_to_gate=b_hid_to_gate,
            W_hid_to_shift=W_hid_to_shift, b_hid_to_shift=b_hid_to_shift,
            W_hid_to_gamma=W_hid_to_gamma, b_hid_to_gamma=b_hid_to_gamma,
            weights_init=weights_init, learn_init=learn_init,
            **kwargs)
    
        self.erase = DenseLayer(self.ctrl_layer, num_units=self.memory_size[0],
            W=W_hid_to_erase, b=b_hid_to_erase, nonlinearity=None,
            name=basename + '.erase')
        self.W_hid_to_erase, self.b_hid_to_erase = W_hid_to_erase, b_hid_to_erase

        self.add = DenseLayer(self.ctrl_layer, num_units=self.memory_size[0],
            W=W_hid_to_add, b=b_hid_to_add, nonlinearity=None,
            name=basename + '.add')
        self.W_hid_to_add, self.b_hid_to_add = W_hid_to_add, b_hid_to_add


class ReadHead(Head):
    """
    docstring for ReadHead
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
        super(ReadHead, self).__init__(incomings, shifts,
            W_hid_to_key=W_hid_to_key, b_hid_to_key=b_hid_to_key,
            W_hid_to_beta=W_hid_to_beta, b_hid_to_beta=b_hid_to_beta,
            W_hid_to_gate=W_hid_to_gate, b_hid_to_gate=b_hid_to_gate,
            W_hid_to_shift=W_hid_to_shift, b_hid_to_shift=b_hid_to_shift,
            W_hid_to_gamma=W_hid_to_gamma, b_hid_to_gamma=b_hid_to_gamma,
            weights_init=weights_init, learn_init=learn_init
            **kwargs)


# class Head(Layer):
#     """
#     docstring for Head
#     """
#     def __init__(self, incoming, shifts=(-1, 1),
#                  W_hid_to_key=lasagne.init.GlorotUniform(),
#                  b_hid_to_key=lasagne.init.Constant(0.)
#                  W_hid_to_beta=lasagne.init.GlorotUniform(),
#                  b_hid_to_beta=lasagne.init.Constant(0.),
#                  W_hid_to_gate=lasagne.init.GlorotUniform(),
#                  b_hid_to_gate=lasagne.init.Constant(0.),
#                  W_hid_to_shift=lasagne.init.GlorotUniform(),
#                  b_hid_to_shift=lasagne.init.Constant(0.),
#                  W_hid_to_gamma=lasagne.init.GlorotUniform(),
#                  b_hid_to_gamma=lasagne.init.Constant(0.),
#                  weights_init=lasagne.init.GlorotUniform(),
#                  learn_init=True,
#                  **kwargs):
#         super(Head, self).__init__(incoming, **kwargs)
#         self.shifts = shifts
#         self.W_hid_to_key, self.b_hid_to_key = W_hid_to_key, b_hid_to_key
#         self.W_hid_to_beta, self.b_hid_to_beta = W_hid_to_beta, b_hid_to_beta
#         self.W_hid_to_gate, self.b_hid_to_gate = W_hid_to_gate, b_hid_to_gate
#         self.W_hid_to_shift, self.b_hid_to_shift = W_hid_to_shift, b_hid_to_shift
#         self.W_hid_to_gamma, self.b_hid_to_gamma = W_hid_to_gamma, b_hid_to_gamma
#         self.weights_init = weights_init

#     def get_output_for(self, input, **kwargs):
#         return input