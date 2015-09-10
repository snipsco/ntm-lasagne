import theano
import theano.tensor as T

import lasagne.init
from lasagne.layers import Layer, MergeLayer, DenseLayer, InputLayer
import lasagne.nonlinearities
import lasagne.layers.helper as helper

import similarities


class Head(MergeLayer):
    """
    docstring for HeadLayer
    [h_t, M_t, w_tm1]
    """
    def __init__(self, incomings, shifts=(-1, 1),
                 W_hid_to_key=lasagne.init.GlorotUniform(),
                 b_hid_to_key=lasagne.init.Constant(0.),
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
        self.memory_size = helper.get_output_shape(self.memory_layer)
        self.basename = kwargs.get('name', 'head')
        incomings.append(InputLayer((self.ctrl_layer.output_shape[0], \
            self.memory_size[0]), name=self.basename + '.recurrent'))
        super(Head, self).__init__(incomings, **kwargs)

        self.learn_init = learn_init
    
        self.key = DenseLayer(self.ctrl_layer, num_units=self.memory_size[0],
            W=W_hid_to_key, b=b_hid_to_key, nonlinearity=None,
            name=self.basename + '.key')
        self.W_hid_to_key, self.b_hid_to_key = self.key.W, self.key.b
        
        self.beta = DenseLayer(self.ctrl_layer, num_units=1,
            W=W_hid_to_beta, b=b_hid_to_beta, nonlinearity=None,
            name=self.basename + '.beta')
        self.W_hid_to_beta, self.b_hid_to_beta = self.beta.W, self.beta.b

        self.gate = DenseLayer(self.ctrl_layer, num_units=1,
            W=W_hid_to_gate, b=b_hid_to_gate, nonlinearity=None,
            name=self.basename + '.gate')
        self.W_hid_to_gate, self.b_hid_to_gate = self.gate.W, self.gate.b

        if len(shifts) != 2:
            raise ValueError('`shifts` must be of length 2 (`%s`.shifts ' +
                             'has length %d)' % (self.basename, len(shifts)))
        if shifts[0] > shifts[1]:
            raise ValueError('`shifts` must be an interval (`%s`.shifts ' +
                             'has value %s)' % (self.basename, shifts))
        self.shifts = (int(shifts[0]), int(shifts[1]))
        num_shifts = self.shifts[1] - self.shifts[0] + 1
        self.shift = DenseLayer(self.ctrl_layer, num_units=num_shifts,
            W=W_hid_to_shift, b=b_hid_to_shift, nonlinearity=None,
            name=self.basename + '.shift')
        self.W_hid_to_shift, self.b_hid_to_shift = self.shift.W, self.shift.b

        self.gamma = DenseLayer(self.ctrl_layer, num_units=1,
            W=W_hid_to_gamma, b=b_hid_to_gamma, nonlinearity=None,
            name=self.basename + '.gamma')
        self.W_hid_to_gamma, self.b_hid_to_gamma = self.gamma.W, self.gamma.b

        self.weights_init = self.add_param(
            weights_init, (1,) + self.memory_size[1:],
            name='weights_init', trainable=learn_init, regularizable=False)


    def get_output_for(self, inputs, **kwargs):
        h_t, M_t, w_tm1 = inputs
        k_t = self.key.get_output_for(h_t, **kwargs)
        beta_t = self.beta.get_output_for(h_t, **kwargs)
        g_t = self.gate.get_output_for(h_t, **kwargs)
        s_t = self.shift.get_output_for(h_t, **kwargs)
        gamma_t = self.gamma.get_output_for(h_t, **kwargs)

        # Content Adressing (3.3.1)
        betaK = beta_t * similarities.cosine_similarity(k_t, M_t)
        w_c = lasagne.nonlinearities.softmax(betaK)

        # Interpolation (3.3.2)
        w_g = g_t * w_c + (1. - g_t) * w_tm1

        # Convolutional Shift (3.3.2)
        # TODO: w_tilde = ...
        w_tilde = w_g

        # Sharpening (3.3.2)
        w = w_tilde ** gamma_t
        w /= T.sum(w) + 1e-9

        return w

    def get_params(self, **tags):
        params = super(Head, self).get_params(**tags)
        params += self.key.get_params(**tags)
        params += self.beta.get_params(**tags)
        params += self.gate.get_params(**tags)
        params += self.shift.get_params(**tags)
        params += self.gamma.get_params(**tags)

        return params


class WriteHead(Head):
    """
    docstring for WriteHead
    """
    def __init__(self, incomings, shifts=(-1, 1),
                 W_hid_to_key=lasagne.init.GlorotUniform(),
                 b_hid_to_key=lasagne.init.Constant(0.),
                 W_hid_to_beta=lasagne.init.GlorotUniform(),
                 b_hid_to_beta=lasagne.init.Constant(0.),
                 W_hid_to_gate=lasagne.init.GlorotUniform(),
                 b_hid_to_gate=lasagne.init.Constant(0.),
                 W_hid_to_shift=lasagne.init.GlorotUniform(),
                 b_hid_to_shift=lasagne.init.Constant(0.),
                 W_hid_to_gamma=lasagne.init.GlorotUniform(),
                 b_hid_to_gamma=lasagne.init.Constant(0.),
                 W_hid_to_erase=lasagne.init.GlorotUniform(),
                 b_hid_to_erase=lasagne.init.Constant(0.),
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
            name=self.basename + '.erase')
        self.W_hid_to_erase, self.b_hid_to_erase = self.erase.W, self.erase.b

        self.add = DenseLayer(self.ctrl_layer, num_units=self.memory_size[0],
            W=W_hid_to_add, b=b_hid_to_add, nonlinearity=None,
            name=self.basename + '.add')
        self.W_hid_to_add, self.b_hid_to_add = self.add.W, self.add.b

    def get_params(self, **tags):
        params = super(WriteHead, self).get_params(**tags)
        params += self.erase.get_params(**tags)
        params += self.add.get_params(**tags)

        return params


class ReadHead(Head):
    """
    docstring for ReadHead
    """
    def __init__(self, incomings, shifts=(-1, 1),
                 W_hid_to_key=lasagne.init.GlorotUniform(),
                 b_hid_to_key=lasagne.init.Constant(0.),
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
            weights_init=weights_init, learn_init=learn_init,
            **kwargs)
