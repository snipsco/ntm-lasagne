import theano
import theano.tensor as T

import lasagne.init
from lasagne.layers import Layer, MergeLayer, DenseLayer, InputLayer
import lasagne.nonlinearities
import lasagne.layers.helper as helper
from lasagne.theano_extensions import padding

import similarities

from lasagne.utils import floatX
import numpy as np


class EquiProba(lasagne.init.Initializer):

    def sample(self, shape):
        # TODO: General case, here it only works for 2D
        M = float(shape[1])
        if M == 0:
            raise ValueError('The second dimension '
                'must be non zero')
        return floatX(np.ones(shape) / M)

def clipped_linear(a, b):
    return lambda x: T.clip(x, a, b)


class Head(MergeLayer):
    """
    docstring for HeadLayer
    [h_t, M_t, w_tm1]
    """
    def __init__(self, incomings, shifts=(-1, 1),
                 W_hid_to_sign=lasagne.init.GlorotUniform(),
                 b_hid_to_sign=lasagne.init.Constant(0.),
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
                 weights_init=EquiProba(),
                 learn_init=True,
                 **kwargs):

        self.ctrl_layer, self.memory_layer = incomings
        self.memory_size = helper.get_output_shape(self.memory_layer)
        self.basename = kwargs.get('name', 'head')
        incomings.append(InputLayer((self.ctrl_layer.output_shape[0], \
            self.memory_size[0]), name=self.basename + '.recurrent'))
        super(Head, self).__init__(incomings, **kwargs)

        self.learn_init = learn_init

        self.sign = DenseLayer(self.ctrl_layer, num_units=self.memory_size[1],
            W=W_hid_to_sign, self.b_hid_to_sign, nonlinearity=clipped_linear(-1., 1.),
            name=self.basename + '.sign')
        self.W_hid_to_sign, self.b_hid_to_sign = self.sign.W, self.sign.b
    
        self.key = DenseLayer(self.ctrl_layer, num_units=self.memory_size[1],
            W=W_hid_to_key, b=b_hid_to_key, nonlinearity=clipped_linear(0., 1.),
            name=self.basename + '.key')
        self.W_hid_to_key, self.b_hid_to_key = self.key.W, self.key.b
        
        self.beta = DenseLayer(self.ctrl_layer, num_units=1,
            W=W_hid_to_beta, b=b_hid_to_beta, nonlinearity=lasagne.nonlinearities.rectify,
            name=self.basename + '.beta')
        self.W_hid_to_beta, self.b_hid_to_beta = self.beta.W, self.beta.b

        self.gate = DenseLayer(self.ctrl_layer, num_units=1,
            W=W_hid_to_gate, b=b_hid_to_gate, nonlinearity=T.nnet.hard_sigmoid,
            name=self.basename + '.gate')
        self.W_hid_to_gate, self.b_hid_to_gate = self.gate.W, self.gate.b

        if len(shifts) != 2:
            raise ValueError('`shifts` must be of length 2 (`%s`.shifts ' +
                             'has length %d)' % (self.basename, len(shifts)))
        if shifts[0] > shifts[1]:
            raise ValueError('`shifts` must be an interval (`%s`.shifts ' +
                             'has value %s)' % (self.basename, shifts))
        self.shifts = (int(shifts[0]), int(shifts[1]))
        self.num_shifts = self.shifts[1] - self.shifts[0] + 1
        self.shift = DenseLayer(self.ctrl_layer, num_units=self.num_shifts,
            W=W_hid_to_shift, b=b_hid_to_shift, nonlinearity=lasagne.nonlinearities.softmax,
            name=self.basename + '.shift')
        self.W_hid_to_shift, self.b_hid_to_shift = self.shift.W, self.shift.b

        self.gamma = DenseLayer(self.ctrl_layer, num_units=1,
            W=W_hid_to_gamma, b=b_hid_to_gamma, nonlinearity=lambda x: 1. + lasagne.nonlinearities.rectify(x),
            name=self.basename + '.gamma')
        self.W_hid_to_gamma, self.b_hid_to_gamma = self.gamma.W, self.gamma.b

        # TODO: Replace the 1 by the number of batches
        self.weights_init = self.add_param(
            weights_init, (1, self.memory_size[0]),
            name='weights_init', trainable=learn_init, regularizable=False)


    def get_output_for(self, inputs, **kwargs):
        h_t, M_t, w_tm1 = inputs
        sign_t = self.sign.get_output_for(h_t, **kwargs)
        k_t = self.key.get_output_for(h_t, **kwargs)
        beta_t = self.beta.get_output_for(h_t, **kwargs)
        g_t = self.gate.get_output_for(h_t, **kwargs)
        s_t = self.shift.get_output_for(h_t, **kwargs)
        gamma_t = self.gamma.get_output_for(h_t, **kwargs)

        # Content Adressing (3.3.1)
        beta_t = T.addbroadcast(beta_t, 1)
        betaK = beta_t * similarities.cosine_similarity(sign_t * k_t, M_t)
        w_c = lasagne.nonlinearities.softmax(betaK)

        # Interpolation (3.3.2)
        g_t = T.addbroadcast(g_t, 1)
        w_g = g_t * w_c + (1. - g_t) * w_tm1

        # Convolutional Shift (3.3.2)
        w_g = w_g.dimshuffle(0, 'x', 'x', 1)
        conv_filter = s_t.dimshuffle(0, 'x', 'x', 1)
        pad = (self.num_shifts // 2, (self.num_shifts - 1) // 2)
        w_g = padding.pad(w_g, [pad], batch_ndim=3)
        convolution = T.nnet.conv2d(w_g, conv_filter,
            image_shape=(self.input_shapes[0][0], 1, 1, self.input_shapes[1][0] + pad[0] + pad[1]),
            filter_shape=(1, 1, 1, self.num_shifts),
            subsample=(1, 1),
            border_mode='valid')
        w_tilde = convolution[:, 0, 0, :]

        # Sharpening (3.3.2)
        gamma_t = T.addbroadcast(gamma_t, 1)
        w = T.pow(w_tilde + 1e-6, gamma_t)
        w /= T.sum(w)

        return w

    def get_params(self, **tags):
        params = super(Head, self).get_params(**tags)
        params += self.sign.get_params(**tags)
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
                 W_hid_to_sign=lasagne.init.GlorotUniform(),
                 b_hid_to_sign=lasagne.init.Constant(0.),
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
                 W_hid_to_sign_add=lasagne.init.GlorotUniform(),
                 b_hid_to_sign_add=lasagne.init.Constant(0.),
                 weights_init=EquiProba(),
                 learn_init=True,
                 **kwargs):
        super(WriteHead, self).__init__(incomings, shifts,
            W_hid_to_sign=W_hid_to_sign, b_hid_to_sign=b_hid_to_sign,
            W_hid_to_key=W_hid_to_key, b_hid_to_key=b_hid_to_key,
            W_hid_to_beta=W_hid_to_beta, b_hid_to_beta=b_hid_to_beta,
            W_hid_to_gate=W_hid_to_gate, b_hid_to_gate=b_hid_to_gate,
            W_hid_to_shift=W_hid_to_shift, b_hid_to_shift=b_hid_to_shift,
            W_hid_to_gamma=W_hid_to_gamma, b_hid_to_gamma=b_hid_to_gamma,
            weights_init=weights_init, learn_init=learn_init,
            **kwargs)
    
        self.erase = DenseLayer(self.ctrl_layer, num_units=self.memory_size[1],
            W=W_hid_to_erase, b=b_hid_to_erase, nonlinearity=T.nnet.hard_sigmoid,
            name=self.basename + '.erase')
        self.W_hid_to_erase, self.b_hid_to_erase = self.erase.W, self.erase.b

        self.add = DenseLayer(self.ctrl_layer, num_units=self.memory_size[1],
            W=W_hid_to_add, b=b_hid_to_add, nonlinearity=clipped_linear(0., 1.),
            name=self.basename + '.add')
        self.W_hid_to_add, self.b_hid_to_add = self.add.W, self.add.b

        self.sign_add = DenseLayer(self.ctrl_layer, num_units=self.memory_size[1],
            W=W_hid_to_sign_add, b=b_hid_to_sign_add, nonlinearity=clipped_linear(-1., 1.),
            name=self.basename + '.sign_add')
        self.W_hid_to_sign_add, self.b_hid_to_sign_add = self.sign_add.W, self.sign_add.b

    def get_params(self, **tags):
        params = super(WriteHead, self).get_params(**tags)
        params += self.erase.get_params(**tags)
        params += self.add.get_params(**tags)
        params += self.sign_add.get_params(**tags)

        return params


class ReadHead(Head):
    """
    docstring for ReadHead
    """
    def __init__(self, incomings, shifts=(-1, 1),
                 W_hid_to_sign=lasagne.init.GlorotUniform(),
                 b_hid_to_sign=lasagne.init.Constant(0.),
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
                 weights_init=EquiProba(),
                 learn_init=True,
                 **kwargs):
        super(ReadHead, self).__init__(incomings, shifts,
            W_hid_to_sign=W_hid_to_sign, b_hid_to_sign=b_hid_to_sign,
            W_hid_to_key=W_hid_to_key, b_hid_to_key=b_hid_to_key,
            W_hid_to_beta=W_hid_to_beta, b_hid_to_beta=b_hid_to_beta,
            W_hid_to_gate=W_hid_to_gate, b_hid_to_gate=b_hid_to_gate,
            W_hid_to_shift=W_hid_to_shift, b_hid_to_shift=b_hid_to_shift,
            W_hid_to_gamma=W_hid_to_gamma, b_hid_to_gamma=b_hid_to_gamma,
            weights_init=weights_init, learn_init=learn_init,
            **kwargs)