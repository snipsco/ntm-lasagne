import theano
import theano.tensor as T
import numpy as np

from lasagne.layers import Layer, DenseLayer
from lasagne.theano_extensions import padding
import lasagne.init
import lasagne.nonlinearities

import similarities
import nonlinearities
import init


class Head(Layer):
    r"""
    The base class :class:`Head` represents a generic head for the
    Neural Turing Machine. The heads are responsible for the read/write
    operations on the memory. An instance of :class:`Head` outputs a
    weight vector defined by

    .. math ::
        \alpha_{t} &= \sigma_{alpha}(h_{t} W_{alpha} + b_{alpha})\\
        k_{t} &= \sigma_{key}(h_{t} W_{key} + b_{key})\\
        \beta_{t} &= \sigma_{beta}(h_{t} W_{beta} + b_{beta})\\
        g_{t} &= \sigma_{gate}(h_{t} W_{gate} + b_{gate})\\
        s_{t} &= \sigma_{shift}(h_{t} W_{shift} + b_{shift})\\
        \gamma_{t} &= \sigma_{gamma}(h_{t} W_{gamma} + b_{gamma})

    .. math ::
        w_{t}^{c} &= softmax(\beta_{t} * K(\alpha_{t} * k_{t}, M_{t}))\\
        w_{t}^{g} &= g_{t} * w_{t}^{c} + (1 - g_{t}) * w_{t-1}\\
        \tilde{w}_{t} &= s_{t} \ast w_{t}^{g}\\
        w_{t} \propto \tilde{w}_{t}^{\gamma_{t}}

    Parameters
    ----------
    controller: a :class:`Controller` instance
        The controller of the Neural Turing Machine.
    num_shifts: int
        Number of shifts allowed by the convolutional shift operation
        (centered on 0, eg. ``num_shifts=3`` represents shifts
        in [-1, 0, 1]).
    memory_shape: tuple
        Shape of the NTM's memory
    W_hid_to_sign: callable, Numpy array, Theano shared variable or ``None``
        If callable, initializer of the weights for the parameter
        :math:`\alpha_{t}`. If ``None``, the parameter :math:`\alpha_{t}` is
        ignored (:math:`\alpha_{t} = 1`). Otherwise a matrix with shape
        ``(controller.num_units, memory_shape[1])``.
    b_hid_to_sign: callable, Numpy array, Theano shared variable or ``None``
        If callable, initializer of the biases for the parameter
        :math:`\alpha_{t}`. If ``None``, no bias. Otherwise a matrix
        with shape ``(memory_shape[1],)``.
    nonlinearity_sign: callable or ``None``
        The nonlinearity that is applied for parameter :math:`\alpha_{t}`. If
        ``None``, the nonlinearity is ``identity``.
    W_hid_to_key: callable, Numpy array or Theano shared variable
    b_hid_to_key: callable, Numpy array, Theano shared variable or ``None``
    nonlinearity_key: callable or ``None``
        Weights, biases and nonlinearity for parameter :math:`k_{t}`.
    W_hid_to_beta: callable, Numpy array or Theano shared variable
    b_hid_to_beta: callable, Numpy array, Theano shared variable or ``None``
    nonlinearity_beta: callable or ``None``
        Weights, biases and nonlinearity for parameter :math:`\beta_{t}`.
    W_hid_to_gate: callable, Numpy array or Theano shared variable
    b_hid_to_gate: callable, Numpy array, Theano shared variable or ``None``
    nonlinearity_gate: callable or ``None``
        Weights, biases and nonlinearity for parameter :math:`g_{t}`.
    W_hid_to_shift: callable, Numpy array or Theano shared variable
    b_hid_to_shift: callable, Numpy array, Theano shared variable or ``None``
    nonlinearity_shift: callable or ``None``
        Weights, biases and nonlinearity for parameter :math:`s_{t}`.
    W_hid_to_gamma: callable, Numpy array or Theano shared variable
    b_hid_to_gamma: callable, Numpy array, Theano shared variable or ``None``
    nonlinearity_gamma: callable or ``None``
        Weights, biases and nonlinearity for parameter :math:`\gamma_{t}`
    weights_init: callable, Numpy array or Theano shared variable
        Initializer for the initial weight vector (:math:`w_{0}`).
    learn_init: bool
        If ``True``, initial hidden values are learned.
    """
    def __init__(self, controller, num_shifts=3, memory_shape=(128, 20),
                 W_hid_to_sign=None,
                 b_hid_to_sign=lasagne.init.Constant(0.),
                 nonlinearity_sign=nonlinearities.ClippedLinear(low=-1., high=1.),
                 W_hid_to_key=lasagne.init.GlorotUniform(),
                 b_hid_to_key=lasagne.init.Constant(0.),
                 nonlinearity_key=nonlinearities.ClippedLinear(low=0., high=1.),
                 W_hid_to_beta=lasagne.init.GlorotUniform(),
                 b_hid_to_beta=lasagne.init.Constant(0.),
                 nonlinearity_beta=lasagne.nonlinearities.rectify,
                 W_hid_to_gate=lasagne.init.GlorotUniform(),
                 b_hid_to_gate=lasagne.init.Constant(0.),
                 nonlinearity_gate=nonlinearities.hard_sigmoid,
                 W_hid_to_shift=lasagne.init.GlorotUniform(),
                 b_hid_to_shift=lasagne.init.Constant(0.),
                 nonlinearity_shift=lasagne.nonlinearities.softmax,
                 W_hid_to_gamma=lasagne.init.GlorotUniform(),
                 b_hid_to_gamma=lasagne.init.Constant(0.),
                 nonlinearity_gamma=lambda x: 1. + lasagne.nonlinearities.rectify(x),
                 weights_init=init.OneHot(),
                 learn_init=False,
                 **kwargs):
        super(Head, self).__init__(controller, **kwargs)

        self.memory_shape = memory_shape
        self.basename = kwargs.get('name', 'head')
        self.learn_init = learn_init

        if W_hid_to_sign is not None:
            self.sign = DenseLayer(controller, num_units=self.memory_shape[1],
                W=W_hid_to_sign, b=b_hid_to_sign, nonlinearity=nonlinearity_sign,
                name=self.basename + '.sign')
            self.W_hid_to_sign, self.b_hid_to_sign = self.sign.W, self.sign.b
        else:
            self.sign = None
            self.W_hid_to_sign, self.b_hid_to_sign = None, None

        self.key = DenseLayer(controller, num_units=self.memory_shape[1],
            W=W_hid_to_key, b=b_hid_to_key, nonlinearity=nonlinearity_key,
            name=self.basename + '.key')
        self.W_hid_to_key, self.b_hid_to_key = self.key.W, self.key.b
        
        self.beta = DenseLayer(controller, num_units=1,
            W=W_hid_to_beta, b=b_hid_to_beta, nonlinearity=nonlinearity_beta,
            name=self.basename + '.beta')
        self.W_hid_to_beta, self.b_hid_to_beta = self.beta.W, self.beta.b

        self.gate = DenseLayer(controller, num_units=1,
            W=W_hid_to_gate, b=b_hid_to_gate, nonlinearity=nonlinearity_gate,
            name=self.basename + '.gate')
        self.W_hid_to_gate, self.b_hid_to_gate = self.gate.W, self.gate.b

        self.num_shifts = num_shifts
        self.shift = DenseLayer(controller, num_units=num_shifts,
            W=W_hid_to_shift, b=b_hid_to_shift, nonlinearity=nonlinearity_shift,
            name=self.basename + '.shift')
        self.W_hid_to_shift, self.b_hid_to_shift = self.shift.W, self.shift.b

        self.gamma = DenseLayer(controller, num_units=1,
            W=W_hid_to_gamma, b=b_hid_to_gamma, nonlinearity=nonlinearity_gamma,
            name=self.basename + '.gamma')
        self.W_hid_to_gamma, self.b_hid_to_gamma = self.gamma.W, self.gamma.b

        self.weights_init = self.add_param(
            weights_init, (1, self.memory_shape[0]),
            name='weights_init', trainable=learn_init, regularizable=False)

    def get_output_for(self, h_t, w_tm1, M_t, **kwargs):
        if self.sign is not None:
            sign_t = self.sign.get_output_for(h_t, **kwargs)
        else:
            sign_t = 1.
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
        w_g_padded = w_g.dimshuffle(0, 'x', 'x', 1)
        conv_filter = s_t.dimshuffle(0, 'x', 'x', 1)
        pad = (self.num_shifts // 2, (self.num_shifts - 1) // 2)
        w_g_padded = padding.pad(w_g_padded, [pad], batch_ndim=3)
        convolution = T.nnet.conv2d(w_g_padded, conv_filter,
            input_shape=(self.input_shape[0], 1, 1, self.memory_shape[0] + pad[0] + pad[1]),
            filter_shape=(self.input_shape[0], 1, 1, self.num_shifts),
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
        if self.sign is not None:
            params += self.sign.get_params(**tags)
        params += self.key.get_params(**tags)
        params += self.beta.get_params(**tags)
        params += self.gate.get_params(**tags)
        params += self.shift.get_params(**tags)
        params += self.gamma.get_params(**tags)

        return params


class WriteHead(Head):
    r"""
    Write head. In addition to the weight vector, the write head
    also outputs an add vector :math:`a_{t}` and an erase vector
    :math:`e_{t}` defined by

    .. math ::
        \delta_{t} &= \sigma_{delta}(h_{t} W_{delta} + b_{delta})\\
        a_{t} &= \delta_{t} * \sigma_{a}(h_{t} W_{a} + b_{a})
        e_{t} &= \sigma_{e}(h_{t} W_{e} + b_{e})

    Parameters
    ----------
    controller: a :class:`Controller` instance
        The controller of the Neural Turing Machine.
    num_shifts: int
        Number of shifts allowed by the convolutional shift operation
        (centered on 0, eg. ``num_shifts=3`` represents shifts
        in [-1, 0, 1]).
    memory_shape: tuple
        Shape of the NTM's memory
    W_hid_to_sign: callable, Numpy array, Theano shared variable or ``None``
    b_hid_to_sign: callable, Numpy array, Theano shared variable or ``None``
    nonlinearity_sign: callable or ``None``
        Weights, biases and nonlinearity for parameter :math:`\alpha_{t}`.
    W_hid_to_key: callable, Numpy array or Theano shared variable
    b_hid_to_key: callable, Numpy array, Theano shared variable or ``None``
    nonlinearity_key: callable or ``None``
        Weights, biases and nonlinearity for parameter :math:`k_{t}`.
    W_hid_to_beta: callable, Numpy array or Theano shared variable
    b_hid_to_beta: callable, Numpy array, Theano shared variable or ``None``
    nonlinearity_beta: callable or ``None``
        Weights, biases and nonlinearity for parameter :math:`\beta_{t}`.
    W_hid_to_gate: callable, Numpy array or Theano shared variable
    b_hid_to_gate: callable, Numpy array, Theano shared variable or ``None``
    nonlinearity_gate: callable or ``None``
        Weights, biases and nonlinearity for parameter :math:`g_{t}`.
    W_hid_to_shift: callable, Numpy array or Theano shared variable
    b_hid_to_shift: callable, Numpy array, Theano shared variable or ``None``
    nonlinearity_shift: callable or ``None``
        Weights, biases and nonlinearity for parameter :math:`s_{t}`.
    W_hid_to_gamma: callable, Numpy array or Theano shared variable
    b_hid_to_gamma: callable, Numpy array, Theano shared variable or ``None``
    nonlinearity_gamma: callable or ``None``
        Weights, biases and nonlinearity for parameter :math:`\gamma_{t}`
    W_hid_to_erase: callable, Numpy array or Theano shared variable
    b_hid_to_erase: callable, Numpy array, Theano shared variable or ``None``
    nonlinearity_erase: callable or ``None``
        Weights, biases and nonlinearity for parameter :math:`e_{t}`
    W_hid_to_add: callable, Numpy array or Theano shared variable
    b_hid_to_add: callable, Numpy array, Theano shared variable or ``None``
    nonlinearity_add: callable or ``None``
        Weights, biases and nonlinearity for parameter :math:`a_{t}`
    W_hid_to_sign_add: callable, Numpy array, Theano shared variable, or ``None``
    b_hid_to_sign_add: callable, Numpy array, Theano shared variable or ``None``
    nonlinearity_sign_add: callable or ``None``
        Weights, biases and nonlinearity for parameter :math:`\delta_{t}`
    weights_init: callable, Numpy array or Theano shared variable
        Initializer for the initial weight vector (:math:`w_{0}`).
    learn_init: bool
        If ``True``, initial hidden values are learned.
    """
    def __init__(self, controller, num_shifts=3, memory_shape=(128, 20),
                 W_hid_to_sign=None,
                 b_hid_to_sign=lasagne.init.Constant(0.),
                 nonlinearity_sign=nonlinearities.ClippedLinear(low=-1., high=1.),
                 W_hid_to_key=lasagne.init.GlorotUniform(),
                 b_hid_to_key=lasagne.init.Constant(0.),
                 nonlinearity_key=nonlinearities.ClippedLinear(low=0., high=1.),
                 W_hid_to_beta=lasagne.init.GlorotUniform(),
                 b_hid_to_beta=lasagne.init.Constant(0.),
                 nonlinearity_beta=lasagne.nonlinearities.rectify,
                 W_hid_to_gate=lasagne.init.GlorotUniform(),
                 b_hid_to_gate=lasagne.init.Constant(0.),
                 nonlinearity_gate=nonlinearities.hard_sigmoid,
                 W_hid_to_shift=lasagne.init.GlorotUniform(),
                 b_hid_to_shift=lasagne.init.Constant(0.),
                 nonlinearity_shift=lasagne.nonlinearities.softmax,
                 W_hid_to_gamma=lasagne.init.GlorotUniform(),
                 b_hid_to_gamma=lasagne.init.Constant(0.),
                 nonlinearity_gamma=lambda x: 1. + lasagne.nonlinearities.rectify(x),
                 W_hid_to_erase=lasagne.init.GlorotUniform(),
                 b_hid_to_erase=lasagne.init.Constant(0.),
                 nonlinearity_erase=nonlinearities.hard_sigmoid,
                 W_hid_to_add=lasagne.init.GlorotUniform(),
                 b_hid_to_add=lasagne.init.Constant(0.),
                 nonlinearity_add=nonlinearities.ClippedLinear(low=0., high=1.),
                 W_hid_to_sign_add=None,
                 b_hid_to_sign_add=lasagne.init.Constant(0.),
                 nonlinearity_sign_add=nonlinearities.ClippedLinear(low=-1., high=1.),
                 weights_init=init.OneHot(),
                 learn_init=False,
                 **kwargs):
        super(WriteHead, self).__init__(controller, num_shifts=num_shifts, memory_shape=memory_shape,
            W_hid_to_sign=W_hid_to_sign, b_hid_to_sign=b_hid_to_sign, nonlinearity_sign=nonlinearity_sign,
            W_hid_to_key=W_hid_to_key, b_hid_to_key=b_hid_to_key, nonlinearity_key=nonlinearity_key,
            W_hid_to_beta=W_hid_to_beta, b_hid_to_beta=b_hid_to_beta, nonlinearity_beta=nonlinearity_beta,
            W_hid_to_gate=W_hid_to_gate, b_hid_to_gate=b_hid_to_gate, nonlinearity_gate=nonlinearity_gate,
            W_hid_to_shift=W_hid_to_shift, b_hid_to_shift=b_hid_to_shift, nonlinearity_shift=nonlinearity_shift,
            W_hid_to_gamma=W_hid_to_gamma, b_hid_to_gamma=b_hid_to_gamma, nonlinearity_gamma=nonlinearity_gamma,
            weights_init=weights_init, learn_init=learn_init, **kwargs)
    
        self.erase = DenseLayer(controller, num_units=self.memory_shape[1],
            W=W_hid_to_erase, b=b_hid_to_erase, nonlinearity=nonlinearity_erase,
            name=self.basename + '.erase')
        self.W_hid_to_erase, self.b_hid_to_erase = self.erase.W, self.erase.b

        self.add = DenseLayer(controller, num_units=self.memory_shape[1],
            W=W_hid_to_add, b=b_hid_to_add, nonlinearity=nonlinearity_add,
            name=self.basename + '.add')
        self.W_hid_to_add, self.b_hid_to_add = self.add.W, self.add.b

        if W_hid_to_sign_add is not None:
            self.sign_add = DenseLayer(controller, num_units=self.memory_shape[1],
                W=W_hid_to_sign_add, b=b_hid_to_sign_add, nonlinearity=nonlinearity_sign_add,
                name=self.basename + '.sign_add')
            self.W_hid_to_sign_add, self.b_hid_to_sign_add = self.sign_add.W, self.sign_add.b
        else:
            self.sign_add = None
            self.W_hid_to_sign_add, self.b_hid_to_sign_add = None, None

    def get_params(self, **tags):
        params = super(WriteHead, self).get_params(**tags)
        params += self.erase.get_params(**tags)
        params += self.add.get_params(**tags)
        if self.sign_add is not None:
            params += self.sign_add.get_params(**tags)

        return params


class ReadHead(Head):
    r"""
    Read head.

    Parameters
    ----------
    controller: a :class:`Controller` instance
        The controller of the Neural Turing Machine.
    num_shifts: int
        Number of shifts allowed by the convolutional shift operation
        (centered on 0, eg. ``num_shifts=3`` represents shifts
        in [-1, 0, 1]).
    memory_shape: tuple
        Shape of the NTM's memory
    W_hid_to_sign: callable, Numpy array, Theano shared variable or ``None``
    b_hid_to_sign: callable, Numpy array, Theano shared variable or ``None``
    nonlinearity_sign: callable or ``None``
        Weights, biases and nonlinearity for parameter :math:`\alpha_{t}`.
    W_hid_to_key: callable, Numpy array or Theano shared variable
    b_hid_to_key: callable, Numpy array, Theano shared variable or ``None``
    nonlinearity_key: callable or ``None``
        Weights, biases and nonlinearity for parameter :math:`k_{t}`.
    W_hid_to_beta: callable, Numpy array or Theano shared variable
    b_hid_to_beta: callable, Numpy array, Theano shared variable or ``None``
    nonlinearity_beta: callable or ``None``
        Weights, biases and nonlinearity for parameter :math:`\beta_{t}`.
    W_hid_to_gate: callable, Numpy array or Theano shared variable
    b_hid_to_gate: callable, Numpy array, Theano shared variable or ``None``
    nonlinearity_gate: callable or ``None``
        Weights, biases and nonlinearity for parameter :math:`g_{t}`.
    W_hid_to_shift: callable, Numpy array or Theano shared variable
    b_hid_to_shift: callable, Numpy array, Theano shared variable or ``None``
    nonlinearity_shift: callable or ``None``
        Weights, biases and nonlinearity for parameter :math:`s_{t}`.
    W_hid_to_gamma: callable, Numpy array or Theano shared variable
    b_hid_to_gamma: callable, Numpy array, Theano shared variable or ``None``
    nonlinearity_gamma: callable or ``None``
        Weights, biases and nonlinearity for parameter :math:`\gamma_{t}`
    weights_init: callable, Numpy array or Theano shared variable
        Initializer for the initial weight vector (:math:`w_{0}`).
    learn_init: bool
        If ``True``, initial hidden values are learned.
    """
    def __init__(self, controller, num_shifts=3, memory_shape=(128, 20),
                 W_hid_to_sign=None,
                 b_hid_to_sign=lasagne.init.Constant(0.),
                 nonlinearity_sign=nonlinearities.ClippedLinear(low=-1., high=1.),
                 W_hid_to_key=lasagne.init.GlorotUniform(),
                 b_hid_to_key=lasagne.init.Constant(0.),
                 nonlinearity_key=nonlinearities.ClippedLinear(low=0., high=1.),
                 W_hid_to_beta=lasagne.init.GlorotUniform(),
                 b_hid_to_beta=lasagne.init.Constant(0.),
                 nonlinearity_beta=lasagne.nonlinearities.rectify,
                 W_hid_to_gate=lasagne.init.GlorotUniform(),
                 b_hid_to_gate=lasagne.init.Constant(0.),
                 nonlinearity_gate=T.nnet.hard_sigmoid,
                 W_hid_to_shift=lasagne.init.GlorotUniform(),
                 b_hid_to_shift=lasagne.init.Constant(0.),
                 nonlinearity_shift=lasagne.nonlinearities.softmax,
                 W_hid_to_gamma=lasagne.init.GlorotUniform(),
                 b_hid_to_gamma=lasagne.init.Constant(0.),
                 nonlinearity_gamma=lambda x: 1. + lasagne.nonlinearities.rectify(x),
                 weights_init=init.OneHot(),
                 learn_init=False,
                 **kwargs):
        super(ReadHead, self).__init__(controller, num_shifts=num_shifts, memory_shape=memory_shape,
            W_hid_to_sign=W_hid_to_sign, b_hid_to_sign=b_hid_to_sign, nonlinearity_sign=nonlinearity_sign,
            W_hid_to_key=W_hid_to_key, b_hid_to_key=b_hid_to_key, nonlinearity_key=nonlinearity_key,
            W_hid_to_beta=W_hid_to_beta, b_hid_to_beta=b_hid_to_beta, nonlinearity_beta=nonlinearity_beta,
            W_hid_to_gate=W_hid_to_gate, b_hid_to_gate=b_hid_to_gate, nonlinearity_gate=nonlinearity_gate,
            W_hid_to_shift=W_hid_to_shift, b_hid_to_shift=b_hid_to_shift, nonlinearity_shift=nonlinearity_shift,
            W_hid_to_gamma=W_hid_to_gamma, b_hid_to_gamma=b_hid_to_gamma, nonlinearity_gamma=nonlinearity_gamma,
            weights_init=weights_init, learn_init=learn_init, **kwargs)
