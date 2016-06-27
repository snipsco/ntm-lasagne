import theano
import theano.tensor as T
import numpy as np
from collections import OrderedDict

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
        k_{t} &= \sigma_{key}(h_{t} W_{key} + b_{key})\\
        \beta_{t} &= \sigma_{beta}(h_{t} W_{beta} + b_{beta})\\
        g_{t} &= \sigma_{gate}(h_{t} W_{gate} + b_{gate})\\
        s_{t} &= \sigma_{shift}(h_{t} W_{shift} + b_{shift})\\
        \gamma_{t} &= \sigma_{gamma}(h_{t} W_{gamma} + b_{gamma})

    .. math ::
        w_{t}^{c} &= softmax(\beta_{t} * K(k_{t}, M_{t}))\\
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
    W_hid_to_key: callable, Numpy array or Theano shared variable
        If callable, initializer of the weights for the parameter
        :math:`k_{t}`. Otherwise a matrix with shape
        ``(controller.num_units, memory_shape[1])``.
    b_hid_to_key: callable, Numpy array, Theano shared variable or ``None``
        If callable, initializer of the biases for the parameter
        :math:`k_{t}`. If ``None``, no bias. Otherwise a matrix
        with shape ``(memory_shape[1],)``.
    nonlinearity_key: callable or ``None``
        The nonlinearity that is applied for parameter :math:`k_{t}`. If
        ``None``, the nonlinearity is ``identity``.
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
        self.name = kwargs.get('name', 'head')
        self.learn_init = learn_init

        # Key
        self.W_hid_to_key = self.add_param(W_hid_to_key, (1, self.input_shape[1], \
            self.memory_shape[1]), name=self.name + '.key.W')
        self.b_hid_to_key = self.add_param(b_hid_to_key, (1, self.memory_shape[1]), \
            name=self.name + '.key.b', regularizable=False)
        self.nonlinearity_key = nonlinearity_key
        # Beta
        self.W_hid_to_beta = self.add_param(W_hid_to_beta, (1, self.input_shape[1], \
            1), name=self.name + '.beta.W')
        self.b_hid_to_beta = self.add_param(b_hid_to_beta, (1, 1), \
            name=self.name + '.beta.b', regularizable=False)
        self.nonlinearity_beta = nonlinearity_beta
        # Gate
        self.W_hid_to_gate = self.add_param(W_hid_to_gate, (1, self.input_shape[1], \
            1), name=self.name + '.gate.W')
        self.b_hid_to_gate = self.add_param(b_hid_to_gate, (1, 1), \
            name=self.name + '.gate.b', regularizable=False)
        self.nonlinearity_gate = nonlinearity_gate
        # Shift
        self.num_shifts = num_shifts
        self.W_hid_to_shift = self.add_param(W_hid_to_shift, (1, self.input_shape[1], \
            self.num_shifts), name=self.name + '.shift.W')
        self.b_hid_to_shift = self.add_param(b_hid_to_shift, (1, self.num_shifts), \
            name=self.name + '.shift.b', regularizable=False)
        self.nonlinearity_shift = nonlinearity_shift
        # Gamma
        self.W_hid_to_gamma = self.add_param(W_hid_to_gamma, (1, self.input_shape[1], \
            1), name=self.name + '.gamma.W')
        self.b_hid_to_gamma = self.add_param(b_hid_to_gamma, (1, 1), \
            name=self.name + '.gamma.b', regularizable=False)
        self.nonlinearity_gamma = nonlinearity_gamma

        self.weights_init = self.add_param(
            weights_init, (1, self.memory_shape[0]),
            name='weights_init', trainable=learn_init, regularizable=False)


class WriteHead(Head):
    r"""
    Write head. In addition to the weight vector, the write head
    also outputs an add vector :math:`a_{t}` and an erase vector
    :math:`e_{t}` defined by

    .. math ::
        a_{t} &= \sigma_{a}(h_{t} W_{a} + b_{a})
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
    weights_init: callable, Numpy array or Theano shared variable
        Initializer for the initial weight vector (:math:`w_{0}`).
    learn_init: bool
        If ``True``, initial hidden values are learned.
    """
    def __init__(self, controller, num_shifts=3, memory_shape=(128, 20),
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
                 weights_init=init.OneHot(),
                 learn_init=False,
                 **kwargs):
        super(WriteHead, self).__init__(controller, num_shifts=num_shifts, memory_shape=memory_shape,
            W_hid_to_key=W_hid_to_key, b_hid_to_key=b_hid_to_key, nonlinearity_key=nonlinearity_key,
            W_hid_to_beta=W_hid_to_beta, b_hid_to_beta=b_hid_to_beta, nonlinearity_beta=nonlinearity_beta,
            W_hid_to_gate=W_hid_to_gate, b_hid_to_gate=b_hid_to_gate, nonlinearity_gate=nonlinearity_gate,
            W_hid_to_shift=W_hid_to_shift, b_hid_to_shift=b_hid_to_shift, nonlinearity_shift=nonlinearity_shift,
            W_hid_to_gamma=W_hid_to_gamma, b_hid_to_gamma=b_hid_to_gamma, nonlinearity_gamma=nonlinearity_gamma,
            weights_init=weights_init, learn_init=learn_init, **kwargs)
        # Erase
        self.W_hid_to_erase = self.add_param(W_hid_to_erase, (1, self.input_shape[1], \
            self.memory_shape[1]), name=self.name + '.erase.W')
        self.b_hid_to_erase = self.add_param(b_hid_to_erase, (1, self.memory_shape[1]), \
            name=self.name + '.erase.b', regularizable=False)
        self.nonlinearity_erase = nonlinearity_erase
        # Add
        self.W_hid_to_add = self.add_param(W_hid_to_add, (1, self.input_shape[1], \
            self.memory_shape[1]), name=self.name + '.add.W')
        self.b_hid_to_add = self.add_param(b_hid_to_add, (1, self.memory_shape[1]), \
            name=self.name + '.add.b', regularizable=False)
        self.nonlinearity_add = nonlinearity_add


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
            W_hid_to_key=W_hid_to_key, b_hid_to_key=b_hid_to_key, nonlinearity_key=nonlinearity_key,
            W_hid_to_beta=W_hid_to_beta, b_hid_to_beta=b_hid_to_beta, nonlinearity_beta=nonlinearity_beta,
            W_hid_to_gate=W_hid_to_gate, b_hid_to_gate=b_hid_to_gate, nonlinearity_gate=nonlinearity_gate,
            W_hid_to_shift=W_hid_to_shift, b_hid_to_shift=b_hid_to_shift, nonlinearity_shift=nonlinearity_shift,
            W_hid_to_gamma=W_hid_to_gamma, b_hid_to_gamma=b_hid_to_gamma, nonlinearity_gamma=nonlinearity_gamma,
            weights_init=weights_init, learn_init=learn_init, **kwargs)


class HeadCollection(object):
    r"""
    The base class :class:`HeadCollection` represents a generic collection 
    of heads. Each head is an instance of :class:`Head`. This allows to 
    process the heads simultaneously if they have the same type. This should 
    be limited to internal uses only.

    Parameters
    ----------
    heads: a list of :class:`Head` instances
        List of the heads.
    """
    def __init__(self, heads):
        self.heads = heads
        # QKFIX: Assume that all the heads have the same number of shifts and nonlinearities
        self.memory_shape = self.heads[0].memory_shape
        self.num_shifts = self.heads[0].num_shifts
        # Key
        self.W_hid_to_key = T.concatenate([head.W_hid_to_key for head in self.heads], axis=0)
        self.b_hid_to_key = T.concatenate([head.b_hid_to_key for head in self.heads], axis=0)
        self.nonlinearity_key = self.heads[0].nonlinearity_key
        # Beta
        self.W_hid_to_beta = T.concatenate([head.W_hid_to_beta for head in self.heads], axis=0)
        self.b_hid_to_beta = T.concatenate([head.b_hid_to_beta for head in self.heads], axis=0)
        self.nonlinearity_beta = self.heads[0].nonlinearity_beta
        # Gate
        self.W_hid_to_gate = T.concatenate([head.W_hid_to_gate for head in self.heads], axis=0)
        self.b_hid_to_gate = T.concatenate([head.b_hid_to_gate for head in self.heads], axis=0)
        self.nonlinearity_gate = self.heads[0].nonlinearity_gate
        # Shift
        self.W_hid_to_shift = T.concatenate([head.W_hid_to_shift for head in self.heads], axis=0)
        self.b_hid_to_shift = T.concatenate([head.b_hid_to_shift for head in self.heads], axis=0)
        self.nonlinearity_shift = self.heads[0].nonlinearity_shift
        # Gamma
        self.W_hid_to_gamma = T.concatenate([head.W_hid_to_gamma for head in self.heads], axis=0)
        self.b_hid_to_gamma = T.concatenate([head.b_hid_to_gamma for head in self.heads], axis=0)
        self.nonlinearity_gamma = self.heads[0].nonlinearity_gamma
        # Initialization
        self.weights_init = T.concatenate([head.weights_init for head in self.heads], axis=0)

    def get_params(self, **tags):
        params = []
        for head in self.heads:
            params += head.get_params(**tags)

        return params

    def get_weights(self, h_t, w_tm1, M_t, **kwargs):
        batch_size = self.heads[0].input_shape[0] # QKFIX: Get the size of the batches from the 1st head
        num_heads = len(self.heads)
        k_t = self.nonlinearity_key(T.dot(h_t, self.W_hid_to_key) + self.b_hid_to_key)
        beta_t = self.nonlinearity_beta(T.dot(h_t, self.W_hid_to_beta) + self.b_hid_to_beta)
        g_t = self.nonlinearity_gate(T.dot(h_t, self.W_hid_to_gate) + self.b_hid_to_gate)
        # QKFIX: If the nonlinearity is softmax (which is usually the case), then the activations
        # need to be reshaped (T.nnet.softmax only accepts 2D inputs)
        try:
            s_t = self.nonlinearity_shift(T.dot(h_t, self.W_hid_to_shift) + self.b_hid_to_shift)
        except ValueError:
            shift_activation_t = T.dot(h_t, self.W_hid_to_shift) + self.b_hid_to_shift
            s_t = self.nonlinearity_shift(shift_activation_t.reshape((h_t.shape[0] * num_heads, self.num_shifts)))
            s_t = s_t.reshape(shift_activation_t.shape)
        gamma_t = self.nonlinearity_gamma(T.dot(h_t, self.W_hid_to_gamma) + self.b_hid_to_gamma)

        # Content Addressing (3.3.1)
        beta_t = T.addbroadcast(beta_t, 2)
        betaK = beta_t * similarities.cosine_similarity(k_t, M_t)
        w_c = lasagne.nonlinearities.softmax(betaK.flatten(ndim=2))
        w_c = w_c.reshape(betaK.shape)

        # Interpolation (3.3.2)
        g_t = T.addbroadcast(g_t, 2)
        w_g = g_t * w_c + (1. - g_t) * w_tm1

        # Convolutional Shift (3.3.2)
        # NOTE: This library is using a flat (zero-padded) convolution instead of the circular
        # convolution from the original paper. In practice, this change has a minimal impact.
        w_g_padded = w_g.reshape((h_t.shape[0] * num_heads, self.memory_shape[0])).dimshuffle(0, 'x', 'x', 1)
        conv_filter = s_t.reshape((h_t.shape[0] * num_heads, self.num_shifts)).dimshuffle(0, 'x', 'x', 1)
        pad = (self.num_shifts // 2, (self.num_shifts - 1) // 2)
        w_g_padded = padding.pad(w_g_padded, [pad], batch_ndim=3)
        convolution = T.nnet.conv2d(w_g_padded, conv_filter,
            input_shape=(None if batch_size is None else \
                batch_size * num_heads, 1, 1, self.memory_shape[0] + pad[0] + pad[1]),
            filter_shape=(None if batch_size is None else \
                batch_size * num_heads, 1, 1, self.num_shifts),
            subsample=(1, 1),
            border_mode='valid')
        w_tilde = convolution[T.arange(h_t.shape[0] * num_heads), T.arange(h_t.shape[0] * num_heads), 0, :]
        w_tilde = w_tilde.reshape((h_t.shape[0], num_heads, self.memory_shape[0]))

        # Sharpening (3.3.2)
        gamma_t = T.addbroadcast(gamma_t, 2)
        w = T.pow(w_tilde + 1e-6, gamma_t)
        w /= T.sum(w, axis=2).dimshuffle(0, 1, 'x')

        return w


class ReadHeadCollection(HeadCollection):
    r"""
    Collection of read heads.

    Parameters
    ----------
    heads: a list of :class:`ReadHead` instances
        List of the read heads.
    """
    def __init__(self, heads):
        assert all([isinstance(head, ReadHead) for head in heads])
        super(ReadHeadCollection, self).__init__(heads=heads)

    def read(self, w_tm1, M_t, **kwargs):
        r_t = T.batched_dot(w_tm1, M_t)

        return r_t.flatten(ndim=2)


class WriteHeadCollection(HeadCollection):
    r"""
    Collection of write heads.

    Parameters
    ----------
    heads: a list of :class:`WriteHead` instances
        List of the write heads.
    """
    def __init__(self, heads):
        assert all([isinstance(head, WriteHead) for head in heads])
        super(WriteHeadCollection, self).__init__(heads=heads)
        # Erase
        self.W_hid_to_erase = T.concatenate([head.W_hid_to_erase for head in self.heads], axis=0)
        self.b_hid_to_erase = T.concatenate([head.b_hid_to_erase for head in self.heads], axis=0)
        self.nonlinearity_erase = self.heads[0].nonlinearity_erase
        # Add
        self.W_hid_to_add = T.concatenate([head.W_hid_to_add for head in self.heads], axis=0)
        self.b_hid_to_add = T.concatenate([head.b_hid_to_add for head in self.heads], axis=0)
        self.nonlinearity_add = self.heads[0].nonlinearity_add

    def write(self, h_tm1, w_tm1, M_tm1, **kwargs):
        e_t = self.nonlinearity_erase(T.dot(h_tm1, self.W_hid_to_erase) + self.b_hid_to_erase)
        a_t = self.nonlinearity_add(T.dot(h_tm1, self.W_hid_to_add) + self.b_hid_to_add)
        # Erase
        M_tp1 = M_tm1 * T.prod(1 - w_tm1.dimshuffle(0, 1, 2, 'x') * e_t.dimshuffle(0, 1, 'x', 2), axis=1)
        # Add
        M_tp1 += T.sum(w_tm1.dimshuffle(0, 1, 2, 'x') * a_t.dimshuffle(0, 1, 'x', 2), axis=1)

        return M_tp1
