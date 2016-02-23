import theano
import theano.tensor as T
import numpy as np

from lasagne.layers import Layer
from lasagne.layers.recurrent import Gate
import lasagne.nonlinearities
import lasagne.init


class Controller(Layer):
    r"""
    The base class :class:`Controller` represents a generic controller
    for the Neural Turing Machine. The controller is a neural network
    (feed-forward or recurrent) making the interface between the
    incoming layer (eg. an instance of :class:`lasagne.layers.InputLayer`)
    and the NTM.

    Parameters
    ----------
    incoming: a :class:`lasagne.layers.Layer` instance
        The layer feeding into the Neural Turing Machine.
    memory_shape: tuple
        Shape of the NTM's memory.
    num_units: int
        Number of hidden units in the controller.
    num_reads: int
        Number of read heads in the Neural Turing Machine.
    hid_init: callable, Numpy array or Theano shared variable
        Initializer for the initial hidden state (:math:`h_{0}`).
    learn_init: bool
        If ``True``, initial hidden values are learned.
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
        raise NotImplementedError

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.num_units)


class DenseController(Controller):
    r"""
    A fully connected (feed-forward) controller for the NTM.

    .. math ::
        h_t = \sigma(x_{t} W_{x} + r_{t} W_{r} + b_{x} + b_{r})

    Parameters
    ----------
    incoming: a :class:`lasagne.layers.Layer` instance
        The layer feeding into the Neural Turing Machine.
    memory_shape: tuple
        Shape of the NTM's memory.
    num_units: int
        Number of hidden units in the controller.
    num_reads: int
        Number of read heads in the Neural Turing Machine.
    W_in_to_hid: callable, Numpy array or Theano shared variable
        If callable, initializer for the weights between the
        input and the hidden state. Otherwise a matrix with
        shape ``(num_inputs, num_units)`` (:math:`W_{x}`).
    b_in_to_hid: callable, Numpy array, Theano shared variable or ``None``
        If callable, initializer for the biases between the
        input and the hidden state. If ``None``, the controller
        has no bias between the input and the hidden state. Otherwise
        a 1D array with shape ``(num_units,)`` (:math:`b_{x}`).
    W_reads_to_hid: callable, Numpy array or Theano shared variable
        If callable, initializer for the weights between the
        read vector and the hidden state. Otherwise a matrix with
        shape ``(num_reads * memory_shape[1], num_units)`` (:math:`W_{r}`).
    b_reads_to_hid: callable, Numpy array, Theano shared variable or ``None``
        If callable, initializer for the biases between the
        read vector and the hidden state. If ``None``, the controller
        has no bias between the read vector and the hidden state.
        Otherwise a 1D array with shape ``(num_units,)`` (:math:`b_{r}`).
    nonlinearity: callable or ``None``
        The nonlinearity that is applied to the controller. If ``None``,
        the controller will be linear (:math:`\sigma`).
    hid_init: callable, np.ndarray or theano.shared
        Initializer for the initial hidden state (:math:`h_{0}`).
    learn_init: bool
        If ``True``, initial hidden values are learned.
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
            activation += self.b_in_to_hid.dimshuffle('x', 0)
        if self.b_reads_to_hid is not None:
            activation += self.b_reads_to_hid.dimshuffle('x', 0)
        state = self.nonlinearity(activation)
        return state, state

    @property
    def outputs_info(self):
        ones_vector = T.ones((self.input_shape[0], 1))
        hid_init = T.dot(ones_vector, self.hid_init)
        hid_init = T.unbroadcast(hid_init, 0)
        return [hid_init, hid_init]


class RecurrentController(Controller):
    r"""
    A "vanilla" recurrent controller for the NTM.

    .. math ::
        h_t = \sigma(x_{t} W_{x} + r_{t} W_{r} +
              h_{t-1} W_{h} + b_{x} + b_{r} + b_{h})

    Parameters
    ----------
    incoming: a :class:`lasagne.layers.Layer` instance
        The layer feeding into the Neural Turing Machine.
    memory_shape: tuple
        Shape of the NTM's memory.
    num_units: int
        Number of hidden units in the controller.
    num_reads: int
        Number of read heads in the Neural Turing Machine.
    W_in_to_hid: callable, Numpy array or Theano shared variable
        If callable, initializer for the weights between the
        input and the hidden state. Otherwise a matrix with
        shape ``(num_inputs, num_units)`` (:math:`W_{x}`).
    b_in_to_hid: callable, Numpy array, Theano shared variable or ``None``
        If callable, initializer for the biases between the
        input and the hidden state. If ``None``, the controller
        has no bias between the input and the hidden state. Otherwise
        a 1D array with shape ``(num_units,)`` (:math:`b_{x}`).
    W_reads_to_hid: callable, Numpy array or Theano shared variable
        If callable, initializer for the weights between the
        read vector and the hidden state. Otherwise a matrix with
        shape ``(num_reads * memory_shape[1], num_units)`` (:math:`W_{r}`).
    b_reads_to_hid: callable, Numpy array, Theano shared variable or ``None``
        If callable, initializer for the biases between the
        read vector and the hidden state. If ``None``, the controller
        has no bias between the read vector and the hidden state.
        Otherwise a 1D array with shape ``(num_units,)`` (:math:`b_{r}`).
    W_hid_to_hid: callable, Numpy array or Theano shared variable
        If callable, initializer for the weights in the hidden-to-hidden
        update. Otherwise a matrix with shape ``(num_units, num_units)``
        (:math:`W_{h}`).
    b_hid_to_hid: callable, Numpy array, Theano shared variable or ``None``
        If callable, initializer for the biases in the hidden-to-hidden
        update. If ``None``, the controller has no bias in the
        hidden-to-hidden update. Otherwise a 1D array with shape
        ``(num_units,)`` (:math:`b_{h}`).
    nonlinearity: callable or ``None``
        The nonlinearity that is applied to the controller. If ``None``,
        the controller will be linear (:math:`\sigma`).
    hid_init: callable, np.ndarray or theano.shared
        Initializer for the initial hidden state (:math:`h_{0}`).
    learn_init: bool
        If ``True``, initial hidden values are learned.
    """
    def __init__(self, incoming, memory_shape, num_units, num_reads,
                 W_in_to_hid=lasagne.init.GlorotUniform(),
                 b_in_to_hid=lasagne.init.Constant(0.),
                 W_reads_to_hid=lasagne.init.GlorotUniform(),
                 b_reads_to_hid=lasagne.init.Constant(0.),
                 W_hid_to_hid=lasagne.init.GlorotUniform(),
                 b_hid_to_hid=lasagne.init.Constant(0.),
                 nonlinearity=lasagne.nonlinearities.rectify,
                 hid_init=lasagne.init.GlorotUniform(),
                 learn_init=False,
                 **kwargs):
        super(RecurrentController, self).__init__(incoming, memory_shape, num_units,
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
        # Hidden / Hidden parameters
        self.W_hid_to_hid, self.b_hid_to_hid = add_weight_and_bias_params(self.num_units,
            W_hid_to_hid, b_hid_to_hid, name='hid_to_hid')

    def step(self, input, reads, hidden, *args):
        if input.ndim > 2:
            input = input.flatten(2)
        if reads.ndim > 2:
            reads = reads.flatten(2)

        activation = T.dot(input, self.W_in_to_hid) + \
                     T.dot(reads, self.W_reads_to_hid) + \
                     T.dot(hidden, self.W_hid_to_hid)
        if self.b_in_to_hid is not None:
            activation += self.b_in_to_hid.dimshuffle('x', 0)
        if self.b_reads_to_hid is not None:
            activation += self.b_reads_to_hid.dimshuffle('x', 0)
        if self.b_hid_to_hid is not None:
            activation += self.b_hid_to_hid.dimshuffle('x', 0)
        state = self.nonlinearity(activation)
        return state, state

    @property
    def outputs_info(self):
        ones_vector = T.ones((self.input_shape[0], 1))
        hid_init = T.dot(ones_vector, self.hid_init)
        hid_init = T.unbroadcast(hid_init, 0)
        return [hid_init, hid_init]
