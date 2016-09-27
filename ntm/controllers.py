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

    def outputs_info(self, batch_size):
        ones_vector = T.ones((batch_size, 1))
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

    def outputs_info(self, batch_size):
        ones_vector = T.ones((batch_size, 1))
        hid_init = T.dot(ones_vector, self.hid_init)
        hid_init = T.unbroadcast(hid_init, 0)
        return [hid_init, hid_init]

class LSTMController(Controller):
    r"""
    A LSTM recurrent controller for the NTM.
    .. math ::
        input-gate = \sigma(x_{t} Wi_{x} + r_{t} Wi_{r} +
              h_{t-1} Wi_{h} + bi_{x} + bi_{r} + bi_{h})
        forget-gate = \sigma(x_{t} Wf_{x} + r_{t} Wf_{r} +
              h_{t-1} Wf_{h} + bf_{x} + bf_{r} + bf_{h})
        output-gate = \sigma(x_{t} Wo_{x} + r_{t} Wo_{r} +
              h_{t-1} Wo_{h} + bo_{x} + bo_{r} + bo_{h})
        candidate-cell-state = \tanh(x_{t} Wc_{x} + r_{t} Wc_{r} +
              h_{t-1} Wc_{h} + bc_{x} + bc_{r} + bc_{h})
        cell-state_{t} = cell-state_{t-1} \odot forget-gate +
              candidate-cell-state \odot input-gate
        h_{t} = \tanh(cell-state_{t}) \odot output-gate
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
    W_in_to_input: callable, Numpy array or Theano shared variable
        If callable, initializer for the weights between the
        input and the input gate. Otherwise a matrix with
        shape ``(num_inputs, num_units)`` (:math:`Wi_{x}`).
    b_in_to_input: callable, Numpy array, Theano shared variable or ``None``
        If callable, initializer for the biases between the
        input and the input gate. If ``None``, the controller
        has no bias between the input and the input gate. Otherwise
        a 1D array with shape ``(num_units,)`` (:math:`bi_{x}`).
    W_reads_to_input: callable, Numpy array or Theano shared variable
        If callable, initializer for the weights between the
        read vector and the input gate. Otherwise a matrix with
        shape ``(num_reads * memory_shape[1], num_units)`` (:math:`Wi_{r}`).
    b_reads_to_input: callable, Numpy array, Theano shared variable or ``None``
        If callable, initializer for the biases between the
        read vector and the input gate. If ``None``, the controller
        has no bias between the read vector and the input gate.
        Otherwise a 1D array with shape ``(num_units,)`` (:math:`bi_{r}`).
    W_hid_to_input: callable, Numpy array or Theano shared variable
        If callable, initializer for the weights between the
        hidden state and the input gate. Otherwise a matrix with
        shape ``(num_units, num_units)`` (:math:`Wi_{h}`).
    b_hid_to_input: callable, Numpy array, Theano shared variable or ``None``
        If callable, initializer for the biases between the
        hidden state and the input gate. If ``None``, the controller
        has no bias between the hidden state and the input gate.
        Otherwise a 1D array with shape ``(num_units,)`` (:math:`bi_{h}`).
    W_in_to_forget: callable, Numpy array or Theano shared variable
        If callable, initializer for the weights between the
        input and the forget gate. Otherwise a matrix with
        shape ``(num_inputs, num_units)`` (:math:`Wf_{x}`).
    b_in_to_forget: callable, Numpy array, Theano shared variable or ``None``
        If callable, initializer for the biases between the
        input and the forget gate. If ``None``, the controller
        has no bias between the input and the forget gate. Otherwise
        a 1D array with shape ``(num_units,)`` (:math:`bf_{x}`).
    W_reads_to_forget: callable, Numpy array or Theano shared variable
        If callable, initializer for the weights between the
        read vector and the forget gate. Otherwise a matrix with
        shape ``(num_reads * memory_shape[1], num_units)`` (:math:`Wf_{r}`).
    b_reads_to_forget: callable, Numpy array, Theano shared variable or ``None``
        If callable, initializer for the biases between the
        read vector and the forget gate. If ``None``, the controller
        has no bias between the read vector and the forget gate.
        Otherwise a 1D array with shape ``(num_units,)`` (:math:`bf_{r}`).
    W_hid_to_forget: callable, Numpy array or Theano shared variable
        If callable, initializer for the weights between the
        hidden state and the forget gate. Otherwise a matrix with
        shape ``(num_units, num_units)`` (:math:`Wf_{h}`).
    b_hid_to_forget: callable, Numpy array, Theano shared variable or ``None``
        If callable, initializer for the biases between the
        hidden state and the forget gate. If ``None``, the controller
        has no bias between the hidden state and the forget gate.
        Otherwise a 1D array with shape ``(num_units,)`` (:math:`bf_{h}`).
    W_in_to_output: callable, Numpy array or Theano shared variable
        If callable, initializer for the weights between the
        input and the output gate. Otherwise a matrix with
        shape ``(num_inputs, num_units)`` (:math:`Wo_{x}`).
    b_in_to_output: callable, Numpy array, Theano shared variable or ``None``
        If callable, initializer for the biases between the
        input and the output gate. If ``None``, the controller
        has no bias between the input and the output gate. Otherwise
        a 1D array with shape ``(num_units,)`` (:math:`bo_{x}`).
    W_reads_to_output: callable, Numpy array or Theano shared variable
        If callable, initializer for the weights between the
        read vector and the output gate. Otherwise a matrix with
        shape ``(num_reads * memory_shape[1], num_units)`` (:math:`Wo_{r}`).
    b_reads_to_output: callable, Numpy array, Theano shared variable or ``None``
        If callable, initializer for the biases between the
        read vector and the output gate. If ``None``, the controller
        has no bias between the read vector and the output gate.
        Otherwise a 1D array with shape ``(num_units,)`` (:math:`bo_{r}`).
    W_hid_to_output: callable, Numpy array or Theano shared variable
        If callable, initializer for the weights between the
        hidden state and the output gate. Otherwise a matrix with
        shape ``(num_units, num_units)`` (:math:`Wo_{h}`).
    b_hid_to_output: callable, Numpy array, Theano shared variable or ``None``
        If callable, initializer for the biases between the
        hidden state and the output gate. If ``None``, the controller
        has no bias between the hidden state and the output gate.
        Otherwise a 1D array with shape ``(num_units,)`` (:math:`bo_{h}`).
    W_in_to_cell: callable, Numpy array or Theano shared variable
        If callable, initializer for the weights between the
        input and the cell state computation gate. Otherwise a matrix
        with shape ``(num_inputs, num_units)`` (:math:`Wc_{x}`).
    b_in_to_cell: callable, Numpy array, Theano shared variable or ``None``
        If callable, initializer for the biases between the
        input and the cell state computation gate. If ``None``,
        the controller has no bias between the input and the cell
        state computation gate. Otherwise a 1D array with shape
        ``(num_units,)`` (:math:`bc_{x}`).
    W_reads_to_cell: callable, Numpy array or Theano shared variable
        If callable, initializer for the weights between the
        read vector and the cell state computation gate. Otherwise a matrix
        with shape ``(num_reads * memory_shape[1], num_units)`` (:math:`Wc_{r}`).
    b_reads_to_cell: callable, Numpy array, Theano shared variable or ``None``
        If callable, initializer for the biases between the
        read vector and the cell state computation gate. If ``None``,
        the controller has no bias between the read vector and the cell
        state computation gate. Otherwise a 1D array with shape
        ``(num_units,)`` (:math:`bc_{r}`).
    W_hid_to_cell: callable, Numpy array or Theano shared variable
        If callable, initializer for the weights between the
        hidden state and the cell state computation gate. Otherwise a matrix
        with shape ``(num_units, num_units)`` (:math:`Wc_{h}`).
    b_hid_to_cell: callable, Numpy array, Theano shared variable or ``None``
        If callable, initializer for the biases between the
        hidden state and the cell state computation gate. If ``None``,
        the controller has no bias between the hidden state and the cell
        state computation gate. Otherwise a 1D array with shape
        ``(num_units,)`` (:math:`bc_{h}`).
    hid_init: callable, np.ndarray or theano.shared
        Initializer for the initial hidden state (:math:`h_{0}`).
    cell_init: callable, np.ndarray or theano.shared
        Initializer for the initial cell state (:math:`cell-state_{0}`).
    learn_init: bool
        If ``True``, initial hidden values are learned.
    """
    def __init__(self, incoming, memory_shape, num_units, num_reads,
                 W_in_to_input=lasagne.init.GlorotUniform(),
                 b_in_to_input=lasagne.init.Constant(0.),
                 W_reads_to_input=lasagne.init.GlorotUniform(),
                 b_reads_to_input=lasagne.init.Constant(0.),
                 W_hid_to_input=lasagne.init.GlorotUniform(),
                 b_hid_to_input=lasagne.init.Constant(0.),
                 W_in_to_forget=lasagne.init.GlorotUniform(),
                 b_in_to_forget=lasagne.init.Constant(0.),
                 W_reads_to_forget=lasagne.init.GlorotUniform(),
                 b_reads_to_forget=lasagne.init.Constant(0.),
                 W_hid_to_forget=lasagne.init.GlorotUniform(),
                 b_hid_to_forget=lasagne.init.Constant(0.),
                 W_in_to_output=lasagne.init.GlorotUniform(),
                 b_in_to_output=lasagne.init.Constant(0.),
                 W_reads_to_output=lasagne.init.GlorotUniform(),
                 b_reads_to_output=lasagne.init.Constant(0.),
                 W_hid_to_output=lasagne.init.GlorotUniform(),
                 b_hid_to_output=lasagne.init.Constant(0.),
                 W_in_to_cell=lasagne.init.GlorotUniform(),
                 b_in_to_cell=lasagne.init.Constant(0.),
                 W_reads_to_cell=lasagne.init.GlorotUniform(),
                 b_reads_to_cell=lasagne.init.Constant(0.),
                 W_hid_to_cell=lasagne.init.GlorotUniform(),
                 b_hid_to_cell=lasagne.init.Constant(0.),
                 nonlinearity=lasagne.nonlinearities.rectify,
                 hid_init=lasagne.init.GlorotUniform(),
                 cell_init=lasagne.init.Constant(0.),
                 learn_init=False,
                 **kwargs):
        super(LSTMController, self).__init__(incoming, memory_shape, num_units,
                                              num_reads, hid_init, learn_init,
                                              **kwargs)
        self.nonlinearity = (lasagne.nonlinearities.identity if
                             nonlinearity is None else nonlinearity)
        self.cell_init = self.add_param(cell_init, (1, num_units),
            name='cell_init', regularizable=False, trainable=learn_init)

        def add_weight_and_bias_params(input_dim, W, b, name):
            return (self.add_param(W, (input_dim, self.num_units),
                name='W_{}'.format(name)),
                self.add_param(b, (self.num_units,),
                name='b_{}'.format(name)) if b is not None else None)
        num_inputs = int(np.prod(self.input_shape[2:]))
        # Inputs / Input Gate parameters
        self.W_in_to_input, self.b_in_to_input = add_weight_and_bias_params(num_inputs,
            W_in_to_input, b_in_to_input, name='in_to_input')
        # Read vectors / Input Gate parameters
        self.W_reads_to_input, self.b_reads_to_input = add_weight_and_bias_params(self.num_reads * self.memory_shape[1],
            W_reads_to_input, b_reads_to_input, name='reads_to_input')
        # Hidden / Input Gate parameters
        self.W_hid_to_input, self.b_hid_to_input = add_weight_and_bias_params(self.num_units,
            W_hid_to_input, b_hid_to_input, name='hid_to_input')
        # Inputs / Forget Gate parameters
        self.W_in_to_forget, self.b_in_to_forget = add_weight_and_bias_params(num_inputs,
            W_in_to_forget, b_in_to_forget, name='in_to_forget')
        # Read vectors / Forget Gate parameters
        self.W_reads_to_forget, self.b_reads_to_forget = add_weight_and_bias_params(self.num_reads * self.memory_shape[1],
            W_reads_to_forget, b_reads_to_forget, name='reads_to_forget')
        # Hidden / Forget Gate parameters
        self.W_hid_to_forget, self.b_hid_to_forget = add_weight_and_bias_params(self.num_units,
            W_hid_to_forget, b_hid_to_forget, name='hid_to_forget')
        # Inputs / Output Gate parameters
        self.W_in_to_output, self.b_in_to_output = add_weight_and_bias_params(num_inputs,
            W_in_to_output, b_in_to_output, name='in_to_output')
        # Read vectors / Output Gate parameters
        self.W_reads_to_output, self.b_reads_to_output = add_weight_and_bias_params(self.num_reads * self.memory_shape[1],
            W_reads_to_output, b_reads_to_output, name='reads_to_output')
        # Hidden / Output Gate parameters
        self.W_hid_to_output, self.b_hid_to_output = add_weight_and_bias_params(self.num_units,
            W_hid_to_output, b_hid_to_output, name='hid_to_output')
        # Inputs / Cell State parameters
        self.W_in_to_cell, self.b_in_to_cell = add_weight_and_bias_params(num_inputs,
            W_in_to_cell, b_in_to_cell, name='in_to_cell')
        # Read vectors / Cell State parameters
        self.W_reads_to_cell, self.b_reads_to_cell = add_weight_and_bias_params(self.num_reads * self.memory_shape[1],
            W_reads_to_cell, b_reads_to_cell, name='reads_to_cell')
        # Hidden / Cell State parameters
        self.W_hid_to_cell, self.b_hid_to_cell = add_weight_and_bias_params(self.num_units,
            W_hid_to_cell, b_hid_to_cell, name='hid_to_cell')

    def step(self, input, reads, hidden, cell, *args):
        if input.ndim > 2:
            input = input.flatten(2)
        if reads.ndim > 2:
            reads = reads.flatten(2)
        # Input Gate output computation
        activation = T.dot(input, self.W_in_to_input) + \
                     T.dot(reads, self.W_reads_to_input) + \
                     T.dot(hidden, self.W_hid_to_input)
        if self.b_in_to_input is not None:
            activation += self.b_in_to_input.dimshuffle('x', 0)
        if self.b_reads_to_input is not None:
            activation += self.b_reads_to_input.dimshuffle('x', 0)
        if self.b_hid_to_input is not None:
            activation += self.b_hid_to_input.dimshuffle('x', 0)
        input_gate = lasagne.nonlinearities.sigmoid(activation)
        # Forget Gate output computation
        activation = T.dot(input, self.W_in_to_forget) + \
                     T.dot(reads, self.W_reads_to_forget) + \
                     T.dot(hidden, self.W_hid_to_forget)
        if self.b_in_to_forget is not None:
            activation += self.b_in_to_forget.dimshuffle('x', 0)
        if self.b_reads_to_forget is not None:
            activation += self.b_reads_to_forget.dimshuffle('x', 0)
        if self.b_hid_to_forget is not None:
            activation += self.b_hid_to_forget.dimshuffle('x', 0)
        forget_gate = lasagne.nonlinearities.sigmoid(activation)
        # Output Gate output computation
        activation = T.dot(input, self.W_in_to_output) + \
                     T.dot(reads, self.W_reads_to_output) + \
                     T.dot(hidden, self.W_hid_to_output)
        if self.b_in_to_output is not None:
            activation += self.b_in_to_output.dimshuffle('x', 0)
        if self.b_reads_to_output is not None:
            activation += self.b_reads_to_output.dimshuffle('x', 0)
        if self.b_hid_to_output is not None:
            activation += self.b_hid_to_output.dimshuffle('x', 0)
        output_gate = lasagne.nonlinearities.sigmoid(activation)
        # New candidate cell state computation
        activation = T.dot(input, self.W_in_to_cell) + \
                     T.dot(reads, self.W_reads_to_cell) + \
                     T.dot(hidden, self.W_hid_to_cell)
        if self.b_in_to_cell is not None:
            activation += self.b_in_to_cell.dimshuffle('x', 0)
        if self.b_reads_to_cell is not None:
            activation += self.b_reads_to_cell.dimshuffle('x', 0)
        if self.b_hid_to_cell is not None:
            activation += self.b_hid_to_cell.dimshuffle('x', 0)
        candidate_cell_state = lasagne.nonlinearities.tanh(activation)
        # New cell state and hidden state computation
        cell_state = cell * forget_gate + candidate_cell_state * input_gate
        state = lasagne.nonlinearities.tanh(cell_state) * output_gate
        return state, cell_state

    def outputs_info(self, batch_size):
        ones_vector = T.ones((batch_size, 1))
        hid_init = T.dot(ones_vector, self.hid_init)
        hid_init = T.unbroadcast(hid_init, 0)
        cell_init = T.dot(ones_vector, self.cell_init)
        cell_init = T.unbroadcast(cell_init, 0)
        return [hid_init, cell_init]

class GRUController(Controller):
    r"""
    A GRU recurrent controller for the NTM.
    .. math ::
        update-gate = \sigma(x_{t} Wz_{x} + r_{t} Wz_{r} +
              h_{t-1} Wz_{h} + bz_{x} + bz_{r} + bz_{h})
        reset-gate = \sigma(x_{t} Wr_{x} + r_{t} Wr_{r} +
              h_{t-1} Wr_{h} + br_{x} + br_{r} + br_{h})
        s = \tanh(x_{t} Ws_{x} + r_{t} Ws_{r} +
              (h_{t-1} \odot reset-gate) Ws_{h})
        h_{t} = (1 - update-gate) \odot s + update-gate \odot h_{t-1}
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
    W_in_to_input: callable, Numpy array or Theano shared variable
        If callable, initializer for the weights between the
        input and the update gate. Otherwise a matrix with
        shape ``(num_inputs, num_units)`` (:math:`Wz_{x}`).
    b_in_to_update: callable, Numpy array, Theano shared variable or ``None``
        If callable, initializer for the biases between the
        input and the update gate. If ``None``, the controller
        has no bias between the input and the update gate. Otherwise
        a 1D array with shape ``(num_units,)`` (:math:`bz_{x}`).
    W_reads_to_update: callable, Numpy array or Theano shared variable
        If callable, initializer for the weights between the
        read vector and the update gate. Otherwise a matrix with
        shape ``(num_reads * memory_shape[1], num_units)`` (:math:`Wz_{r}`).
    b_reads_to_update: callable, Numpy array, Theano shared variable or ``None``
        If callable, initializer for the biases between the
        read vector and the update gate. If ``None``, the controller
        has no bias between the read vector and the update gate.
        Otherwise a 1D array with shape ``(num_units,)`` (:math:`bz_{r}`).
    W_hid_to_update: callable, Numpy array or Theano shared variable
        If callable, initializer for the weights between the
        hidden state and the update gate. Otherwise a matrix with
        shape ``(num_units, num_units)`` (:math:`Wz_{h}`).
    b_hid_to_update: callable, Numpy array, Theano shared variable or ``None``
        If callable, initializer for the biases between the
        hidden state and the update gate. If ``None``, the controller
        has no bias between the hidden state and the update gate.
        Otherwise a 1D array with shape ``(num_units,)`` (:math:`bz_{h}`).
    W_in_to_reset: callable, Numpy array or Theano shared variable
        If callable, initializer for the weights between the
        input and the reset gate. Otherwise a matrix with
        shape ``(num_inputs, num_units)`` (:math:`Wr_{x}`).
    b_in_to_reset: callable, Numpy array, Theano shared variable or ``None``
        If callable, initializer for the biases between the
        input and the reset gate. If ``None``, the controller
        has no bias between the input and the reset gate. Otherwise
        a 1D array with shape ``(num_units,)`` (:math:`br_{x}`).
    W_reads_to_reset: callable, Numpy array or Theano shared variable
        If callable, initializer for the weights between the
        read vector and the reset gate. Otherwise a matrix with
        shape ``(num_reads * memory_shape[1], num_units)`` (:math:`Wr_{r}`).
    b_reads_to_reset: callable, Numpy array, Theano shared variable or ``None``
        If callable, initializer for the biases between the
        read vector and the reset gate. If ``None``, the controller
        has no bias between the read vector and the reset gate.
        Otherwise a 1D array with shape ``(num_units,)`` (:math:`br_{r}`).
    W_hid_to_reset: callable, Numpy array or Theano shared variable
        If callable, initializer for the weights between the
        hidden state and the reset gate. Otherwise a matrix with
        shape ``(num_units, num_units)`` (:math:`Wr_{h}`).
    b_hid_to_reset: callable, Numpy array, Theano shared variable or ``None``
        If callable, initializer for the biases between the
        hidden state and the reset gate. If ``None``, the controller
        has no bias between the hidden state and the reset gate.
        Otherwise a 1D array with shape ``(num_units,)`` (:math:`br_{h}`).
    W_in_to_hid: callable, Numpy array or Theano shared variable
        If callable, initializer for the weights between the
        input and the hidden gate. Otherwise a matrix with
        shape ``(num_inputs, num_units)`` (:math:`Ws_{x}`).
    b_in_to_hid: callable, Numpy array, Theano shared variable or ``None``
        If callable, initializer for the biases between the
        input and the hidden gate. If ``None``, the controller
        has no bias between the input and the hidden gate. Otherwise
        a 1D array with shape ``(num_units,)`` (:math:`bs_{x}`).
    W_reads_to_hid: callable, Numpy array or Theano shared variable
        If callable, initializer for the weights between the
        read vector and the hidden gate. Otherwise a matrix with
        shape ``(num_reads * memory_shape[1], num_units)`` (:math:`Ws_{r}`).
    b_reads_to_hid: callable, Numpy array, Theano shared variable or ``None``
        If callable, initializer for the biases between the
        read vector and the hidden gate. If ``None``, the controller
        has no bias between the read vector and the hidden gate.
        Otherwise a 1D array with shape ``(num_units,)`` (:math:`bs_{r}`).
    W_hid_to_hid: callable, Numpy array or Theano shared variable
        If callable, initializer for the weights between the
        hidden state and the hidden gate. Otherwise a matrix with
        shape ``(num_units, num_units)`` (:math:`Ws_{h}`).
    b_hid_to_hid: callable, Numpy array, Theano shared variable or ``None``
        If callable, initializer for the biases between the
        hidden state and the hidden gate. If ``None``, the controller
        has no bias between the hidden state and the hidden gate.
        Otherwise a 1D array with shape ``(num_units,)`` (:math:`bs_{h}`).
    hid_init: callable, np.ndarray or theano.shared
        Initializer for the initial hidden state (:math:`h_{0}`).
    learn_init: bool
        If ``True``, initial hidden values are learned.
    """
    def __init__(self, incoming, memory_shape, num_units, num_reads,
                 W_in_to_update=lasagne.init.GlorotUniform(),
                 b_in_to_update=lasagne.init.Constant(0.),
                 W_reads_to_update=lasagne.init.GlorotUniform(),
                 b_reads_to_update=lasagne.init.Constant(0.),
                 W_hid_to_update=lasagne.init.GlorotUniform(),
                 b_hid_to_update=lasagne.init.Constant(0.),
                 W_in_to_reset=lasagne.init.GlorotUniform(),
                 b_in_to_reset=lasagne.init.Constant(0.),
                 W_reads_to_reset=lasagne.init.GlorotUniform(),
                 b_reads_to_reset=lasagne.init.Constant(0.),
                 W_hid_to_reset=lasagne.init.GlorotUniform(),
                 b_hid_to_reset=lasagne.init.Constant(0.),
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
        super(GRUController, self).__init__(incoming, memory_shape, num_units,
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
        # Inputs / Update Gate parameters
        self.W_in_to_update, self.b_in_to_update = add_weight_and_bias_params(num_inputs,
            W_in_to_update, b_in_to_update, name='in_to_update')
        # Read vectors / Update Gate parameters
        self.W_reads_to_update, self.b_reads_to_update = add_weight_and_bias_params(self.num_reads * self.memory_shape[1],
            W_reads_to_update, b_reads_to_update, name='reads_to_update')
        # Hidden / Update Gate parameters
        self.W_hid_to_update, self.b_hid_to_update = add_weight_and_bias_params(self.num_units,
            W_hid_to_update, b_hid_to_update, name='hid_to_update')
        # Inputs / Reset Gate parameters
        self.W_in_to_reset, self.b_in_to_reset = add_weight_and_bias_params(num_inputs,
            W_in_to_reset, b_in_to_reset, name='in_to_reset')
        # Read vectors / Reset Gate parameters
        self.W_reads_to_reset, self.b_reads_to_reset = add_weight_and_bias_params(self.num_reads * self.memory_shape[1],
            W_reads_to_reset, b_reads_to_reset, name='reads_to_reset')
        # Hidden / Reset Gate parameters
        self.W_hid_to_reset, self.b_hid_to_reset = add_weight_and_bias_params(self.num_units,
            W_hid_to_reset, b_hid_to_reset, name='hid_to_reset')
        # Inputs / Hidden Gate parameters
        self.W_in_to_hid, self.b_in_to_hid = add_weight_and_bias_params(num_inputs,
            W_in_to_hid, b_in_to_hid, name='in_to_hid')
        # Read vectors / Hidden Gate parameters
        self.W_reads_to_hid, self.b_reads_to_hid = add_weight_and_bias_params(self.num_reads * self.memory_shape[1],
            W_reads_to_hid, b_reads_to_hid, name='reads_to_hid')
        # Hidden / Hidden Gate parameters
        self.W_hid_to_hid, self.b_hid_to_hid = add_weight_and_bias_params(self.num_units,
            W_hid_to_hid, b_hid_to_hid, name='hid_to_hid')

    def step(self, input, reads, hidden, *args):
        if input.ndim > 2:
            input = input.flatten(2)
        if reads.ndim > 2:
            reads = reads.flatten(2)
        # Update Gate output computation
        activation = T.dot(input, self.W_in_to_update) + \
                     T.dot(reads, self.W_reads_to_update) + \
                     T.dot(hidden, self.W_hid_to_update)
        if self.b_in_to_update is not None:
            activation += self.b_in_to_update.dimshuffle('x', 0)
        if self.b_reads_to_update is not None:
            activation += self.b_reads_to_update.dimshuffle('x', 0)
        if self.b_hid_to_update is not None:
            activation += self.b_hid_to_update.dimshuffle('x', 0)
        update_gate = lasagne.nonlinearities.sigmoid(activation)
        # Reset Gate output computation
        activation = T.dot(input, self.W_in_to_reset) + \
                     T.dot(reads, self.W_reads_to_reset) + \
                     T.dot(hidden, self.W_hid_to_reset)
        if self.b_in_to_reset is not None:
            activation += self.b_in_to_reset.dimshuffle('x', 0)
        if self.b_reads_to_reset is not None:
            activation += self.b_reads_to_reset.dimshuffle('x', 0)
        if self.b_hid_to_reset is not None:
            activation += self.b_hid_to_reset.dimshuffle('x', 0)
        reset_gate = lasagne.nonlinearities.sigmoid(activation)
        # Hidden Gate output computation
        activation = T.dot(input, self.W_in_to_hid) + \
                     T.dot(reads, self.W_reads_to_hid) + \
                     T.dot((hidden * reset_gate), self.W_hid_to_hid)
        if self.b_in_to_hid is not None:
            activation += self.b_in_to_hid.dimshuffle('x', 0)
        if self.b_reads_to_hid is not None:
            activation += self.b_reads_to_hid.dimshuffle('x', 0)
        if self.b_hid_to_hid is not None:
            activation += self.b_hid_to_hid.dimshuffle('x', 0)
        hidden_gate = lasagne.nonlinearities.tanh(activation)
        # New hidden state computation
        ones = T.ones(update_gate.shape)
        state = (ones - update_gate) * hidden_gate + update_gate * hidden
        return state, state

    def outputs_info(self, batch_size):
        ones_vector = T.ones((batch_size, 1))
        hid_init = T.dot(ones_vector, self.hid_init)
        hid_init = T.unbroadcast(hid_init, 0)
        return [hid_init, hid_init]
