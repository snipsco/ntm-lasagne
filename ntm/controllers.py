import theano
import theano.tensor as T
import numpy as np

from lasagne.layers import Layer, MergeLayer, DenseLayer, InputLayer
from lasagne.layers.recurrent import Gate, LSTMLayer
import lasagne.nonlinearities
import lasagne.init
import lasagne.utils

from collections import OrderedDict


class Controller(object):
    """
    docstring for Controller
    """
    def __init__(self, incoming, num_units, num_reads, name=None):
        if isinstance(incoming, tuple):
            self.input_shape = incoming
            self.input_layer = None
        else:
            self.input_shape = incoming.output_shape
            self.input_layer = incoming

        self.name = name
        self.params = OrderedDict()
        self.num_units = num_units
        self.num_reads = num_reads

    def add_param(self, spec, shape, name=None, **tags):
        # prefix the param name with the layer name if it exists
        if name is not None:
            if self.name is not None:
                name = "%s.%s" % (self.name, name)

        param = lasagne.utils.create_param(spec, shape, name)
        # parameters should be trainable and regularizable by default
        tags['trainable'] = tags.get('trainable', True)
        tags['regularizable'] = tags.get('regularizable', True)
        self.params[param] = set(tag for tag, value in tags.items() if value)

        return param

    def step(self, input, reads, *args, **kwargs):
        raise NotImplementedError

    @property
    def outputs_info(self):
        return []


class DenseController(Controller):
    """
    docstring for DenseController
    """
    def __init__(self, incoming, num_units, num_reads,
                 W_in_to_hid=lasagne.init.GlorotUniform(),
                 b_in_to_hid=lasagne.init.Constant(0.),
                 W_reads_to_hid=lasagne.init.GlorotUniform(),
                 b_reads_to_hid=lasagne.init.Constant(0.),
                 nonlinearity=lasagne.nonlinearities.rectify,
                 **kwargs):
        super(DenseController, self).__init__(incoming, num_units,
                                              num_reads, **kwargs)
        self.nonlinearity = (lasagne.nonlinearities.identity if 
                             nonlinearity is None else nonlinearity)

        def add_weight_and_bias_params(input_dim, W, b, name):
            return (self.add_param(W, (input_dim, self.num_units), 
                name='W_{}'.format(name)),
                self.add_param(b, (self.num_units,),
                name='b_{}'.format(name)) if b is not None else None)
        # Inputs / Hidden parameters
        num_inputs = int(np.prod(self.input_shape[2:]))
        self.W_in_to_hid, self.b_in_to_hid = add_weight_and_bias_params(num_inputs,
            W_in_to_hid, b_in_to_hid, name='in_to_hid')
        # Read vectors / Hidden parameters
        self.W_reads_to_hid, self.b_reads_to_hid = add_weight_and_bias_params(num_reads,
            W_reads_to_hid, b_reads_to_hid, name='reads_to_hid')

    def step(self, input, reads):
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
        return self.nonlinearity(activation)

    @property
    def outputs_info(self):
        return []


if __name__ == '__main__':
    import lasagne.layers
    inp = lasagne.layers.InputLayer((1, None, 10))
    ctrl = DenseController(inp, num_units=100, num_reads=20, name='controller')