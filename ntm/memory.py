import theano
import theano.tensor as T
import numpy as np

from lasagne.layers import InputLayer
import lasagne.init


class Memory(InputLayer):
    """
    docstring for Memory
    """
    def __init__(self, shape, 
        memory_init=lasagne.init.Constant(0.),
        learn_init=True,
        **kwargs):
        super(Memory, self).__init__(shape, **kwargs)
        self.memory_init = self.add_param(
            memory_init, shape,
            name=self.name, trainable=learn_init, regularizable=False)
