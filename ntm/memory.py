import theano
import theano.tensor as T
import numpy as np

from lasagne.layers import InputLayer
import lasagne.init


class Memory(InputLayer):
    r"""
    Memory of the Neural Turing Machine.

    Parameters
    ----------
    memory_shape: tuple
        Shape of the NTM's memory.
    memory_init: callable, Numpy array or Theano shared variable
        Initializer for the initial state of the memory (:math:`M_{0}`).
        The initial state of the memory must be non-zero.
    learn_init: bool
        If ``True``, initial state of the memory is learned.
    """
    def __init__(self, memory_shape,
        memory_init=lasagne.init.Constant(1e-6),
        learn_init=True,
        **kwargs):
        super(Memory, self).__init__(memory_shape, **kwargs)
        self.memory_init = self.add_param(
            memory_init, memory_shape,
            name='memory_init', trainable=learn_init, regularizable=False)
