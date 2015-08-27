import theano
import theano.tensor as T
import numpy as np

from lasagne.layers import Layer
from heads import ReadHead, WriteHead


class NTM(Layer):
    """
    docstring for NTM
    """
    def __init__(self, incoming,
                 memory_shape,
                 heads,
                 **kwargs):
        super(NTM, self).__init__(incoming, **kwargs)
        self.memory_shape = memory_shape
        self.heads = heads
        self.read_heads = [head for head in heads if isinstance(head, ReadHead)]
        self.write_heads = [head for head in heads if isinstance(head, WriteHead)]


if __name__ == '__main__':
    import lasagne.layers
    inp = lasagne.layers.InputLayer((None, None, 10))
    ntm = NTM(inp, memory_shape=(128, 20), heads=[])