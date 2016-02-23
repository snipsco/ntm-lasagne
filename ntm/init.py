import numpy as np

import lasagne.init
from lasagne.utils import floatX


class OneHot(lasagne.init.Initializer):
    """
    Initialize the weights to one-hot vectors.
    """
    def sample(self, shape):
        if len(shape) != 2:
            raise ValueError('The OneHot initializer '
                             'only works with 2D arrays.')
        M = np.min(shape)
        arr = np.zeros(shape)
        arr[:M, :M] += 1 * np.eye(M)
        return floatX(arr)
