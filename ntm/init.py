import numpy as np
import lasagne.init
from lasagne.utils import floatX


class EquiProba(lasagne.init.Initializer):

    def sample(self, shape):
        # TODO: General case, here it only works for 2D
        M = float(shape[1])
        if M == 0:
            raise ValueError('The second dimension '
                'must be non zero')
        return floatX(np.ones(shape) / M)

class OneHot(lasagne.init.Initializer):

    def sample(self, shape):
        # TODO: General case, here it only works for 2D
        M = np.min(shape)
        arr = np.zeros(shape)
        arr[:M, :M] += 1 * np.eye(M)
        return floatX(arr)