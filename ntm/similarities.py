import theano
import theano.tensor as T
import numpy as np


def cosine_similarity(x, y, eps=1e-9):
    y = y.dimshuffle(1, 0)
    z = T.dot(x, y)
    z /= x.norm(2) * y.norm(2, axis=0).dimshuffle('x', 0) + eps

    return z