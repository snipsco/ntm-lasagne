import theano
import theano.tensor as T
import numpy as np


def cosine_similarity(x, y, eps=1e-9):
    xe, ye = x + eps, y + eps
    z = T.dot(x, y)
    z /= xe.norm(2)
    z /= ye.norm(2, axis=0)

    return z