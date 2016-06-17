import theano
import theano.tensor as T
import numpy as np


def cosine_similarity(x, y, eps=1e-6):
    r"""
    Cosine similarity between a vector and each row of a base matrix.

    Parameters
    ----------
    x: a 3D Theano variable
        Vector to compare to each row of the matrix y.
    y: a 3D Theano variable
        Matrix to be compared to
    eps: float
        Precision of the operation (necessary for differentiability).

    Return
    ------
    z: a 3D Theano variable
        A vector whose components are the cosine similarities
        between x and each row of y.
    """
    z = T.batched_dot(x, y.dimshuffle(0, 2, 1))
    z /= T.sqrt(T.sum(x * x, axis=2).dimshuffle(0, 1, 'x') * T.sum(y * y, axis=2).dimshuffle(0, 'x', 1) + eps)

    return z
