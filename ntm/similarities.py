import theano
import theano.tensor as T
import numpy as np


def cosine_similarity(x, y, eps=1e-6):
    r"""
    Cosine similarity between a vector and each row of a base matrix.

    Parameters
    ----------
    x: a 1D Theano variable
        Vector to compare to each row of the matrix y.
    y: a 2D Theano variable
        Matrix to be compared to
    eps: float
        Precision of the operation (necessary for differentiability).

    Return
    ------
    z: a 1D Theano variable
        A vector whose components are the cosine similarities
        between x and each row of y.
    """
    def _cosine_similarity(x, y, eps=1e-6):
        y = y.dimshuffle(1, 0)
        z = T.dot(x, y)
        z /= T.sqrt(T.sum(x * x) * T.sum(y * y, axis=0) + eps)

        return z

    def step(x_b, y_b):
        return _cosine_similarity(x_b, y_b, eps)
    z, _ = theano.map(step, sequences=[x, y])

    return z
