import theano
import theano.tensor as T
import numpy as np


def cosine_similarity(x, y, eps=1e-6):
    y = y.dimshuffle(1, 0)
    z = T.dot(x, y)
    z /= T.sqrt(T.sum(x * x) * T.sum(y * y, axis=0) + eps)

    return z

def cosine_similarity_batched(x, y, eps=1e-6):
    def step(x_b, y_b):
        return cosine_similarity(x_b, y_b, eps)
    z, _ = theano.map(step, sequences=[x, y])

    return z
