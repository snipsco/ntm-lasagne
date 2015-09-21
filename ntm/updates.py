import theano
import theano.tensor as T
import numpy as np

from lasagne.updates import get_or_compute_grads
from collections import OrderedDict

def graves_rmsprop(loss_or_grads, params, chi=0.95, alpha=0.9, beta=1e-4, epsilon=1e-4):
    """
    Graves' rmsprop
    http://arxiv.org/pdf/1308.0850v5.pdf, p.23

    n_i = chi * n_i-1 + (1 - chi) * grad^2
    g_i = chi * g_i-1 + (1 - chi) * grad
    Delta_i = alpha * Delta_i-1 - beta * grad / 
              sqrt(n_i - g_i^2 + epsilon)
    w_i = w_i-1 + Delta_i

    """
    grads = get_or_compute_grads(loss_or_grads, params)
    updates = OrderedDict()

    for param, grad in zip(params, grads):
        value = param.get_value(borrow=True)
        n = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                          broadcastable=param.broadcastable)
        g = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                          broadcastable=param.broadcastable)
        delta = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                              broadcastable=param.broadcastable)
        n_ip1 = chi * n + (1. - chi) * grad ** 2
        g_ip1 = chi * g + (1. - chi) * grad
        delta_ip1 = alpha * delta - beta * grad / T.sqrt(n_ip1 + \
                    g_ip1 ** 2 + epsilon)
        updates[n] = n_ip1
        updates[g] = g_ip1
        updates[delta] = delta_ip1
        updates[param] = param + delta_ip1

    return updates
