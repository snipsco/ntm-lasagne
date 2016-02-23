import theano
import theano.tensor as T
import numpy as np

from lasagne.updates import get_or_compute_grads
from collections import OrderedDict

def graves_rmsprop(loss_or_grads, params, learning_rate=1e-4, chi=0.95, alpha=0.9, epsilon=1e-4):
    r"""
    Alex Graves' RMSProp [1]_.

    .. math ::
        n_{i} &= \chi * n_i-1 + (1 - \chi) * grad^{2}\\
        g_{i} &= \chi * g_i-1 + (1 - \chi) * grad\\
        \Delta_{i} &= \alpha * Delta_{i-1} - learning_rate * grad /
                  sqrt(n_{i} - g_{i}^{2} + \epsilon)\\
        w_{i} &= w_{i-1} + \Delta_{i}

    References
    ----------
    .. [1] Graves, Alex.
           "Generating Sequences With Recurrent Neural Networks", p.23
           arXiv:1308.0850

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
        delta_ip1 = alpha * delta - learning_rate * grad / T.sqrt(n_ip1 + \
                    g_ip1 ** 2 + epsilon)
        updates[n] = n_ip1
        updates[g] = g_ip1
        updates[delta] = delta_ip1
        updates[param] = param + delta_ip1

    return updates
