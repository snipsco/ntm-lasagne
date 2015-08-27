import theano
import theano.tensor as T


class Head(object):
    """
    Base class for the head
    """
    def __init__(self, key=None, beta=None, gate=None,
                 shift=None, gamma=None, name=None):
        self.key = key
        self.beta = beta
        self.gate = gate
        self.shift = shift
        self.gamma = gamma
        self.name = name