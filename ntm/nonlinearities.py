import theano.tensor as T


class ClippedLinear(object):
    """
    Clipped linear activation.
    """
    def __init__(self, low=0., high=1.):
        super(ClippedLinear, self).__init__()
        self.low = low
        self.high = high
    
    def __call__(self, x):
        return T.clip(x, self.low, self.high)

def hard_sigmoid(x):
    return T.nnet.hard_sigmoid(x)