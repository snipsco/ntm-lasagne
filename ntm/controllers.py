import theano
import theano.tensor as T
import numpy as np

from lasagne.layers import Layer, MergeLayer
from lasagne.layers.recurrent import Gate, LSTMLayer
import lasagne.nonlinearities
import lasagne.init


class Controller(object):
    """
    docstring for Controller
    """
    def __init__(self, heads, **kwargs):
        self.heads = heads

    def step(self, input, reads, hids, **kwargs):
        """
        Step function for the controller
        """
        raise NotImplementedError

    def non_sequences(self, **kwargs):
        raise NotImplementedError


class LSTMController(Controller, LSTMLayer):
    """
    docstring for LSTMController
    See: https://github.com/Lasagne/Lasagne/pull/294#issuecomment-112104602
    """
    def __init__(self, incoming, heads, num_units,
                 ingate=Gate(),
                 forgetgate=Gate(),
                 cell=Gate(W_cell=None, nonlinearity=lasagne.nonlinearities.tanh),
                 outgate=Gate(),
                 nonlinearity=lasagne.nonlinearities.tanh,
                 cell_init=lasagne.init.Constant(0.),
                 hid_init=lasagne.init.Constant(0.),
                 backwards=False,
                 learn_init=False,
                 peepholes=True,
                 gradient_steps=-1,
                 grad_clipping=False,
                 unroll_scan=False,
                 precompute_input=True,
                 mask_input=None,
                 **kwargs):
        Controller.__init__(self, heads, **kwargs)
        LSTMLayer.__init__(self, incoming, num_units, ingate=ingate, forgetgate=forgetgate,
            cell=cell, outgate=outgate, nonlinearity=nonlinearity, cell_init=cell_init,
            hid_init=hid_init, backwards=backwards, learn_init=learn_init, peepholes=peepholes,
            gradient_steps=gradient_steps, grad_clipping=grad_clipping, unroll_scan=unroll_scan,
            precompute_input=precompute_input, mask_input=mask_input, **kwargs)

    def step(input, reads, cell_previous, hid_previous, W_hid_stacked,
                 W_cell_to_ingate, W_cell_to_forgetgate,
                 W_cell_to_outgate, W_in_stacked, b_stacked):

        # At each call to scan, input_n will be (n_time_steps, 4*num_units).
        # We define a slicing function that extract the input to each LSTM gate
        def slice_w(x, n):
            return x[:, n*self.num_units:(n+1)*self.num_units]

        # if not self.precompute_input:
        input = T.dot(input, W_in_stacked) + b_stacked

        # Calculate gates pre-activations and slice
        gates = input + T.dot(hid_previous, W_hid_stacked)

        # Clip gradients
        if self.grad_clipping is not False:
            gates = theano.gradient.grad_clip(
                gates, -self.grad_clipping, self.grad_clipping)

        # Extract the pre-activation gate values
        ingate = slice_w(gates, 0)
        forgetgate = slice_w(gates, 1)
        cell_input = slice_w(gates, 2)
        outgate = slice_w(gates, 3)

        if self.peepholes:
            # Compute peephole connections
            ingate += cell_previous * W_cell_to_ingate
            forgetgate += cell_previous * W_cell_to_forgetgate

        # Apply nonlinearities
        ingate = self.nonlinearity_ingate(ingate)
        forgetgate = self.nonlinearity_forgetgate(forgetgate)
        cell_input = self.nonlinearity_cell(cell_input)
        outgate = self.nonlinearity_outgate(outgate)

        # Compute new cell value
        cell = forgetgate * cell_previous + ingate * cell_input

        if self.peepholes:
            outgate += cell * W_cell_to_outgate

        # Compute new hidden unit activation
        hid = outgate * self.nonlinearity(cell)
        return [cell, hid]


# For the controller, create a step function that takes input and hidden states (stateS
# because of LSTM that outputs the hidden state and the cell state) and returns the 
# output and hidden states

if __name__ == '__main__':
    import lasagne.layers
    inp = lasagne.layers.InputLayer((None, None, 10))
    ctrl = LSTMController(inp, heads=[], num_units=100)
    print ctrl.num_units
    print ctrl.heads