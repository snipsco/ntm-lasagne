import theano
import theano.tensor as T
import numpy as np

from lasagne.layers import Layer
import lasagne.init
import lasagne.layer.helper as helper
from heads import ReadHead, WriteHead


class NTM(Layer):
    """
    docstring for NTM
    """
    def __init__(self, incoming,
                 heads,
                 memory_shape
                 memory=lasagne.init.Constant(0.),
                 controller=Controller(),
                 backwards=False,
                 gradient_steps=-1,
                 **kwargs):
        super(NTM, self).__init__(incoming, **kwargs)

        self.memory_shape = memory_shape
        self.memory = memory
        self.heads = heads
        self.controller = controller
        self.backwards = backwards
        self.gradient_steps = gradient_steps
        self.read_heads = [head for head in heads if isinstance(head, ReadHead)]
        self.write_heads = [head for head in heads if isinstance(head, WriteHead)]

        # self.erase = theano.function([w, e], [],
        #     updates=[self.memory, self.memory * (1 - T.outer(w, e))])
        # self.add = theano.function([w, a], [],
        #     updates=[self.memory, self.memory + T.outer(w, a)])

    def get_output_shape_for(self, input_shape):
        return self.controller.get_output_shape_for(input_shape)

    def get_output_for(self, input, **kwargs):

        def step(x_t, M_t, h_tm1, c_tm1, *w_tm1):
            # Get the read weights (using w_tm1 or read, M_t)
            # reads = ...
            # Apply the controller (using input, reads, h_tm1, c_tm1)
            # h_t, c_t = ...
            # Apply the heads / Update the weights (using h_t, M_t, w_tm1)
            # w_t = ...
            # Update the memory (using M_t, w_t of write)
            # M_tp1 = ...
            # Get the output (using h_t)
            # y_t = ...
            # Return [y_t, M_tp1, h_t, c_t, w_t]
            pass

        hids, _ = theano.scan(
            fn=step,
            sequences=input,
            outputs_info=self.controller.outputs_info
            go_backwards=self.backwards,
            truncate_gradients=self.gradient_steps,
            non_sequences=self.controller.non_sequences,
            strict=True)


if __name__ == '__main__':
    import lasagne.layers
    inp = lasagne.layers.InputLayer((None, None, 10))
    ntm = NTM(inp, memory_shape=(128, 20), heads=[])