import theano
import theano.tensor as T
import numpy as np

from lasagne.layers import Layer, InputLayer
import lasagne.init
import lasagne.layer.helper as helper
from heads import ReadHead, WriteHead, HeadLayer


class NTM(Layer):
    """
    docstring for NTM
    """
    def __init__(self, incoming,
                 memory,
                 controller,
                 heads,
                 **kwargs):
        super(NTM, self).__init__(incoming, **kwargs)

        # Populate the HeadLayers with memory & previous layers
        self.memory = memory
        self.controller = controller
        self.heads = heads

    def get_output_shape_for(self, input_shape):
        return self.controller.get_output_shape_for(input_shape)

    def get_output_for(self, input, **kwargs):

        def step(x_t, M_tm1, h_tm1, *params):
            # In the list params there are, in that order
            #   - w_tm1 for all the writing heads
            #   - w_tm1 for all the reading heads
            #   - Additional requirements for the controller (e.g. c_tm1 for LSTM)
            #   - M_0
            #   - h_0
            #   - w_0, W_hid_to_key, b_hid_to_key, ... for all the writing heads (15)
            #   - w_0, W_hid_to_key, b_hid_to_key, ... for all the reading heads (11)
            #   - Controller parameters (e.g. W & b for Dense)
            #   - Additional initial req. for the controller (e.g. c_0 for LSTM)
            num_write_heads = len(filter(lambda head: isinstance(head, WriteHead), self.heads))
            num_read_heads = len(filter(lambda head: isinstance(head, ReadHead), self.heads))
            num_heads = num_write_heads + num_read_heads
            outputs_t = []

            # Update the memory (using w_tm1 of the writing heads & M_tm1)
            M_t = M_tm1
            # Erase
            for i in range(num_write_heads):
                M_t *= 1. - T.outer(params[i], helper.get_output(self.heads[i].erase, h_tm1))
            # Add
            for i in range(num_write_heads):
                M_t += T.outer(params[i], helper.get_output(self.heads[i].add, h_tm1))
            outputs_t.append(M_t)

            # Get the read vector (using w_tm1 of the reading heads & M_t)
            read_vectors = []
            for i in range(num_write_heads, num_heads):
                read_vectors.append(T.dot(params[i], M_t))
            r_t = T.concatenate(read_vectors)

            # Apply the controller (using x_t, r_t & requirements for the controller)
            if self.controller.outputs_info is None or not self.controller.outputs_info:
                ctrl_tm1 = []
            else:
                num_ctrl_req = len(self.controller.outputs_info) - 1
                ctrl_tm1 = [h_tm1] + params[num_heads:num_heads + num_ctrl_req]
            h_t, ctrl_t = self.controller.step(x_t, r_t, *ctrl_tm1, *self.controller.non_sequences)
            outputs_t.append(h_t)

            # Update the weights (using h_t, M_t & w_tm1)
            for i in range(num_heads):
                outputs_t.append(helper.get_output(self.heads[i], [h_t, M_t, params[i]]))

            outputs_t += ctrl_t

            return outputs_t

        hids, _ = theano.scan(
            fn=step,
            sequences=input,
            outputs_info=[self.memory.memory_init, ],
            non_sequences=self.controller.non_sequences,
            strict=True)

        return hids[1]


if __name__ == '__main__':
    import lasagne.layers
    inp = lasagne.layers.InputLayer((None, None, 10))
    ntm = NTM(inp, memory_shape=(128, 20), heads=[])