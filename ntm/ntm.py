import theano
import theano.tensor as T
import numpy as np

from lasagne.layers import Layer, InputLayer
import lasagne.init
import lasagne.layers.helper as helper
from heads import ReadHead, WriteHead


class NTMLayer(Layer):
    """
    docstring for NTMLayer
    """
    def __init__(self, incoming,
                 memory,
                 controller,
                 heads,
                 **kwargs):
        super(NTMLayer, self).__init__(incoming, **kwargs)

        # Populate the HeadLayers with memory & previous layers
        self.memory = memory
        self.controller = controller
        # TODO: Sort the heads to have WriteHeads > ReadHeads
        self.heads = heads

    def get_output_shape_for(self, input_shapes):
        return (input_shapes[0], input_shapes[1], self.controller.num_units)

    def get_params(self, **tags):
        params = super(NTMLayer, self).get_params(**tags)
        params += self.controller.get_params(**tags)
        params += self.memory.get_params(**tags)
        for head in self.heads:
            params += head.get_params(**tags)

        return params

    def get_output_for(self, input, **kwargs):

        input = input.dimshuffle(1, 0, 2)

        def step(x_t, M_tm1, h_tm1, *params):
            # In the list params there are, in that order
            #   - w_tm1 for all the writing heads
            #   - w_tm1 for all the reading heads
            #   - Additional requirements for the controller (e.g. c_tm1 for LSTM)
            #   - W_hid_to_key, b_hid_to_key, ... for all the writing heads (14)
            #   - W_hid_to_key, b_hid_to_key, ... for all the reading heads (10)
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
                M_t *= 1. - T.outer(params[i], self.heads[i].erase.get_output_for(h_tm1))
            # Add
            for i in range(num_write_heads):
                M_t += T.outer(params[i], self.heads[i].add.get_output_for(h_tm1))
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
                ctrl_tm1 = [h_tm1] + list(params[num_heads:num_heads + num_ctrl_req])
            ctrl_t = self.controller.step(x_t, r_t, *(ctrl_tm1 + self.controller.non_sequences))
            outputs_t.append(ctrl_t[0])
            # outputs_t.append(h_tm1)

            # Update the weights (using h_t, M_t & w_tm1)
            for i in range(num_heads):
                outputs_t.append(self.heads[i].get_output_for([ctrl_t[0], M_t, params[i]]))

            outputs_t += ctrl_t[1:]

            return outputs_t

        non_seqs = self.controller.non_sequences
        for head in self.heads:
            non_seqs += [head.W_hid_to_key, head.b_hid_to_key,
                head.W_hid_to_beta, head.b_hid_to_beta,
                head.W_hid_to_gate, head.b_hid_to_gate,
                head.W_hid_to_shift, head.b_hid_to_shift,
                head.W_hid_to_gamma, head.b_hid_to_gamma]
            if isinstance(head, WriteHead):
                non_seqs += [head.W_hid_to_erase, head.b_hid_to_erase,
                    head.W_hid_to_add, head.b_hid_to_add]
            non_seqs += self.controller.get_params()

        outs_info = [self.memory.memory_init, self.controller.hid_init]
        outs_info += [head.weights_init for head in self.heads]
        if self.controller.outputs_info is not None:
            outs_info += self.controller.outputs_info[1:]

        hids, _ = theano.scan(
            fn=step,
            sequences=input,
            outputs_info=outs_info,
            non_sequences=non_seqs,
            strict=True)

        theano.printing.Print("\n" * 20 + "WEIGHTS")(hids[2])

        # dimshuffle back to (n_batch, n_time_steps, n_features))
        hid_out = hids[1].dimshuffle(1, 0, 2)

        # return hid_out
        return hids


if __name__ == '__main__':
    import lasagne.layers
    inp = lasagne.layers.InputLayer((None, None, 10))
    ntm = NTM(inp, memory_shape=(128, 20), heads=[])