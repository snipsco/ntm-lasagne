import theano
import theano.tensor as T

from lasagne.layers import Layer

from heads import ReadHead, WriteHead, ReadHeadCollection, WriteHeadCollection


class NTMLayer(Layer):
    r"""
    A Neural Turing Machine layer.

    Parameters
    ----------
    incoming: a :class:`lasagne.layers.Layer` instance
        The layer feeding into the Neural Turing Machine. This
        layer must match the incoming layer in the controller.
    memory: a :class:`Memory` instance
        The memory of the NTM.
    controller: a :class:`Controller` instance
        The controller of the NTM.
    heads: a list of :class:`Head` instances
        The read and write heads of the NTM.
    only_return_final: bool
        If ``True``, only return the final sequential output (e.g.
        for tasks where a single target value for the entire
        sequence is desired).  In this case, Theano makes an
        optimization which saves memory.
    """
    def __init__(self, incoming,
                 memory,
                 controller,
                 heads,
                 only_return_final=False,
                 **kwargs):
        super(NTMLayer, self).__init__(incoming, **kwargs)

        self.memory = memory
        self.controller = controller
        self.heads = heads
        self.write_heads = WriteHeadCollection(heads=\
            filter(lambda head: isinstance(head, WriteHead), heads))
        self.read_heads = ReadHeadCollection(heads=\
            filter(lambda head: isinstance(head, ReadHead), heads))
        self.only_return_final = only_return_final

    def get_output_shape_for(self, input_shapes):
        if self.only_return_final:
            return (input_shapes[0], self.controller.num_units)
        else:
            return (input_shapes[0], input_shapes[1], self.controller.num_units)

    def get_params(self, **tags):
        params = super(NTMLayer, self).get_params(**tags)
        params += self.controller.get_params(**tags)
        params += self.memory.get_params(**tags)
        for head in self.heads:
            params += head.get_params(**tags)

        return params

    def get_output_for(self, input, get_details=False, **kwargs):

        input = input.dimshuffle(1, 0, 2)

        def step(x_t, M_tm1, h_tm1, state_tm1, ww_tm1, wr_tm1, *params):
            # Update the memory (using w_tm1 of the writing heads & M_tm1)
            M_t = self.write_heads.write(h_tm1, ww_tm1, M_tm1)

            # Get the read vector (using w_tm1 of the reading heads & M_t)
            r_t = self.read_heads.read(wr_tm1, M_t)

            # Apply the controller (using x_t, r_t & the requirements for the controller)
            h_t, state_t = self.controller.step(x_t, r_t, h_tm1, state_tm1)

            # Update the weights (using h_t, M_t & w_tm1)
            ww_t = self.write_heads.get_weights(h_t, ww_tm1, M_t)
            wr_t = self.read_heads.get_weights(h_t, wr_tm1, M_t)

            return [M_t, h_t, state_t, ww_t, wr_t]

        memory_init = T.tile(self.memory.memory_init, (input.shape[1], 1, 1))
        memory_init = T.unbroadcast(memory_init, 0)

        write_weights_init = T.tile(self.write_heads.weights_init, (input.shape[1], 1, 1))
        write_weights_init = T.unbroadcast(write_weights_init, 0)
        read_weights_init = T.tile(self.read_heads.weights_init, (input.shape[1], 1, 1))
        read_weights_init = T.unbroadcast(read_weights_init, 0)

        non_seqs = self.controller.get_params() + self.memory.get_params() + \
            self.write_heads.get_params() + self.read_heads.get_params()

        hids, _ = theano.scan(
            fn=step,
            sequences=input,
            outputs_info=[memory_init] + self.controller.outputs_info(input.shape[1]) + \
                         [write_weights_init, read_weights_init],
            non_sequences=non_seqs,
            strict=True)

        # dimshuffle back to (n_batch, n_time_steps, n_features)
        if get_details:
            hid_out = [
                hids[0].dimshuffle(1, 0, 2, 3),
                hids[1].dimshuffle(1, 0, 2),
                hids[2].dimshuffle(1, 0, 2),
                hids[3].dimshuffle(1, 0, 2, 3),
                hids[4].dimshuffle(1, 0, 2, 3)]
        else:
            if self.only_return_final:
                hid_out = hids[1][-1]
            else:
                hid_out = hids[1].dimshuffle(1, 0, 2)

        return hid_out
