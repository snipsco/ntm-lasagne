import theano
import theano.tensor as T

from lasagne.layers import Layer

from heads import ReadHead, WriteHead


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
        self.write_heads = filter(lambda head: isinstance(head, WriteHead), heads)
        self.read_heads = filter(lambda head: isinstance(head, ReadHead), heads)
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
        num_write_heads = len(self.write_heads)
        num_read_heads = len(self.read_heads)

        def step(x_t, M_tm1, h_tm1, state_tm1, *params):
            # Update the memory (using w_tm1 of the writing heads & M_tm1)
            M_t = M_tm1
            # Erase
            for i in range(num_write_heads):
                erase = self.write_heads[i].erase.get_output_for(h_tm1, **kwargs)
                erasing, _ = theano.map(T.outer, sequences=[params[i], erase])
                M_t *= 1. - erasing
            # Add
            for i in range(num_write_heads):
                if self.write_heads[i].sign_add is not None:
                    sign = self.write_heads[i].sign_add.get_output_for(h_tm1, **kwargs)
                else:
                    sign = 1.
                add = self.write_heads[i].add.get_output_for(h_tm1, **kwargs)
                adding, _ = theano.map(T.outer, sequences=[params[i], sign * add])
                M_t += adding

            # Get the read vector (using w_tm1 of the reading heads & M_t)
            read_vectors = []
            for i in range(num_write_heads, num_write_heads + num_read_heads):
                reading, _ = theano.map(T.dot, sequences=[params[i], M_t])
                read_vectors.append(reading)
            r_t = T.stack(read_vectors, axis=1)

            # Apply the controller (using x_t, r_t & the requirements for the controller)
            h_t, state_t = self.controller.step(x_t, r_t, h_tm1, state_tm1)

            # Update the weights (using h_t, M_t & w_tm1)
            write_weights_t, read_weights_t = [], []
            for i in range(num_write_heads):
                weights = self.write_heads[i].get_output_for(h_t, \
                    params[i], M_t, **kwargs)
                write_weights_t.append(weights)
            for i in range(num_read_heads):
                weights = self.read_heads[i].get_output_for(h_t, \
                    params[num_write_heads + i], M_t, **kwargs)
                read_weights_t.append(weights)

            return [M_t, h_t, state_t] + write_weights_t + read_weights_t

        memory_init = T.tile(self.memory.memory_init, (self.input_shape[0], 1, 1))
        memory_init = T.unbroadcast(memory_init, 0)

        ones_vector = T.ones((self.input_shape[0], 1))
        write_weights_init = [T.unbroadcast(T.dot(ones_vector, \
            head.weights_init), 0) for head in self.write_heads]
        read_weights_init = [T.unbroadcast(T.dot(ones_vector, \
            head.weights_init), 0) for head in self.read_heads]

        non_seqs = self.controller.get_params() + self.memory.get_params()
        for head in self.heads:
            non_seqs += head.get_params()

        hids, _ = theano.scan(
            fn=step,
            sequences=input,
            outputs_info=[memory_init] + self.controller.outputs_info + \
                         write_weights_init + read_weights_init,
            non_sequences=non_seqs,
            strict=True)

        # dimshuffle back to (n_batch, n_time_steps, n_features)
        if get_details:
            hid_out = [hids[0].dimshuffle(1, 0, 2, 3)]
            hid_out += [hid.dimshuffle(1, 0, 2) for hid in hids[1:]]
        else:
            if self.only_return_final:
                hid_out = hids[1][-1]
            else:
                hid_out = hids[1].dimshuffle(1, 0, 2)

        return hid_out
