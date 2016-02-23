# NTM-Lasagne

NTM-Lasagne is a library to create Neural Turing Machines (NTMs) in [Theano](http://deeplearning.net/software/theano/) using the [Lasagne](http://lasagne.readthedocs.org/) library.

### Installation
`TODO`

### Example
```python
input_size = 8

# Input Layer
l_input = InputLayer((batch_size, None, input_size), input_var=input_var)
_, seqlen, _ = l_input.input_var.shape

# Neural Turing Machine Layer
memory = Memory((128, 20), name='memory', memory_init=lasagne.init.Constant(1e-6), learn_init=False)
controller = DenseController(l_input, memory_shape=(128, 20),
    num_units=100, num_reads=1,
    nonlinearity=lasagne.nonlinearities.rectify,
    name='controller')
heads = [
    WriteHead(controller, num_shifts=3, memory_shape=(128, 20), name='write', learn_init=False,
        nonlinearity_key=lasagne.nonlinearities.rectify,
        nonlinearity_add=lasagne.nonlinearities.rectify),
    ReadHead(controller, num_shifts=3, memory_shape=(128, 20), name='read', learn_init=False,
        nonlinearity_key=lasagne.nonlinearities.rectify)
]
l_ntm = NTMLayer(l_input, memory=memory, controller=controller, heads=heads)

# Output Layer
l_output_reshape = ReshapeLayer(l_ntm, (-1, 100))
l_output_dense = DenseLayer(l_output_reshape, num_units=input_size, nonlinearity=lasagne.nonlinearities.sigmoid, \
    name='dense')
l_output = ReshapeLayer(l_output_dense, (batch_size, seqlen, input_size))

return l_output, l_ntm
```