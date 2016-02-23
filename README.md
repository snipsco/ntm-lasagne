# NTM-Lasagne

NTM-Lasagne is a library to create Neural Turing Machines (NTMs) in [Theano](http://deeplearning.net/software/theano/) using the [Lasagne](http://lasagne.readthedocs.org/) library.

### Installation
`TODO`

### Example
```python
# Neural Turing Machine Layer
memory = Memory((128, 20), memory_init=lasagne.init.Constant(1e-6),
    learn_init=False, name='memory')
controller = DenseController(l_input, memory_shape=(128, 20),
    num_units=100, num_reads=1,
    nonlinearity=lasagne.nonlinearities.rectify,
    name='controller')
heads = [
    WriteHead(controller, num_shifts=3, memory_shape=(128, 20),
        nonlinearity_key=lasagne.nonlinearities.rectify,
        nonlinearity_add=lasagne.nonlinearities.rectify,
        learn_init=False, name='write'),
    ReadHead(controller, num_shifts=3, memory_shape=(128, 20),
        nonlinearity_key=lasagne.nonlinearities.rectify,
        learn_init=False, name='read')
]
l_ntm = NTMLayer(l_input, memory=memory, controller=controller, heads=heads)
```