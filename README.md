# NTM-Lasagne

NTM-Lasagne is a library to create Neural Turing Machines (NTMs) in [Theano](http://deeplearning.net/software/theano/) using the [Lasagne](http://lasagne.readthedocs.org/) library.

This library features:
 - A Neural Turing Machine layer `NTMLayer`, where all its components (controller, heads, memory) are fully customizable.
 - Two types of controllers: a feed-forward `DenseController` and a "vanilla" recurrent `RecurrentController`.
 - A dashboard to visualize the inner mechanism of the NTM.
 - Generators to sample examples from algorithmic tasks.

### Installation

This library is compatible with Python 2.7.8, and may be partly compatible with Python 3.x. NTM-Lasagne requires the bleeding-edge versions of Lasagne and Theano. To install this library, clone this repository and then run the `setup.py` script.

```
git clone https://github.com/snipsco/ntm-lasagne.git
cd ntm-lasagne/
pip install -r requirements.txt
python setup.py install
```

### Example

Here is minimal example to define a `NTMLayer`

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

For more detailed examples, check the [`examples` folder](examples/). If you would like to train a Neural Turing Machine on one of these examples, simply run the corresponding script, like

```
python examples/copy.py
```
