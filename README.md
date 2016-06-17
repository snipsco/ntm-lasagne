# NTM-Lasagne

NTM-Lasagne is a library to create Neural Turing Machines (NTMs) in [Theano](http://deeplearning.net/software/theano/) using the [Lasagne](http://lasagne.readthedocs.org/) library. If you want to learn more about NTMs, check out our [blog post](https://medium.com/snips-ai/ntm-lasagne-a-library-for-neural-turing-machines-in-lasagne-2cdce6837315#.63t84s5r5).

This library features:
 - A Neural Turing Machine layer `NTMLayer`, where all its components (controller, heads, memory) are fully customizable.
 - Two types of controllers: a feed-forward `DenseController` and a "vanilla" recurrent `RecurrentController`.
 - A dashboard to visualize the inner mechanism of the NTM.
 - Generators to sample examples from algorithmic tasks.

### Installation

This library is compatible with Python 2.7.8, and may be partly compatible with Python 3.x. NTM-Lasagne requires the bleeding-edge versions of Lasagne and Theano. Check the [Lasagne installation instructions](http://lasagne.readthedocs.org/en/latest/user/installation.html#bleeding-edge-version) for details, or install them with `pip install -r requirements.txt`.
To install this library, clone this repository and then run the `setup.py` script.

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
PYTHONPATH=. python examples/copy-task.py
```

and be patient while Theano compiles the model ;-). Graph optimization is computationally intensive. If you are encountering suspiciously long compilation times (more than a few minutes), you may need to increase the amount of memory allocated (if you run it on a Virtual Machine). Alternatively, turning off the swap may help for debugging (with `swapoff`/`swapon`).
Note: unlucky initialisation of the parameters might lead to a diverging solution (NaN scores).

Alex Graves, Greg Wayne, Ivo Danihelka, *Neural Turing Machines*, [[arXiv](https://arxiv.org/abs/1410.5401)]