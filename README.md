# NTM-Lasagne

[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](https://raw.githubusercontent.com/snipsco/ntm-lasagne/master/LICENSE)

NTM-Lasagne is a library to create Neural Turing Machines (NTMs) in [Theano](http://deeplearning.net/software/theano/) using the [Lasagne](http://lasagne.readthedocs.org/) library. If you want to learn more about NTMs, check out our [blog post](https://medium.com/snips-ai/ntm-lasagne-a-library-for-neural-turing-machines-in-lasagne-2cdce6837315#.63t84s5r5).

This library features:
 - A Neural Turing Machine layer `NTMLayer`, where all its components (controller, heads, memory) are fully customizable.
 - Two types of controllers: a feed-forward `DenseController` and a "vanilla" recurrent `RecurrentController`.
 - A dashboard to visualize the inner mechanism of the NTM.
 - Generators to sample examples from algorithmic tasks.

## Getting started
To avoid any conflict with your existing Python setup, and to keep this project self-contained, it is suggested to work in a virtual environment with [`virtualenv`](http://docs.python-guide.org/en/latest/dev/virtualenvs/). To install `virtualenv`:
```bash
sudo pip install --upgrade virtualenv
```

Create a virtual environment called `venv`, activate it and install the requirements given by `requirements.txt`. NTM-Lasagne requires the bleeding-edge version, check the [Lasagne installation instructions](http://lasagne.readthedocs.org/en/latest/user/installation.html#bleeding-edge-version) for details. The latest version of [Lasagne](https://github.com/Lasagne/Lasagne/) is included in the `requirements.txt`.
```bash
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
pip install .
```

## Example
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

## Tests
This projects has a few basic tests. To run these tests, you can run the `py.test` on the project folder
```bash
venv/bin/py.test ntm -vv
```

## Known issues
Graph optimization is computationally intensive. If you are encountering suspiciously long compilation times (more than a few minutes), you may need to increase the amount of memory allocated (if you run it on a Virtual Machine). Alternatively, turning off the swap may help for debugging (with `swapoff`/`swapon`).

Note: Unlucky initialisation of the parameters might lead to a diverging solution (`NaN` scores).

## Paper
Alex Graves, Greg Wayne, Ivo Danihelka, *Neural Turing Machines*, [[arXiv](https://arxiv.org/abs/1410.5401)]

## Contributing

Please see the [Contribution Guidelines](https://github.com/snipsco/ntm-lasagne/blob/master/CONTRIBUTING.md).

## Copyright

This library is provided by [Snips](https://www.snips.ai) as Open Source software. See [LICENSE](https://github.com/snipsco/ntm-lasagne/blob/master/LICENSE) for more information.
