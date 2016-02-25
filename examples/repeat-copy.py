import theano
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt

from lasagne.layers import InputLayer, DenseLayer, ReshapeLayer
import lasagne.layers
import lasagne.nonlinearities
import lasagne.updates
import lasagne.objectives
import lasagne.init

from ntm.layers import NTMLayer
from ntm.memory import Memory
from ntm.controllers import DenseController
from ntm.heads import WriteHead, ReadHead
from ntm.updates import graves_rmsprop

from utils.generators import RepeatCopyTask
from utils.visualization import Dashboard


def model(input_var, batch_size=1, size=8, num_units=100, memory_shape=(128, 20)):

    # Input Layer
    l_input = InputLayer((batch_size, None, size + 2), input_var=input_var)
    _, seqlen, _ = l_input.input_var.shape

    # Neural Turing Machine Layer
    memory = Memory(memory_shape, name='memory', memory_init=lasagne.init.Constant(1e-6), learn_init=False)
    controller = DenseController(l_input, memory_shape=memory_shape,
        num_units=num_units, num_reads=1,
        nonlinearity=lasagne.nonlinearities.rectify,
        name='controller')
    heads = [
        WriteHead(controller, num_shifts=3, memory_shape=memory_shape, name='write', learn_init=False,
            nonlinearity_key=lasagne.nonlinearities.rectify,
            nonlinearity_add=lasagne.nonlinearities.rectify),
        ReadHead(controller, num_shifts=3, memory_shape=memory_shape, name='read', learn_init=False,
            nonlinearity_key=lasagne.nonlinearities.rectify)
    ]
    l_ntm = NTMLayer(l_input, memory=memory, controller=controller, heads=heads)

    # Output Layer
    l_output_reshape = ReshapeLayer(l_ntm, (-1, num_units))
    l_output_dense = DenseLayer(l_output_reshape, num_units=size + 2, nonlinearity=lasagne.nonlinearities.sigmoid, \
        name='dense')
    l_output = ReshapeLayer(l_output_dense, (batch_size, seqlen, size + 2))

    return l_output, l_ntm


if __name__ == '__main__':
    # Define the input and expected output variable
    input_var, target_var = T.tensor3s('input', 'target')
    # The generator to sample examples from
    generator = RepeatCopyTask(batch_size=1, max_iter=1000000, size=8, min_length=3, \
        max_length=5, max_repeats=5, unary=True, end_marker=True)
    # The model (1-layer Neural Turing Machine)
    l_output, l_ntm = model(input_var, batch_size=generator.batch_size,
        size=generator.size, num_units=100, memory_shape=(128, 20))
    # The generated output variable and the loss function
    pred_var = T.clip(lasagne.layers.get_output(l_output), 1e-6, 1. - 1e-6)
    loss = T.mean(lasagne.objectives.binary_crossentropy(pred_var, target_var))
    # Create the update expressions
    params = lasagne.layers.get_all_params(l_output, trainable=True)
    updates = lasagne.updates.adam(loss, params, learning_rate=5e-4)
    # Compile the function for a training step, as well as the prediction function and
    # a utility function to get the inner details of the NTM
    train_fn = theano.function([input_var, target_var], loss, updates=updates)
    ntm_fn = theano.function([input_var], pred_var)
    ntm_layer_fn = theano.function([input_var], lasagne.layers.get_output(l_ntm, get_details=True))

    # Training
    try:
        scores, all_scores = [], []
        for i, (example_input, example_output) in generator:
            score = train_fn(example_input, example_output)
            scores.append(score)
            all_scores.append(score)
            if i % 500 == 0:
                mean_scores = np.mean(scores)
                print 'Batch #%d: %.6f' % (i, mean_scores)
                scores = []
    except KeyboardInterrupt:
        pass

    # Visualization
    def marker(generator):
        def marker_(params):
            num_repeats_length = params['repeats'] if generator.unary else 1
            return params['length'] + num_repeats_length
        return marker_
    markers = [
        {
            'location': marker(generator),
            'style': {'color': 'red', 'ls': '-'}
        }
    ]

    dashboard = Dashboard(generator=generator, ntm_fn=ntm_fn, ntm_layer_fn=ntm_layer_fn, \
        memory_shape=(128, 20), markers=markers, cmap='bone')

    # Example
    params = generator.sample_params()
    dashboard.sample(**params)
