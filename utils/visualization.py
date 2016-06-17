import theano
import theano.tensor as T
import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt

class Dashboard(object):

    def __init__(self, ntm_fn, generator, memory_shape, ntm_layer_fn=None, \
        cmap='bone', markers=[]):
        super(Dashboard, self).__init__()
        self.ntm_fn = ntm_fn
        self.ntm_layer_fn = ntm_layer_fn
        self.memory_shape = memory_shape
        self.generator = generator
        self.markers = markers
        self.cmap = cmap

    def sample(self, **params):
        params = self.generator.sample_params(**params)
        example_input, example_output = self.generator.sample(**params)
        self.show(example_input, example_output, params)

    def show(self, example_input, example_output, params):
        example_prediction = self.ntm_fn(example_input)
        num_columns = 1
        if self.ntm_layer_fn is not None:
            num_columns = 3
            example_ntm = self.ntm_layer_fn(example_input)
        subplot_shape = (3, num_columns)
        title_props = matplotlib.font_manager.FontProperties(weight='bold', \
            size=9)

        ax1 = plt.subplot2grid(subplot_shape, (0, 2))
        ax1.imshow(example_input[0].T, interpolation='nearest', cmap=self.cmap)
        ax1.set_title('Input')
        ax1.title.set_font_properties(title_props)
        ax1.get_xaxis().set_visible(False)
        ax1.get_yaxis().set_visible(False)

        ax2 = plt.subplot2grid(subplot_shape, (1, 2))
        ax2.imshow(example_output[0].T, interpolation='nearest', cmap=self.cmap)
        ax2.set_title('Output')
        ax2.title.set_font_properties(title_props)
        ax2.get_xaxis().set_visible(False)
        ax2.get_yaxis().set_visible(False)

        ax3 = plt.subplot2grid(subplot_shape, (2, 2))
        ax3.imshow(example_prediction[0].T, interpolation='nearest', \
            cmap=self.cmap)
        ax3.set_title('Prediction')
        ax3.title.set_font_properties(title_props)
        ax3.get_xaxis().set_visible(False)
        ax3.get_yaxis().set_visible(False)

        if self.ntm_layer_fn is not None:
            ax4 = plt.subplot2grid(subplot_shape, (0, 0), rowspan=3)
            ax4.imshow(example_ntm[3][0,:,0].T, interpolation='nearest', \
                cmap=self.cmap)
            ax4.set_title('Write Weights')
            ax4.title.set_font_properties(title_props)
            ax4.get_xaxis().set_visible(False)
            for marker in self.markers:
                marker_style = marker.get('style', {})
                ax4.plot([marker['location'](params), \
                    marker['location'](params)], [0, \
                    self.memory_shape[0] - 1], **marker_style)
            ax4.set_xlim([-0.5, example_input.shape[1] - 0.5])
            ax4.set_ylim([-0.5, self.memory_shape[0] - 0.5])
            ax4.tick_params(axis='y', labelsize=9)

            ax5 = plt.subplot2grid(subplot_shape, (0, 1), rowspan=3)
            ax5.imshow(example_ntm[4][0,:,0].T, interpolation='nearest', \
                cmap=self.cmap)
            ax5.set_title('Read Weights')
            ax5.title.set_font_properties(title_props)
            ax5.get_xaxis().set_visible(False)
            for marker in self.markers:
                marker_style = marker.get('style', {})
                ax5.plot([marker['location'](params), \
                    marker['location'](params)], [0, \
                    self.memory_shape[0] - 1], **marker_style)
            ax5.set_xlim([-0.5, example_input.shape[1] - 0.5])
            ax5.set_ylim([-0.5, self.memory_shape[0] - 0.5])
            ax5.tick_params(axis='y', labelsize=9)

        plt.show()


def learning_curve(scores):
    sc = pd.Series(scores)
    ma = pd.rolling_mean(sc, window=500)

    ax = plt.subplot(1, 1, 1)
    ax.plot(sc.index, sc, color='lightgray')
    ax.plot(ma.index, ma, color='red')
    ax.set_yscale('log')
    ax.set_xlim(sc.index.min(), sc.index.max())
    plt.show()
