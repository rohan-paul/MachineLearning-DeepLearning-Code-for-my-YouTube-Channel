"""
A high-level strategy of coding feedforward propagation is as follows:

1. Perform a sum product at each neuron.
2. Compute activation.
3. Repeat the first two steps at each neuron until the output layer.
4. Compute the loss by comparing the prediction with the actual output.

"""
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt


def forward_propagation(inputs, outputs, weights):
    pre_activation_hidden_layer_value = np.dot(input, weights[0]) + weights[1]

    # invoke sigmoid
    hidden_value_post_activation = 1 / (1 + np.exp(-pre_activation_hidden_layer_value))

    # output value
    output_value = np.dot(hidden_value_post_activation, weights[2]) + weights[3]

    # mse
    mean_squared_error = np.mean(np.square(output_value - outputs))

    return mean_squared_error
