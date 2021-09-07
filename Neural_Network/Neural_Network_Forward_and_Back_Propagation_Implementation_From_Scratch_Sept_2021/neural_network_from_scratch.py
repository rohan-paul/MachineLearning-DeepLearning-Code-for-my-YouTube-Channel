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
    pre_activation_hidden_layer_value = np.dot(inputs, weights[0]) + weights[1]

    # invoke sigmoid
    hidden_value_post_activation = 1 / (1 + np.exp(-pre_activation_hidden_layer_value))

    # output value
    output_value = np.dot(hidden_value_post_activation, weights[2]) + weights[3]

    # mse
    mean_squared_error = np.mean(np.square(output_value - outputs))

    return mean_squared_error


def train_weights_gradient_descent(inputs, outputs, weights, lr):
    weights_original = deepcopy(weights)
    weights_temp = deepcopy(weights)
    weights_updated = deepcopy(weights)

    original_loss = forward_propagation(inputs, outputs, weights_original)

    # Loop through the network layers
    for i, layer in enumerate(weights_original):
        # Now within each layer loop through the actual weights
        for index, weight in np.ndenumerate(layer):

            # store a deep copy of original weights
            weights_temp = deepcopy(weights)

            # update weights
            weights_temp[i][index] += 0.0001

            # calculate new loss after updating weights
            _loss_plus = forward_propagation(inputs, outputs, weights_temp)

            grad = (_loss_plus - original_loss) / (0.0001)

            """ Note that what I am doing in the above 3 lines are - updating a weight by a very small amount and then calculating the gradient i.e. Change of Loss wrt change in weight.

            This is equivalent to the process of differentiation. So here I am doing it very manually and NOT using Calculus Rules.

            Finally, we update the parameter present in the
            corresponding ith layer and index, of weights_updated which is captured by

            weights_updated[i][index]

            The updated weight value will be reduced in proportion to the value of the
            gradient. Further, instead of completely reducing it by a value equal to
            the gradient value, we bring in a mechanism to build trust slowly by
            using the learning rate – lr

            So the formulae is

            Updated W = Original W - LR * Gradient of Loss wrt Weight

            """

            # Updated W = Original W - LR * Gradient of Loss wrt Weight
            weights_updated[i][index] -= grad * lr

    return weights_updated, original_loss


x = np.array([[1, 1]])
y = np.array([[0]])

# Randomly created weight matrix that will be fed to the network initially
weights = [
    np.array(
        [[-0.0053, 0.3793], [-0.5820, -0.5204], [-0.2723, 0.1896]], dtype=np.float32
    ).T,
    np.array([-0.0140, 0.5607, -0.0628], dtype=np.float32),
    np.array([[0.1528, -0.1745, -0.1135]], dtype=np.float32).T,
    np.array([-0.5516], dtype=np.float32),
]

losses = []

# Now I run the train_weights function for 100 epochs and for
# each epoch I take the loss for that epoch and append to an array
# Note the train_weights function returns updated_weights and the original_loss for the respective epoch

for epoch in range(100):
    # Note for each consecutive epoch I am passing the updated weights
    # from previous epoch
    weights, loss_org = train_weights_gradient_descent(x, y, weights, 0.01)
    losses.append(loss_org)

plt.plot(losses)
plt.title("Loss over epochs")


# At last to run this .py file from any other jupyter notebook (so that I can see the plot)
# Just run below command. https://ipython.org/ipython-doc/3/interactive/magics.html#magic-run
# %run -i ./neural_network_from_scratch.py
