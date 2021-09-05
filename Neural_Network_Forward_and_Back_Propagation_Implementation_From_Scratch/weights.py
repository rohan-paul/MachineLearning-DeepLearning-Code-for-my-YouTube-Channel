import numpy as np

weights = [
    np.array(
        [[-0.0053, 0.3793], [-0.5820, -0.5204], [-0.2723, 0.1896]], dtype=np.float32
    ).T,
    np.array([-0.0140, 0.5607, -0.0628], dtype=np.float32),
    np.array([[0.1528, -0.1745, -0.1135]], dtype=np.float32).T,
    np.array([-0.5516], dtype=np.float32),
]

# [print(w) for w in weights]
print(weights[0])  # weight values connection input layers to hidden layers

print(weights[1])  # Bias terms associated with the hidden layers node

print(weights[2])  # weights connecting the hidden layer to the output layer

print(weights[3])  # Bias associated with the Final output layer value
