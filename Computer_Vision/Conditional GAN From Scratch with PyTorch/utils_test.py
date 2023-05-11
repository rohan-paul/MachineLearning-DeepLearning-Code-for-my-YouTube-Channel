import torch
import torch.nn as nn

from utils import *

def test_weights_init():
    # Create a sample model with Conv2d and BatchNorm2d layers
    model = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3),
        nn.BatchNorm2d(16),
        nn.ConvTranspose2d(16, 3, kernel_size=3),
        nn.BatchNorm2d(3)
    )

    # Initialize the model weights
    model.apply(weights_init)

    # Check the weights of Conv2d layers
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            assert torch.allclose(module.weight.mean(), torch.tensor(0.0), atol=0.02)
            assert torch.allclose(module.weight.std(), torch.tensor(0.02), atol=0.02)

    # Check the weights of BatchNorm2d layers
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            assert torch.allclose(module.weight.mean(), torch.tensor(0.0), atol=0.02)
            assert torch.allclose(module.weight.std(), torch.tensor(0.02), atol=0.02)
            assert torch.allclose(module.bias, torch.tensor(0.0))

    print("Unit test passed!")

# Run the unit test
test_weights_init()