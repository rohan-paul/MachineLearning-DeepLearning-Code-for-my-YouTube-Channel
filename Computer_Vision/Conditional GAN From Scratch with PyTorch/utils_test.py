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
# test_weights_init()


def test_concat_vectors():
    # Create sample input tensors
    x = torch.tensor([[1, 2, 3], [4, 5, 6]])
    y = torch.tensor([[7, 8, 9], [10, 11, 12]])

    # Perform concatenation
    combined = concat_vectors(x, y)

    # Check the output type and shape
    assert isinstance(combined, torch.Tensor)
    assert combined.shape == (2, 6)  # Expected shape after concatenation

    # Check the values in the concatenated tensor
    expected_combined = torch.tensor([[1, 2, 3, 7, 8, 9], [4, 5, 6, 10, 11, 12]])
    assert torch.allclose(combined, expected_combined)

    print("Unit test passed!")

# Run the unit test
# test_concat_vectors()
