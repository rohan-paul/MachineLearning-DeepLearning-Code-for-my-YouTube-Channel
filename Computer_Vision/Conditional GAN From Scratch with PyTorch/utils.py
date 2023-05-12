import torch
from torch import nn
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import torch.nn.functional as F

torch.manual_seed(0)  # Set for our testing purposes, please do not change!


def plot_images_from_tensor(image_tensor, num_images=25, size=(1, 28, 28), nrow=5, show=True):
    """
    Plots a grid of images from a given tensor.

    The function first scales the image tensor to the range [0, 1]. It then detaches the tensor from the computation
    graph and moves it to the CPU if it's not already there. After that, it creates a grid of images and plots the grid.

    Args:
        image_tensor (torch.Tensor): A 4D tensor containing the images.
            The tensor is expected to be in the shape (batch_size, channels, height, width).
        num_images (int, optional): The number of images to include in the grid. Default is 25.
        size (tuple, optional): The size of a single image in the form of (channels, height, width). Default is (1, 28, 28).
        nrow (int, optional): Number of images displayed in each row of the grid. The final grid size is (num_images // nrow, nrow). Default is 5.
        show (bool, optional): Determines if the plot should be shown. Default is True.

    Returns:
        None. The function outputs a plot of a grid of images.
    """

    # Normalize the image tensor to [0, 1]
    image_tensor = (image_tensor + 1) / 2

    # Detach the tensor from its computation graph and move it to the CPU
    image_unflat = image_tensor.detach().cpu()

    # Create a grid of images using the make_grid function from torchvision.utils
    image_grid = make_grid(image_unflat[:num_images], nrow=nrow)

    # Plot the grid of images
    # The permute() function is used to rearrange the dimensions of the grid for plotting
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())

    # Show the plot if the 'show' parameter is True
    if show:
        plt.show()



""" The reason for doing "image_grid.permute(1, 2, 0)"

PyTorch modules processing image data expect tensors in the format C × H × W.

Whereas PILLow and Matplotlib expect image arrays in the format H × W × C

so to use them with matplotlib you need to reshape it
to put the channels as the last dimension:

I could have used permute() method as well like below
"np.transpose(npimg, (1, 2, 0))"

------------------

Tensor.detach() is used to detach a tensor from the current computational graph. It returns a new tensor that doesn't require a gradient.

When we don't need a tensor to be traced for the gradient computation, we detach the tensor from the current computational graph.

We also need to detach a tensor when we need to move the tensor from GPU to CPU.

"""


def weights_init(m):
    """
    Initialize the weights of convolutional and batch normalization layers.

    Args:
        m (torch.nn.Module): Module instance.

    """
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)


def ohe_vector_from_labels(labels, n_classes):
    return F.one_hot(labels, num_classes=n_classes)


"""
x = torch.tensor([4, 3, 2, 1, 0])
F.one_hot(x, num_classes=6)

# Expected result
# tensor([[0, 0, 0, 0, 1, 0],
#         [0, 0, 0, 1, 0, 0],
#         [0, 0, 1, 0, 0, 0],
#         [0, 1, 0, 0, 0, 0],
#         [1, 0, 0, 0, 0, 0]])
"""


""" Concatenation of Multiple Tensor with `torch.cat()` - RULE - To concatenate WITH torch.cat(), where the list of tensors are concatenated across the specified dimensions, requires 2 conditions to be satisfied

1. All tensors need to have the same number of dimensions and
2. All dimensions except the one that they are concatenated on, need to have the same size. """


def concat_vectors(x, y):
    """
    Concatenate two tensors along the second dimension.

    Args:
        x (torch.Tensor): First input tensor.
        y (torch.Tensor): Second input tensor.

    Returns:
        torch.Tensor: Concatenated tensor.

    """
    combined = torch.cat((x.float(), y.float()), 1)
    return combined

def calculate_input_dim(z_dim, mnist_shape, n_classes):
    """
    Calculate the input dimensions for the generator and discriminator networks.

    Args:
        z_dim (int): Dimension of the random noise vector (latent space).
        mnist_shape (tuple): Shape of the MNIST images, e.g., (1, 28, 28).
        n_classes (int): Number of classes in the dataset.

    Returns:
        tuple: Tuple containing the generator input dimension and discriminator image channel.

    mnist_shape = (1, 28, 28)
    n_classes = 10"""
    generator_input_dim = z_dim + n_classes

    # mnist_shape[0] is 1 as its grayscale images
    discriminator_image_channel = mnist_shape[0] + n_classes

    return generator_input_dim, discriminator_image_channel
