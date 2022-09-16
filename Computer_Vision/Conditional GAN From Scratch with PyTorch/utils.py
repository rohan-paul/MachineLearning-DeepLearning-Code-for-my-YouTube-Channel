import torch
from torch import nn
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import torch.nn.functional as F

torch.manual_seed(0)  # Set for our testing purposes, please do not change!


def plot_images_from_tensor(
    image_tensor, num_images=25, size=(1, 28, 28), nrow=5, show=True
):
    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=nrow)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
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
    combined = torch.cat((x.float(), y.float()), 1)
    return combined


def calculate_input_dim(z_dim, mnist_shape, n_classes):
    """mnist_shape = (1, 28, 28)
    n_classes = 10"""
    generator_input_dim = z_dim + n_classes

    # mnist_shape[0] is 1 as its grayscale images
    discriminator_image_channel = mnist_shape[0] + n_classes

    return generator_input_dim, discriminator_image_channel
