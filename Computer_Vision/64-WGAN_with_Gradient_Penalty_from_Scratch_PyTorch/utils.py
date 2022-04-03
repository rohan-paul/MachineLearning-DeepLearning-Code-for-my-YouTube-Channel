import torch
from torch import nn
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

torch.manual_seed(0)  # Set for testing purposes, please do not change!


def plot_images_from_tensor(image_tensor, num_images=25, size=(1, 28, 28)):
    image_tensor = (image_tensor + 1) / 2
    img_detached = image_tensor.detach().cpu()
    # The cpu() operation transfers the tensor to the CPU (if not already there),
    # while detach() cuts the computation graph.
    image_grid = make_grid(img_detached[:num_images], nrow=5)
    # make_grid() returns a tensor which contains the grid of images.
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()


""" The reason for doing "image_grid.permute(1, 2, 0)"

PyTorch modules processing image data expect tensors in the format C × H × W.

Whereas PILLow and Matplotlib expect image arrays in the format H × W × C

So to use them with matplotlib you need to reshape it
to put the channels as the last dimension:

I could have used permute() method as well like below
"np.transpose(npimg, (1, 2, 0))"

------------------

Tensor.detach() is used to detach a tensor from the current computational graph. It returns a new tensor that doesn't require a gradient.

When we don't need a tensor to be traced for the gradient computation, we detach the tensor from the current computational graph.

We also need to detach a tensor when we need to move the tensor from GPU to CPU.

"""


def make_grad_hook():
    gradients_list = []

    def grad_hook(m):
        """isinstance(object, type)
        The isinstance() function returns True if the specified object is of the specified type, otherwise False."""
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            gradients_list.append(m.weight.grad)

    return gradients_list, grad_hook


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)


def get_noise(n_samples, z_dim, device="cpu"):
    """torch.rand(*sizes, out=None) → Returns a tensor containing a set of random numbers drawn from
    the uniform distribution in the interval [0,1), the shape is defined by the variable parameter sizes."""
    return torch.randn(n_samples, z_dim, device=device)


##############################
# Generator Loss Calculation
##############################
"""
#### Generator Loss = -[average critic score on fake images]

Generator Loss: D(G(z))

The generator tries to maximize this function. In other words,
It tries to maximize the discriminator's output for its fake instances. In these functions: """


def get_gen_loss(critic_fake_prediction):
    gen_loss = -1.0 * torch.mean(critic_fake_prediction)
    return gen_loss


# UNIT TEST of Generator Loss Calculation
assert torch.isclose(get_gen_loss(torch.tensor(1.0)), torch.tensor(-1.0))

assert torch.isclose(get_gen_loss(torch.rand(10000)), torch.tensor(-0.5), 0.05)

print("Success!")

##############################
# Critic Loss Calculation
##############################
def get_crit_loss(critic_fake_prediction, crit_real_pred, gp, c_lambda):

    """The math for the loss functions for the critic and generator is:

    Critic Loss: D(x) - D(G(z))

    Now for the Critic Loss, as per the Paper, we have to maximize the above expression.
    So, arithmetically, maximizing an expression, means minimizing the -ve of that expression

    i.e. -(D(x) - D(G(z)))
    i.e. -D(x) + D(G(z))
    i.e. -D(real_imgs) + D(G(real_imgs))
    i.e. -D(real_imgs) + D(fake_imgs)
    """
    crit_loss = (
        torch.mean(critic_fake_prediction) - torch.mean(crit_real_pred) + c_lambda * gp
    )
    return crit_loss


# UNIT TEST of Critic Loss Calculation
assert torch.isclose(
    get_crit_loss(torch.tensor(1.0), torch.tensor(2.0), torch.tensor(3.0), 0.1),
    torch.tensor(-0.7),
)
assert torch.isclose(
    get_crit_loss(torch.tensor(20.0), torch.tensor(-20.0), torch.tensor(2.0), 10),
    torch.tensor(60.0),
)

print("Success!")
