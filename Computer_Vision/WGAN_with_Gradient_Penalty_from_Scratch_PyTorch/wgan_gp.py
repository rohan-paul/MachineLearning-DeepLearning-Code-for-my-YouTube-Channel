import torch
from torch import nn

torch.manual_seed(0)

##############################################
# Generator
##############################################


class Generator(nn.Module):
    """ The Generator class for the Wasserstein Generative Adversarial Network (WGAN) with gradient penalty.
    This class represents the generator component of the WGAN, which generates images from random noise.

    Args:
    z_dim (int): The dimension of the random noise vector, default is 10.
    im_chan (int): The number of channels in the output images, default is 1.
    hidden_dim (int): The inner dimension, default is 64. """
    def __init__(self, z_dim=10, im_chan=1, hidden_dim=64):
        super(Generator, self).__init__()
        self.z_dim = z_dim

        self.generator = nn.Sequential(
            self.build_gen_block(z_dim, hidden_dim * 4),
            self.build_gen_block(
                hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=1
            ),
            self.build_gen_block(hidden_dim * 2, hidden_dim),
            self.build_gen_block(hidden_dim, im_chan, kernel_size=4, final_layer=True),
        )

    def build_gen_block(
        self,
        input_channels,
        output_channels,
        kernel_size=3,
        stride=2,
        final_layer=False,
    ):
        """
        Returns a sequence of operations corresponding to a generator block;
        a transposed convolution, a batchnorm (except in the final layer), and an activation.
        Parameters:
            input_channels: Number of channels in the input image
            output_channels: Number of channels produced by the convolution.
            kernel_size: the size of each convolutional filter
            stride: the stride of the convolution
            final_layer: a boolean, true if it is the final layer and false otherwise
                      (affects activation and batchnorm)
        """

        if not final_layer:
            return nn.Sequential(
                nn.ConvTranspose2d(
                    input_channels, output_channels, kernel_size, stride
                ),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True),
            )
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(
                    input_channels, output_channels, kernel_size, stride
                ),
                nn.Tanh(),
            )

    def forward(self, noise):
        """
        Forward pass of the generator.

        Args:
        noise (torch.Tensor): A noise tensor with dimensions (n_samples, z_dim).

        Returns:
        torch.Tensor: The generated images.
        """
        x = noise.view(len(noise), self.z_dim, 1, 1)
        """ view(dim1,dim2,...) returns a view of the same underlying information,
        but reshaped to a tensor of shape dim1 x dim2 x ... """
        return self.generator(x)


##############################
# Critic
##############################

class Critic(nn.Module):
    """
    The Critic class for the Wasserstein Generative Adversarial Network (WGAN) with gradient penalty.
    This class represents the critic component of the WGAN, which assesses the quality of images produced by the Generator.

    Args:
    im_chan (int): The number of channels in the input images, default is 1.
    hidden_dim (int): The inner dimension, default is 64.
    """
    def __init__(self, im_chan=1, hidden_dim=64):
        super(Critic, self).__init__()

        # Define the critic network architecture.
        # The critic is designed to assess the quality of images produced by the generator.
        # This critic uses several blocks of convolution, batch normalization, and activation.
        self.critic = nn.Sequential(
            self.make_crit_block(im_chan, hidden_dim), # initial block using image as input
            self.make_crit_block(hidden_dim, hidden_dim * 2), # intermediate block
            self.make_crit_block(hidden_dim * 2, 1, final_layer=True), # final block outputs a single value (quality score)
        )

    def make_crit_block(self, input_channels, output_channels, kernel_size=4, stride=2, final_layer=False):
        """
        Returns a sequence of operations for a critic block: a convolution,
        a batch normalization (except in the final layer), and a leaky ReLU activation.

        Args:
        input_channels (int): The number of input channels.
        output_channels (int): The number of output channels.
        kernel_size (int, optional): The size of the convolutional kernels. Default is 4.
        stride (int, optional): The stride for the convolutional layer. Default is 2.
        final_layer (bool, optional): Whether this block is the final layer. Default is False.

        Returns:
        torch.nn.Sequential: The block as a sequence of operations.
        """
        if not final_layer:
            # For non-final layers, use leaky ReLU activation and batch normalization
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride), # Convolution
                nn.BatchNorm2d(output_channels), # Batch normalization
                nn.LeakyReLU(0.2, inplace=True), # Leaky ReLU activation
            )
        else:
            # For the final layer, use no activation and no batch normalization
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride), # Convolution
            )

    def forward(self, image):
        """
        Forward pass of the critic.

        Args:
        image (torch.Tensor): An image tensor with dimensions (n_samples, im_chan, width, height).

        Returns:
        torch.Tensor: The critic's quality assessment of the image.
        """
        crit_pred = self.critic(image) # Pass the image through the critic
        return crit_pred.view(len(crit_pred), -1) # Flatten the output

