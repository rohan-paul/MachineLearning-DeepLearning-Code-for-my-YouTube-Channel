import torch
from torch import nn

torch.manual_seed(0)

##############################################
# Generator
##############################################


class Generator(nn.Module):
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
        x = noise.view(len(noise), self.z_dim, 1, 1)
        """ view(dim1,dim2,...) returns a view of the same underlying information,
        but reshaped to a tensor of shape dim1 x dim2 x ... """
        return self.generator(x)


##############################
# Critic
##############################
class Critic(nn.Module):
    def __init__(self, im_chan=1, hidden_dim=64):
        super(Critic, self).__init__()
        self.critic = nn.Sequential(
            self.make_crit_block(im_chan, hidden_dim),
            self.make_crit_block(hidden_dim, hidden_dim * 2),
            self.make_crit_block(hidden_dim * 2, 1, final_layer=True),
        )

    def make_crit_block(
        self,
        input_channels,
        output_channels,
        kernel_size=4,
        stride=2,
        final_layer=False,
    ):

        if not final_layer:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride),
                nn.BatchNorm2d(output_channels),
                nn.LeakyReLU(0.2, inplace=True),
            )
        else:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride),
            )

    def forward(self, image):
        crit_pred = self.critic(image)
        return crit_pred.view(len(crit_pred), -1)
