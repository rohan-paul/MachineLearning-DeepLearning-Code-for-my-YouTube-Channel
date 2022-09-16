import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from conditional_gan import *
from utils import *


mnist_shape = (1, 28, 28)
n_classes = 10


criterion = nn.BCEWithLogitsLoss()
n_epochs = 200
z_dim = 64
display_step = 500
batch_size = 128
lr = 0.0002
device = "cuda"

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)

dataloader = DataLoader(
    MNIST(
        "/content/drive/MyDrive/All_Datasets/MNIST", download=False, transform=transform
    ),
    batch_size=batch_size,
    shuffle=True,
)


generator_input_dim, discriminator_image_channel = calculate_input_dim(
    z_dim, mnist_shape, n_classes
)

gen = Generator(input_dim=generator_input_dim).to(device)

gen_opt = torch.optim.Adam(gen.parameters(), lr=lr)

disc = Discriminator(image_channel=discriminator_image_channel).to(device)

disc_opt = torch.optim.Adam(disc.parameters(), lr=lr)


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)


gen = gen.apply(weights_init)

disc = disc.apply(weights_init)


cur_step = 0
generator_losses = []
discriminator_losses = []

noise_and_labels = False
fake = False

fake_image_and_labels = False
real_image_and_labels = False
disc_fake_pred = False
disc_real_pred = False

for epoch in range(n_epochs):
    # Dataloader returns the batches and the labels
    for real, labels in tqdm(dataloader):
        cur_batch_size = len(real)
        # Flatten the batch of real images from the dataset
        real = real.to(device)

        # create one hot encoded vectors from labels and n_classes
        one_hot_labels = ohe_vector_from_labels(labels.to(device), n_classes)
        print("one_hot_labels ", one_hot_labels.size())  # => torch.Size([128, 10])

        """ The above ([128, 10]) need to be converted to ([128, 10, 28, 28])

        Because, Concatenation of Multiple Tensor with `torch.cat()` - RULE - To concatenate WITH torch.cat(), where the list of tensors are concatenated across the specified dimensions, requires 2 conditions to be satisfied

         1. All tensors need to have the same number of dimensions and
         2. All dimensions except the one that they are concatenated on, need to have the same size.

        To do that, first I am adding extra dimension with 'None'
        the easiest way to add extra dimensions to an array is by using the keyword None,
        when indexing at the position to add the extra dimension.
        Note, in below with keyword None, I am only adding extra dummy empty dimension

        a = torch.rand(1, 2)
        ic(a) # => tensor([[0.1749, 0.6387]])
        ic(a[None, :]) # => tensor([[[0.1749, 0.6387]]])

        a = torch.rand([1,2,3,4])
        ic(a.shape) # => torch.Size([1, 2, 3, 4])
        ic(a[None, :].shape) # => torch.Size([1, 1, 2, 3, 4])
        """
        image_one_hot_labels = one_hot_labels[:, :, None, None]
        print(
            "image_one_hot_labels.size ", image_one_hot_labels.size()
        )  # => torch.Size([128, 10, 1, 1])

        image_one_hot_labels = image_one_hot_labels.repeat(
            1, 1, mnist_shape[1], mnist_shape[2]
        )
        print(
            "image_one_hot_labels.size ", image_one_hot_labels.size()
        )  # => torch.Size([128, 10, 28, 28])

        #########################
        #  Train Discriminator
        #########################
        # Zero out the discriminator gradients
        disc_opt.zero_grad()
        # Get noise corresponding to the current batch_size
        fake_noise = create_noise_vector(cur_batch_size, z_dim, device=device)

        # Now we can get the images from the generator
        # Steps: 1) Combine the noise vectors and the one-hot labels for the generator
        #        2) Generate the conditioned fake images

        noise_and_labels = concat_vectors(fake_noise, one_hot_labels)
        fake = gen(noise_and_labels)

        # Make sure that enough images were generated
        assert len(fake) == len(real)

        # Now we can get the predictions from the discriminator
        # Steps: 1) Create the input for the discriminator
        #           a) Combine the fake images with image_one_hot_labels,
        #              remember to detach the generator (.detach()) so we do not backpropagate
        #              through it
        #           b) Combine the real images with image_one_hot_labels
        #        2) Get the discriminator's prediction on the fakes as disc_fake_pred
        #        3) Get the discriminator's prediction on the reals as disc_real_pred

        # Combine the fake images with image_one_hot_labels
        fake_image_and_labels = concat_vectors(fake, image_one_hot_labels)

        # Combine the real images with image_one_hot_labels
        real_image_and_labels = concat_vectors(real, image_one_hot_labels)

        # Get the discriminator's prediction on the reals and fakes
        disc_fake_pred = disc(fake_image_and_labels.detach())
        disc_real_pred = disc(real_image_and_labels)

        # Make sure that enough predictions were made
        assert len(disc_real_pred) == len(real)
        # Make sure that the inputs are different
        assert torch.any(fake_image_and_labels != real_image_and_labels)

        # Calculate Discriminator Loss on fakes and reals
        disc_fake_loss = criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
        disc_real_loss = criterion(disc_real_pred, torch.ones_like(disc_real_pred))

        # Get average Discriminator Loss
        disc_loss = (disc_fake_loss + disc_real_loss) / 2

        # Backpropagate and update weights
        disc_loss.backward(retain_graph=True)
        disc_opt.step()

        # Keep track of the average discriminator loss
        discriminator_losses += [disc_loss.item()]

        #########################
        #  Train Generators
        #########################

        gen_opt.zero_grad()

        fake_image_and_labels = concat_vectors(fake, image_one_hot_labels)
        # This will error if we didn't concatenate wer labels to wer image correctly
        disc_fake_pred = disc(fake_image_and_labels)

        """ Now calculate Generator Loss and note that, here, unlike the disc_loss, with
        disc_fake_pred, I am passing a vector containing its elements as 1 with torch.ones_like
        Because, Generator wants to fool the Discriminator by telling it that all these fake images are actually real, i.e. with value of 1
        """
        gen_loss = criterion(disc_fake_pred, torch.ones_like(disc_fake_pred))

        # Backpropagate and update weights
        gen_loss.backward()
        gen_opt.step()

        # Keep track of the generator losses
        generator_losses += [gen_loss.item()]

        ##################################
        #  Log Progress and Visualization
        #  for each display_step = 50
        ##################################
        if cur_step % display_step == 0 and cur_step > 0:
            # Calculate Generator Mean loss for the latest display_steps (i.e. latest 50 steps)
            # list[-x:]   # last x items in the array
            gen_mean = sum(generator_losses[-display_step:]) / display_step

            # Calculate Discriminator Mean loss for the latest display_steps (i.e. latest 50 steps)
            disc_mean = sum(discriminator_losses[-display_step:]) / display_step
            print(
                f"Step {cur_step}: Generator loss: {gen_mean}, discriminator loss: {disc_mean}"
            )

            # Plot both the real images and fake generated images
            plot_images_from_tensor(fake)
            plot_images_from_tensor(real)

            step_bins = 20
            x_axis = sorted(
                [i * step_bins for i in range(len(generator_losses) // step_bins)]
                * step_bins
            )
            num_examples = (len(generator_losses) // step_bins) * step_bins
            plt.plot(
                range(num_examples // step_bins),
                torch.Tensor(generator_losses[:num_examples])
                .view(-1, step_bins)
                .mean(1),
                label="Generator Loss",
            )
            plt.plot(
                range(num_examples // step_bins),
                torch.Tensor(discriminator_losses[:num_examples])
                .view(-1, step_bins)
                .mean(1),
                label="Discriminator Loss",
            )
            plt.legend()
            plt.show()
        elif cur_step == 0:
            print("Let Long Training Continue")
        cur_step += 1
