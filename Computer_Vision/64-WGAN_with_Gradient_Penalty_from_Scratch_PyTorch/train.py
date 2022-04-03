import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from wgan_gp import *
from utils import *

torch.manual_seed(0)


n_epochs = 100
z_dim = 64
display_step = 50  # Only for visualization of my output during training
batch_size = 128
lr = 0.0002
beta_1 = 0.5
beta_2 = 0.999
c_lambda = 10

crit_repeats = 5
# Number of times the Critic will be trained for each Generator Training

device = "cuda"


transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)

dataloader = DataLoader(
    MNIST(
        "/content/drive/MyDrive/All_Datasets/MNIST", download=True, transform=transform
    ),
    batch_size=batch_size,
    shuffle=True,
)


generator = Generator(z_dim).to(device)
gen_optimizer = torch.optim.Adam(generator.parameters(), lr=lr, betas=(beta_1, beta_2))

critic = Critic().to(device)
critic_optimizer = torch.optim.Adam(critic.parameters(), lr=lr, betas=(beta_1, beta_2))


generator = generator.apply(weights_init)
critic = critic.apply(weights_init)

#######################################################################
# Gradient Penalty Calculation -  Calculate Gradient of Critic Score
#######################################################################
def gradient_of_critic_score(critic, real, fake, epsilon):

    interpolated_images = real * epsilon + fake * (1 - epsilon)

    mixed_scores = critic(interpolated_images)

    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    return gradient


###############################################################################
# Unit Test for above Method
###############################################################################
def test_gradient_of_critic_score(image_shape):
    real = torch.randn(*image_shape, device=device) + 1
    fake = torch.randn(*image_shape, device=device) - 1

    epsilon_shape = [1 for _ in image_shape]  # [1, 1, 1, 1]

    epsilon_shape[0] = image_shape[0]
    epsilon = torch.rand(epsilon_shape, device=device).requires_grad_()

    """ epsilon will be a tensor like below

    tensor([[[[0.4260]]],
            [[[0.5640]]],
            [[[0.2338]]],
            [[[0.3583]]]], requires_grad=True)
    """

    gradient = gradient_of_critic_score(critic, real, fake, epsilon)

    assert tuple(gradient.shape) == image_shape
    assert gradient.max() > 0
    assert gradient.min() < 0
    return gradient


gradient = test_gradient_of_critic_score((256, 1, 28, 28))
print("Success!")

###############################################################################
# Gradient Penalty Calculation - Calculate the Penalty on The Norm of Gradient
###############################################################################
def gradient_penalty_l2_norm(gradient):

    gradient = gradient.view(len(gradient), -1)

    gradient_norm = gradient.norm(2, dim=1)

    penalty = torch.mean((gradient_norm - 1) ** 2)

    return penalty


###############################################################################
# Unit Test for above Method
###############################################################################
def test_gradient_penalty_l2_norm(image_shape):

    bad_gradient = torch.zeros(*image_shape)
    print(bad_gradient)

    bad_gradient_penalty = gradient_penalty_l2_norm(bad_gradient)
    assert torch.isclose(bad_gradient_penalty, torch.tensor(1.0))

    image_size = torch.prod(torch.Tensor(image_shape[1:]))  # 28 * 28 => 784

    print("torch.sqrt(image_size) ", torch.sqrt(image_size))

    good_gradient = torch.ones(*image_shape) / torch.sqrt(image_size)  # => tensor(28.)

    good_gradient_penalty = gradient_penalty_l2_norm(good_gradient)

    assert torch.isclose(good_gradient_penalty, torch.tensor(0.0))

    random_gradient = test_gradient_of_critic_score(image_shape)

    random_gradient_penalty = gradient_penalty_l2_norm(random_gradient)

    assert torch.abs(random_gradient_penalty - 1) < 0.1


test_gradient_penalty_l2_norm((256, 1, 28, 28))
print("Success!")


##############################
# Final Training
##############################

import matplotlib.pyplot as plt

current_step = 0
generator_losses = []
critic_losses_across_critic_repeats = []
for epoch in range(n_epochs):
    # Dataloader returns the batches
    for real, _ in tqdm(dataloader):
        cur_batch_size = len(real)
        real = real.to(device)

        mean_critic_loss_for_this_iteration = 0
        for _ in range(crit_repeats):

            #########################
            #  Train Critic
            #########################
            critic_optimizer.zero_grad()

            fake_noise = get_noise(cur_batch_size, z_dim, device=device)

            fake = generator(fake_noise)

            critic_fake_prediction = critic(fake.detach())

            crit_real_pred = critic(real)

            epsilon = torch.rand(len(real), 1, 1, 1, device=device, requires_grad=True)
            # epsilon will be a Tensor of size torch.Size([128, 1, 1, 1]) for batch_size of 128

            gradient = gradient_of_critic_score(critic, real, fake.detach(), epsilon)

            gp = gradient_penalty_l2_norm(gradient)

            crit_loss = get_crit_loss(
                critic_fake_prediction, crit_real_pred, gp, c_lambda
            )

            # Keep track of the average critic loss in this batch
            mean_critic_loss_for_this_iteration += crit_loss.item() / crit_repeats

            # Update gradients
            crit_loss.backward(retain_graph=True)
            # Update optimizer i.e. the weights
            critic_optimizer.step()
        critic_losses_across_critic_repeats += [mean_critic_loss_for_this_iteration]

        #########################
        #  Train Generators
        #########################
        gen_optimizer.zero_grad()

        fake_noise_2 = get_noise(cur_batch_size, z_dim, device=device)

        fake_2 = generator(fake_noise_2)

        critic_fake_prediction = critic(fake_2)

        gen_loss = get_gen_loss(critic_fake_prediction)

        gen_loss.backward()

        # Update the weights
        gen_optimizer.step()

        # Keep track of the average generator loss
        generator_losses += [gen_loss.item()]

        ##################################
        #  Log Progress and Visualization
        ##################################
        # Do the below visualization for each display_step (i.e. each 50 step)
        if current_step % display_step == 0 and current_step > 0:
            # Calculate Generator Mean loss for the latest display_steps (i.e. latest 50 steps)
            # list[-x:]   # last x items in the array
            generator_mean_loss_display_step = (
                sum(generator_losses[-display_step:]) / display_step
            )

            # Calculate Critic Mean loss for the latest display_steps (i.e. latest 50 steps)
            critic_mean_loss_display_step = (
                sum(critic_losses_across_critic_repeats[-display_step:]) / display_step
            )
            print(
                f"Step {current_step}: Generator loss: {generator_mean_loss_display_step}, critic loss: {critic_mean_loss_display_step}"
            )

            # Plot both the real images and fake generated images
            plot_images_from_tensor(fake)
            plot_images_from_tensor(real)

            step_bins = 20
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
                torch.Tensor(critic_losses_across_critic_repeats[:num_examples])
                .view(-1, step_bins)
                .mean(1),
                label="Critic Loss",
            )
            plt.legend()
            plt.show()

        current_step += 1
