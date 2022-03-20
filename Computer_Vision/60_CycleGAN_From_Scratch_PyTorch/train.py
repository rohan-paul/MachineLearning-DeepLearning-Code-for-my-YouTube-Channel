import numpy as np
import itertools
import time
import datetime

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.utils import make_grid
import torch.nn.functional as F
import torch

from matplotlib.pyplot import figure
from IPython.display import clear_output

from PIL import Image
import matplotlib.image as mpimg

from utils import *
from cyclegan import *

cuda = True if torch.cuda.is_available() else False
print("Using CUDA" if cuda else "Not using CUDA")

""" So generally both torch.Tensor and torch.cuda.Tensor are equivalent. You can do everything you like with them both.
The key difference is just that torch.Tensor occupies CPU memory while torch.cuda.Tensor occupies GPU memory.
Of course operations on a CPU Tensor are computed with CPU while operations for the GPU / CUDA Tensor are computed on GPU. """
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor


##############################################
# Defining all hyperparameters
##############################################


class Hyperparameters(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


hp = Hyperparameters(
    epoch=0,
    n_epochs=200,
    dataset_train_mode="train",
    dataset_test_mode="test",
    batch_size=4,
    lr=0.0002,
    decay_start_epoch=100,
    b1=0.5,
    b2=0.999,
    n_cpu=8,
    img_size=128,
    channels=3,
    n_critic=5,
    sample_interval=100,
    num_residual_blocks=19,
    lambda_cyc=10.0,
    lambda_id=5.0,
)

##############################################
# Setting Root Path for Google Drive or Kaggle
##############################################

# Root Path for Google Drive
root_path = "/content/drive/MyDrive/All_Datasets/summer2winter_yosemite"

# Root Path for Kaggle
# root_path = '../input/summer2winter-yosemite'


########################################################
# Methods for Image Visualization
########################################################
def show_img(img, size=10):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.figure(figsize=(size, size))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


"""  The reason for doing "np.transpose(npimg, (1, 2, 0))"

PyTorch modules processing image data expect tensors in the format C × H × W.
Whereas PILLow and Matplotlib expect image arrays in the format H × W × C
so to use them with matplotlib you need to reshape it
to put the channels as the last dimension:

I could have used permute() method as well like below
plt.imshow(pytorch_tensor_image.permute(1, 2, 0))
"""


def to_img(x):
    x = x.view(x.size(0) * 2, hp.channels, hp.img_size, hp.img_size)
    return x


def plot_output(path, x, y):
    img = mpimg.imread(path)
    plt.figure(figsize=(x, y))
    plt.imshow(img)
    plt.show()


##############################################
# Defining Image Transforms to apply
##############################################
transforms_ = [
    transforms.Resize((hp.img_size, hp.img_size), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

train_dataloader = DataLoader(
    ImageDataset(root_path, mode=hp.dataset_train_mode, transforms_=transforms_),
    batch_size=hp.batch_size,
    shuffle=True,
    num_workers=1,
)
val_dataloader = DataLoader(
    ImageDataset(root_path, mode=hp.dataset_test_mode, transforms_=transforms_),
    batch_size=16,
    shuffle=True,
    num_workers=1,
)

##############################################
# SAMPLING IMAGES
##############################################


def save_img_samples(batches_done):
    """Saves a generated sample from the test set"""
    print("batches_done ", batches_done)
    imgs = next(iter(val_dataloader))

    Gen_AB.eval()
    Gen_BA.eval()

    real_A = Variable(imgs["A"].type(Tensor))
    fake_B = Gen_AB(real_A)
    real_B = Variable(imgs["B"].type(Tensor))
    fake_A = Gen_BA(real_B)
    # Arange images along x-axis
    real_A = make_grid(real_A, nrow=16, normalize=True)
    real_B = make_grid(real_B, nrow=16, normalize=True)
    fake_A = make_grid(fake_A, nrow=16, normalize=True)
    fake_B = make_grid(fake_B, nrow=16, normalize=True)
    # Arange images along y-axis
    image_grid = torch.cat((real_A, fake_B, real_B, fake_A), 1)

    path = root_path + "/%s.png" % (batches_done)  # Path when running in Google Colab

    # path =  '/kaggle/working' + "/%s.png" % (batches_done)    # Path when running inside Kaggle
    save_image(image_grid, path, normalize=False)
    return path


##############################################
# SETUP, LOSS, INITIALIZE MODELS and BUFFERS
##############################################

# Creating criterion object (Loss Function) that will
# measure the error between the prediction and the target.
criterion_GAN = torch.nn.MSELoss()

criterion_cycle = torch.nn.L1Loss()

criterion_identity = torch.nn.L1Loss()

input_shape = (hp.channels, hp.img_size, hp.img_size)

##############################################
# Initialize generator and discriminator
##############################################

Gen_AB = GeneratorResNet(input_shape, hp.num_residual_blocks)
Gen_BA = GeneratorResNet(input_shape, hp.num_residual_blocks)

Disc_A = Discriminator(input_shape)
Disc_B = Discriminator(input_shape)

if cuda:
    Gen_AB = Gen_AB.cuda()
    Gen_BA = Gen_BA.cuda()
    Disc_A = Disc_A.cuda()
    Disc_B = Disc_B.cuda()
    criterion_GAN.cuda()
    criterion_cycle.cuda()
    criterion_identity.cuda()

##############################################
# Initialize weights
##############################################

Gen_AB.apply(initialize_conv_weights_normal)
Gen_BA.apply(initialize_conv_weights_normal)

Disc_A.apply(initialize_conv_weights_normal)
Disc_B.apply(initialize_conv_weights_normal)


##############################################
# Buffers of previously generated samples
##############################################

fake_A_buffer = ReplayBuffer()

fake_B_buffer = ReplayBuffer()


##############################################
# Defining all Optimizers
##############################################
optimizer_G = torch.optim.Adam(
    itertools.chain(Gen_AB.parameters(), Gen_BA.parameters()),
    lr=hp.lr,
    betas=(hp.b1, hp.b2),
)
optimizer_Disc_A = torch.optim.Adam(Disc_A.parameters(), lr=hp.lr, betas=(hp.b1, hp.b2))

optimizer_Disc_B = torch.optim.Adam(Disc_B.parameters(), lr=hp.lr, betas=(hp.b1, hp.b2))


##############################################
# Learning rate update schedulers
##############################################
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
    optimizer_G, lr_lambda=LambdaLR(hp.n_epochs, hp.epoch, hp.decay_start_epoch).step
)

lr_scheduler_Disc_A = torch.optim.lr_scheduler.LambdaLR(
    optimizer_Disc_A,
    lr_lambda=LambdaLR(hp.n_epochs, hp.epoch, hp.decay_start_epoch).step,
)

lr_scheduler_Disc_B = torch.optim.lr_scheduler.LambdaLR(
    optimizer_Disc_B,
    lr_lambda=LambdaLR(hp.n_epochs, hp.epoch, hp.decay_start_epoch).step,
)

##############################################
# Final Training Function
##############################################


def train(
    Gen_BA,
    Gen_AB,
    Disc_A,
    Disc_B,
    train_dataloader,
    n_epochs,
    criterion_identity,
    criterion_cycle,
    lambda_cyc,
    criterion_GAN,
    optimizer_G,
    fake_A_buffer,
    fake_B_buffer,
    clear_output,
    optimizer_Disc_A,
    optimizer_Disc_B,
    Tensor,
    sample_interval,
    lambda_id,
):
    # TRAINING
    prev_time = time.time()
    for epoch in range(hp.epoch, n_epochs):
        for i, batch in enumerate(train_dataloader):

            # Set model input
            real_A = Variable(batch["A"].type(Tensor))
            real_B = Variable(batch["B"].type(Tensor))

            # Adversarial ground truths i.e. target vectors
            # 1 for real images and 0 for fake generated images
            valid = Variable(
                Tensor(np.ones((real_A.size(0), *Disc_A.output_shape))),
                requires_grad=False,
            )

            fake = Variable(
                Tensor(np.zeros((real_A.size(0), *Disc_A.output_shape))),
                requires_grad=False,
            )

            #########################
            #  Train Generators
            #########################

            Gen_AB.train()
            Gen_BA.train()

            """
            PyTorch stores gradients in a mutable data structure. So we need to set it to a clean state before we use it.
            Otherwise, it will have old information from a previous iteration.
            """
            optimizer_G.zero_grad()

            # Identity loss
            # First pass real_A images to the Genearator, that will generate A-domains images
            loss_id_A = criterion_identity(Gen_BA(real_A), real_A)

            # Then pass real_B images to the Genearator, that will generate B-domains images
            loss_id_B = criterion_identity(Gen_AB(real_B), real_B)

            loss_identity = (loss_id_A + loss_id_B) / 2

            # GAN losses for GAN_AB
            fake_B = Gen_AB(real_A)

            loss_GAN_AB = criterion_GAN(Disc_B(fake_B), valid)

            # GAN losses for GAN_BA
            fake_A = Gen_BA(real_B)

            loss_GAN_BA = criterion_GAN(Disc_A(fake_A), valid)

            loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

            # Cycle Consistency losses
            reconstructed_A = Gen_BA(fake_B)

            """
            Forward Cycle Consistency Loss
            Forward cycle loss:  lambda * ||G_BtoA(G_AtoB(A)) - A|| (Equation 2 in the paper)

            Compute the cycle consistency loss by comparing the reconstructed_A images with real real_A  images of domain A.
            Lambda for cycle loss is 10.0. Penalizing 10 times and forcing to learn the translation.
            """
            loss_cycle_A = criterion_cycle(reconstructed_A, real_A)

            """
            Backward Cycle Consistency Loss
            Backward cycle loss: lambda * ||G_AtoB(G_BtoA(B)) - B|| (Equation 2 of the Paper)
            Compute the cycle consistency loss by comparing the reconstructed_B images with real real_B images of domain B.
            Lambda for cycle loss is 10.0. Penalizing 10 times and forcing to learn the translation.
            """
            reconstructed_B = Gen_AB(fake_A)

            loss_cycle_B = criterion_cycle(reconstructed_B, real_B)

            loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

            """
            Finally, Total Generators Loss and Back propagation
            Add up all the Generators loss and cyclic loss (Equation 3 of paper.
            Also Equation I the code representation of the equation) and perform backpropagation with optimization.
            """
            loss_G = loss_GAN + lambda_cyc * loss_cycle + lambda_id * loss_identity

            loss_G.backward()

            """
            Now we just need to update all the parameters!
            Θ_{k+1} = Θ_k - η * ∇_Θ ℓ(y_hat, y)
            """
            optimizer_G.step()

            #########################
            #  Train Discriminator A
            #########################

            optimizer_Disc_A.zero_grad()

            # Real loss
            loss_real = criterion_GAN(Disc_A(real_A), valid)
            # Fake loss (on batch of previously generated samples)

            fake_A_ = fake_A_buffer.push_and_pop(fake_A)

            loss_fake = criterion_GAN(Disc_A(fake_A_.detach()), fake)

            """ Total loss for Disc_A
            And I divide by 2 because as per Paper - "we divide the objective by 2 while
            optimizing D, which slows down the rate at which D learns,
            relative to the rate of G."
            """
            loss_Disc_A = (loss_real + loss_fake) / 2

            """ do backpropagation i.e.
            ∇_Θ will get computed by this call below to backward() """
            loss_Disc_A.backward()

            """
            Now we just need to update all the parameters!
            Θ_{k+1} = Θ_k - η * ∇_Θ ℓ(y_hat, y)
            """
            optimizer_Disc_A.step()

            #########################
            #  Train Discriminator B
            #########################

            optimizer_Disc_B.zero_grad()

            # Real loss
            loss_real = criterion_GAN(Disc_B(real_B), valid)

            # Fake loss (on batch of previously generated samples)
            fake_B_ = fake_B_buffer.push_and_pop(fake_B)

            loss_fake = criterion_GAN(Disc_B(fake_B_.detach()), fake)

            """ Total loss for Disc_B
            And I divide by 2 because as per Paper - "we divide the objective by 2 while
            optimizing D, which slows down the rate at which D learns,
            relative to the rate of G."
            """
            loss_Disc_B = (loss_real + loss_fake) / 2

            """ do backpropagation i.e.
            ∇_Θ will get computed by this call below to backward() """
            loss_Disc_B.backward()

            """
            Now we just need to update all the parameters!
            Θ_{k+1} = Θ_k − η * ∇_Θ ℓ(y_hat, y)
            """
            optimizer_Disc_B.step()

            loss_D = (loss_Disc_A + loss_Disc_B) / 2

            ##################
            #  Log Progress
            ##################

            # Determine approximate time left
            batches_done = epoch * len(train_dataloader) + i

            batches_left = n_epochs * len(train_dataloader) - batches_done

            time_left = datetime.timedelta(
                seconds=batches_left * (time.time() - prev_time)
            )
            prev_time = time.time()

            print(
                "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, adv: %f, cycle: %f, identity: %f] ETA: %s"
                % (
                    epoch,
                    n_epochs,
                    i,
                    len(train_dataloader),
                    loss_D.item(),
                    loss_G.item(),
                    loss_GAN.item(),
                    loss_cycle.item(),
                    loss_identity.item(),
                    time_left,
                )
            )

            # If at sample interval save image
            if batches_done % sample_interval == 0:
                clear_output()
                plot_output(save_img_samples(batches_done), 30, 40)


##############################################
# Execute the Final Training Function
##############################################

train(
    Gen_BA=Gen_BA,
    Gen_AB=Gen_AB,
    Disc_A=Disc_A,
    Disc_B=Disc_B,
    train_dataloader=train_dataloader,
    n_epochs=hp.n_epochs,
    criterion_identity=criterion_identity,
    criterion_cycle=criterion_cycle,
    lambda_cyc=hp.lambda_cyc,
    criterion_GAN=criterion_GAN,
    optimizer_G=optimizer_G,
    fake_A_buffer=fake_A_buffer,
    fake_B_buffer=fake_B_buffer,
    clear_output=clear_output,
    optimizer_Disc_A=optimizer_Disc_A,
    optimizer_Disc_B=optimizer_Disc_B,
    Tensor=Tensor,
    sample_interval=hp.sample_interval,
    lambda_id=hp.lambda_id,
)
