import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
import torchvision
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

from tqdm import tqdm

from wgan.generator import Generator_Advanced_64, Generator_Basic
from wgan.discriminator import Discriminator_Advanced_64, Discriminator_Basic
from dataloader import fetch_entire_dataloader

cuda = True if torch.cuda.is_available() else False

dl = fetch_entire_dataloader("cxr_p","/media",0.2,2,32,4, gaze_task="cam_reg_convex", gan = True)

# Initialize generator and discriminator
generator = Generator_Advanced_64()
discriminator = Discriminator_Advanced_64()

if cuda:
    generator.cuda()
    discriminator.cuda()

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


# ----------
#  Training
# ----------

for epoch in range(100):
    for i, (imgs, _) in tqdm(enumerate(dl), total = len(dl.dataset)//32 + 1):
        # imgs = imgs[:, 1, :, :].reshape(imgs.shape[0], 1, imgs.shape[2], imgs.shape[3])
        batch_size = imgs.size(0)

        # Adversarial ground truths
        valid = Tensor(imgs.size(0), 1, 1, 1).fill_(1.0)
        valid.requires_grad = False 
        fake = Tensor(imgs.size(0), 1, 1, 1).fill_(0.0)
        fake.requires_grad = False 

        # Configure input
        real_imgs = imgs.type(Tensor)

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise as generator input
        z = torch.randn((imgs.shape[0], 100, 1, 1), dtype=real_imgs.dtype, device=real_imgs.device)
        

        # Generate a batch of images
        gen_imgs = generator(z)

        # Loss measures generator's ability to fool the discriminator
        g_loss = -torch.mean(discriminator(gen_imgs))

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        alpha = torch.rand(real_imgs.shape[0], 1, 1, 1, dtype=real_imgs.dtype, device=real_imgs.device)
        x_fake = generator(z)
        r = alpha * real_imgs + (1 - alpha) * x_fake
        d_output = discriminator(r)
        gradient_norm = torch.autograd.grad(d_output.sum(), r, create_graph=True)[0].view(batch_size, -1).norm(2, dim=1)
        d_loss = -torch.mean(discriminator(real_imgs)) + torch.mean(discriminator(x_fake)) + 10 * torch.mean((gradient_norm - 1) ** 2)

        d_loss.backward()
        optimizer_D.step()

    print(f"Epoch: {epoch}, Batch: {i}, D loss: {d_loss.item()}, G loss: {g_loss.item()}")
    if epoch % 10 == 0:
        torch.save(generator.state_dict(), f"./wgan/generator_ckpt_{epoch}.pt")
        torch.save(discriminator.state_dict(), f"./wgan/discriminator_ckpt_{epoch}.pt")

        shape = tuple(gen_imgs.shape)

        grid_img = torchvision.utils.make_grid(gen_imgs, nrow=11)
        torchvision.utils.save_image(grid_img, f'./wgan/generated_images_ckpt_{epoch}_cxr.png')

torch.save(generator.state_dict(), f"./wgan/generator_ckpt_{epoch}.pt")
torch.save(discriminator.state_dict(), f"./wgan/discriminator_ckpt_{epoch}.pt")