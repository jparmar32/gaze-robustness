import argparse
import os
import numpy as np
import math
import torchvision

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

from tqdm import tqdm 
from dataloader import fetch_entire_dataloader
from acgan.generator import _netG, Generator_Advanced_224_Basic, Generator_Advanced_224
from acgan.discriminator import _netD, _netD_224, Discriminator_Advanced_224_Basic, Discriminator_Advanced_224

## use a 224 generator and discriminator
cuda = True if torch.cuda.is_available() else False

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def compute_acc(preds, labels):
    correct = 0
    preds_ = preds.data.max(1)[1]
    correct = preds_.eq(labels.data).cpu().sum()
    acc = float(correct) / float(len(labels.data)) * 100.0
    return acc


# Configure data loader
dl = fetch_entire_dataloader("cxr_p","/media",0.2,2,32,4, gaze_task=None, gan = True, label_class=0)



# some hyper parameters
ngpu = int(1)
nz = int(110)
ngf = int(64)
ndf = int(64)
num_classes = int(2)
nc = 3

netG = Generator_Advanced_224(ngpu, nz)
netG.apply(weights_init_normal)
netD = Discriminator_Advanced_224(ngpu, num_classes)
netD.apply(weights_init_normal)


# loss functions
dis_criterion = nn.BCELoss()
aux_criterion = nn.NLLLoss()

#change to 224
inputs = torch.FloatTensor(32, 1, 224, 224)
noise = torch.FloatTensor(32, nz, 1, 1)
eval_noise = torch.FloatTensor(32, nz, 1, 1).normal_(0, 1)
dis_label = torch.FloatTensor(32)
aux_label = torch.LongTensor(32)
real_label = 1
fake_label = 0

# if using cuda
if cuda:
    netD.cuda()
    netG.cuda()
    dis_criterion.cuda()
    aux_criterion.cuda()
    inputs, dis_label, aux_label = inputs.cuda(), dis_label.cuda(), aux_label.cuda()
    noise, eval_noise = noise.cuda(), eval_noise.cuda()

eval_noise_ = np.random.normal(0, 1, (32, nz))
eval_label = np.random.randint(0, num_classes, 32)
eval_onehot = np.zeros((32, num_classes))
eval_onehot[np.arange(32), eval_label] = 1
eval_noise_[np.arange(32), :num_classes] = eval_onehot[np.arange(32)]
eval_noise_ = (torch.from_numpy(eval_noise_))
eval_noise.data.copy_(eval_noise_.view(32, nz, 1, 1))


# setup optimizer
optimizerD = torch.optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerG = torch.optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))

avg_loss_D = 0.0
avg_loss_G = 0.0
avg_loss_A = 0.0
# ----------
#  Training
# ----------

for epoch in range(100):
    for i, data in tqdm(enumerate(dl), total = len(dl.dataset)//32 + 1):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        netD.zero_grad()
        real_cpu, label = data
        batch_size = real_cpu.size(0)
        if cuda:
            real_cpu = real_cpu.cuda()

        with torch.no_grad():

            inputs.resize_as_(real_cpu).copy_(real_cpu)
            dis_label.resize_(batch_size).fill_(real_label)
            aux_label.resize_(batch_size).copy_(label)
        dis_output, aux_output = netD(inputs)

        dis_errD_real = dis_criterion(dis_output, dis_label)
        aux_errD_real = aux_criterion(aux_output, aux_label)
        errD_real = dis_errD_real + aux_errD_real
        errD_real.backward()
        with torch.no_grad():
            D_x = dis_output.mean()

        # compute the current classification accuracy
        accuracy = compute_acc(aux_output, aux_label)

        # train with fake
        with torch.no_grad():
            noise.resize_(batch_size, nz, 1, 1).normal_(0, 1)
        label = np.random.randint(0, num_classes, batch_size)
        noise_ = np.random.normal(0, 1, (batch_size, nz))
        class_onehot = np.zeros((batch_size, num_classes))
        class_onehot[np.arange(batch_size), label] = 1
        noise_[np.arange(batch_size), :num_classes] = class_onehot[np.arange(batch_size)]
        noise_ = (torch.from_numpy(noise_))
        with torch.no_grad():
            noise.copy_(noise_.view(batch_size, nz, 1, 1))
            aux_label.resize_(batch_size).copy_(torch.from_numpy(label))

        fake = netG(noise)
        with torch.no_grad():
            dis_label.fill_(fake_label)
        dis_output, aux_output = netD(fake.detach())
        dis_errD_fake = dis_criterion(dis_output, dis_label)
        aux_errD_fake = aux_criterion(aux_output, aux_label)
        errD_fake = dis_errD_fake + aux_errD_fake
        errD_fake.backward()
        with torch.no_grad():
            D_G_z1 = dis_output.mean()
        errD = errD_real + errD_fake
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        with torch.no_grad():
            dis_label.fill_(real_label)  # fake labels are real for generator cost
        dis_output, aux_output = netD(fake)
        dis_errG = dis_criterion(dis_output, dis_label)
        aux_errG = aux_criterion(aux_output, aux_label)
        errG = dis_errG + aux_errG
        errG.backward()
        with torch.no_grad():
            D_G_z2 = dis_output.mean()
        optimizerG.step()

        # compute the average loss
        curr_iter = epoch * len(dl) + i
        all_loss_G = avg_loss_G * curr_iter
        all_loss_D = avg_loss_D * curr_iter
        all_loss_A = avg_loss_A * curr_iter
        all_loss_G += errG.item()
        all_loss_D += errD.item()
        all_loss_A += accuracy
        avg_loss_G = all_loss_G / (curr_iter + 1)
        avg_loss_D = all_loss_D / (curr_iter + 1)
        avg_loss_A = all_loss_A / (curr_iter + 1)

    if epoch % 10 == 0:
        print(f"Epoch: {epoch}, Batch: {i}, D loss: {avg_loss_D}, G loss: {avg_loss_G }")
        torch.save(netG.state_dict(), f"./acgan/negative_no_batch/generator_ckpt_{epoch}.pt")
        torch.save(netD.state_dict(), f"./acgan/negative_no_batch/discriminator_ckpt_{epoch}.pt")

        print(fake.shape)

        grid_img = torchvision.utils.make_grid(fake, nrow=11)
        torchvision.utils.save_image(grid_img, f'./acgan/negative_no_batch/generated_images_ckpt_{epoch}_cxr.png')

torch.save(netG.state_dict(), f"./acgan/negative_no_batch/generator_ckpt_{epoch}.pt")
torch.save(netD.state_dict(), f"./acgan/negative_no_batch/discriminator_ckpt_{epoch}.pt")

