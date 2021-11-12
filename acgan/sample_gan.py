### just need to sample, ten take some thins from generatioss
import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision

from generator import Generator

#import generator and load in svaed one 
generator = Generator().cuda()
generator.load_state_dict(torch.load('./generator_ckpt_99.pt'))

# Noise to generate images from.
# It should be a batch of vectors with 1000 channels.

z = torch.randn((8 ** 2, 100)).cuda()

# Get labels ranging from 0 to n_classes for n rows
labels = np.array([num for _ in range(8) for num in range(2)])
labels = torch.Tensor(labels).cuda()
gen_imgs = generator(z, labels).detach()


grid_img = torchvision.utils.make_grid(gen_imgs, nrow=2)
torchvision.utils.save_image(grid_img, './final_generated_cxr.png')