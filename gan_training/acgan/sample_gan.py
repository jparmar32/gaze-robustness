### just need to sample, ten take some thins from generatioss
import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision

from generator import _netG

#import generator and load in svaed one 
generator = _netG(1, 110).cuda()
generator.load_state_dict(torch.load('./experiment_one/generator_ckpt_99.pt'))

# Noise to generate images from.
# It should be a batch of vectors with 1000 channels.

noise = torch.randn(32, 110, 1, 1).cuda()

# Get labels ranging from 0 to n_classes for n rows
labels = np.array([num for _ in range(8) for num in range(2)])
labels = torch.Tensor(labels).cuda()
gen_imgs = generator(noise).detach()

print(gen_imgs)

grid_img = torchvision.utils.make_grid(gen_imgs, nrow=2)
torchvision.utils.save_image(grid_img, './experiment_one/final_generated_cxr.png')