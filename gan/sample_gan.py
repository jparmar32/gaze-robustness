### just need to sample, ten take some thins from generatioss
import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision

from generator import Generator_Advanced_64, Generator_Basic


#impot generator and load in svaed one 

generator = Generator_Advanced_64().cuda()
generator.load_state_dict(torch.load('./experiment_one/generator_ckpt_90.pt'))

# Noise to generate images from.
# It should be a batch of vectors with 1000 channels.
noise = torch.randn(32, 100, 1, 1).cuda()

# Feed noise into the generator to create new images
images = generator(noise).detach()

print(images.shape)


grid_img = torchvision.utils.make_grid(images, nrow=8)
torchvision.utils.save_image(grid_img, './experiment_one/final_generated_cxr.png')