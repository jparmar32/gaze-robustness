### just need to sample, ten take some thins from generatioss
import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
#impot generator and load in svaed one 

# Noise to generate images from.
# It should be a batch of vectors with 1000 channels.
noise = torch.randn(48, 100, 1, 1)

# Feed noise into the generator to create new images
images = G(noise).detach()
images.shape

# Preview the imagesz
show_images(images)