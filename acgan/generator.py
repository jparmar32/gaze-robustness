import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class _netG(nn.Module):
    def __init__(self, ngpu, nz):
        super(_netG, self).__init__()
        self.ngpu = ngpu
        self.nz = nz

        # first linear layer
        self.fc1 = nn.Linear(110, 768)
        # Transposed Convolution 2 1,1
        self.tconv2 = nn.Sequential(
            nn.ConvTranspose2d(768, 384, 5, 2, 0, bias=False),
            #nn.BatchNorm2d(384),
            nn.LeakyReLU(0.1, inplace=True),
        )
        # Transposed Convolution 3 5,5
        self.tconv3 = nn.Sequential(
            nn.ConvTranspose2d(384, 256, 5, 2, 0, bias=False),
            #nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
        )
        # Transposed Convolution 4 13,13
        self.tconv4 = nn.Sequential(
            nn.ConvTranspose2d(256, 192, 5, 2, 0, bias=False),
            #nn.BatchNorm2d(192),
            nn.LeakyReLU(0.1, inplace=True),
        )
        # Transposed Convolution 5 29,29
        self.tconv5 = nn.Sequential(
            nn.ConvTranspose2d(192, 64, 5, 2, 0, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
        )
        # Transposed Convolution 5 68
        self.tconv6 = nn.Sequential(
            nn.ConvTranspose2d(64, 3, 8, 2, 0, bias=False),
            nn.Tanh(),
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            input = input.view(-1, self.nz)
            fc1 = nn.parallel.data_parallel(self.fc1, input, range(self.ngpu))
            fc1 = fc1.view(-1, 768, 1, 1)
            tconv2 = nn.parallel.data_parallel(self.tconv2, fc1, range(self.ngpu))
            tconv3 = nn.parallel.data_parallel(self.tconv3, tconv2, range(self.ngpu))
            tconv4 = nn.parallel.data_parallel(self.tconv4, tconv3, range(self.ngpu))
            tconv5 = nn.parallel.data_parallel(self.tconv5, tconv4, range(self.ngpu))
            tconv5 = nn.parallel.data_parallel(self.tconv6, tconv5, range(self.ngpu))
            output = tconv5
        else:
            input = input.view(-1, self.nz)
            fc1 = self.fc1(input)
            fc1 = fc1.view(-1, 768, 1, 1)
            tconv2 = self.tconv2(fc1)
            tconv3 = self.tconv3(tconv2)
            tconv4 = self.tconv4(tconv3)
            tconv5 = self.tconv5(tconv4)
            tconv5 = self.tconv6(tconv5)
            output = tconv5
        return output


### if using generator advanced -> transform to either crop 64 x 64 or downsampe
class Generator_Advanced_224_Basic(nn.Module):
    """
    Takes a 100 element input vector and transforms
    it into an image with dimensions 64x64

    input is bacth_size x n_ch x width x height.
    In this case, (batch_size, 100, 1, 1)
    """
    
    # Note on convolutional transpose operations:
    # Input: (batch, chs_in, h_in, w_in)
    # Output: (batch, chs_out, h_out, w_out)
    # h_out = (h_in-1) * stride - 2 * padding + kernel
    
    # Anything with optimizable parameters needs to be
    # setup within the init method
    def __init__(self, ngpu, nz):
        super(Generator_Advanced_224_Basic, self).__init__() 
        self.ngpu = ngpu
        self.nz = nz
        self.main = nn.Sequential(
            
            # Hidden Layer 1
            nn.ConvTranspose2d(in_channels=self.nz, 
                               out_channels=512,
                               kernel_size=4,
                               stride=1,
                               padding=0,
                               bias=False),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            
            # Hidden Layer 2
            nn.ConvTranspose2d(in_channels=512,
                               out_channels=256,
                               kernel_size=4,
                               stride=4,
                               padding=0,
                               bias=False),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            
            # Hidden Layer 3
            nn.ConvTranspose2d(in_channels=256,
                               out_channels=128,
                               kernel_size=4,
                               stride=4,
                               padding=0,
                               bias=False),
                               nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            
            # Hidden Layer 4
            nn.ConvTranspose2d(in_channels=128,
                               out_channels=64,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            
            # Output Layer
            nn.ConvTranspose2d(in_channels=64, 
                               out_channels=1,
                               kernel_size=4,
                               stride=2,
                               padding=17,
                               bias=False),
            nn.Tanh()
        )
        
    # All modules must override `forward`
    def forward(self, x):
        x = self.main(x)
        return x

class Generator_Advanced_224(nn.Module):
    """
    Takes a 100 element input vector and transforms
    it into an image with dimensions 224x224

    input is bacth_size x n_ch x width x height.
    In this case, (batch_size, 1, 224, 224)
    """
    
    # Note on convolutional transpose operations:
    # Input: (batch, chs_in, h_in, w_in)
    # Output: (batch, chs_out, h_out, w_out)
    # h_out = (h_in-1) * stride - 2 * padding + kernel
    
    # Anything with optimizable parameters needs to be
    # setup within the init method
    def __init__(self, ngpu, nz):
        super(Generator_Advanced_224, self).__init__()
        self.ngpu = ngpu
        self.nz = nz
        self.main = nn.Sequential(
            
            # Hidden Layer 1
            nn.ConvTranspose2d(in_channels=self.nz, 
                               out_channels=1024,
                               kernel_size=7,
                               stride=1,
                               padding=0,
                               bias=False),
            #nn.BatchNorm2d(num_features=1024),
            nn.ReLU(),
            
            # Hidden Layer 2
            nn.ConvTranspose2d(in_channels=1024,
                               out_channels=512,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=False),
            #nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            
            # Hidden Layer 3
            nn.ConvTranspose2d(in_channels=512,
                               out_channels=256,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=False),
                               #nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            
            # Hidden Layer 4
            nn.ConvTranspose2d(in_channels=256,
                               out_channels=128,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=False),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            
            # Hidden Layer 5
            nn.ConvTranspose2d(in_channels=128, 
                               out_channels=64,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),

            ## ouutput layer
            nn.ConvTranspose2d(in_channels=64, 
                               out_channels=1,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=False),
            nn.Tanh()
        )
        
    # All modules must override `forward`
    def forward(self, x):
        x = self.main(x)
        return x


