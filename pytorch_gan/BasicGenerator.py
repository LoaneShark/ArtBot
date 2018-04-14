import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicGenerator(nn.Module):
    def __init__(self, z_dim = 20, use_bias = True):
        super(BasicGenerator, self).__init__()
        self.z_dim = z_dim

        self.generator = nn.Sequential(
            # 100 x 1
            nn.ConvTranspose2d(z_dim, 1024, 4, stride = 1, padding = 0, bias = use_bias),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(.2, True),
            # 4 x 4 x 512
            nn.ConvTranspose2d(1024, 512, 4, stride = 2, padding = 1, bias = use_bias),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(.2, True),
            # 8 x 8 x 256
            nn.ConvTranspose2d(512, 256, 4, stride = 2, padding = 1, bias = use_bias),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(.2, True),
            # 16 x 16 x 128
            nn.ConvTranspose2d(256, 128, 4, stride = 2, padding = 1, bias = use_bias),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(.2, True),
            # 16 x 16 x 128
            nn.ConvTranspose2d(128, 1, 4, stride = 2, padding = 1, bias = use_bias),
            nn.Tanh()
            # 32 x 32 x 1
        )

    def forward(self, inp):
        inp = self.generator(inp)
        return inp