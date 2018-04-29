import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicGenerator(nn.Module):
    def __init__(self, z_dim = 20, use_bias = True, dual_input = False, output_channels = 1):
        super(BasicGenerator, self).__init__()
        self.input_dim = z_dim
        self.dual_input = dual_input

        if dual_input:
            self.start_binary = nn.Sequential(
                nn.Linear(z_dim, 200),
                nn.LeakyReLU(.2, True)
            )

            self.start_class = nn.Sequential(
                nn.Linear(11, 200),
                nn.LeakyReLU(.2, True)
            )

            self.input_combine = nn.Sequential(
                nn.Linear(400, 512),
                nn.LeakyReLU(.2, True)
            )

            self.input_dim = 512

        self.generator = nn.Sequential(
            nn.ConvTranspose2d(self.input_dim, 1024, 4, stride = 1, padding = 0, bias = use_bias),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(.2, True),

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
            nn.ConvTranspose2d(128, output_channels, 4, stride = 2, padding = 1, bias = use_bias),
            nn.Tanh()
            # 32 x 32 x 1
        )

    def forward(self, noise, class_label = None):
        if self.dual_input:            
            hidden_bin = self.start_binary(noise.squeeze())
            hidden_class = self.start_class(class_label.squeeze())
            inp = self.input_combine(torch.cat((hidden_bin, hidden_class), 1))
            inp = inp.view(inp.size(0), inp.size(1), 1, 1)
        else:
            inp = noise
        out = self.generator(inp)
        return out