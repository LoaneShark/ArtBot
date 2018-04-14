import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicDiscriminator(nn.Module):
    def __init__(self, use_bias = True, dual_output = False):
        super(BasicDiscriminator, self).__init__()
        self.dual_output = dual_output

        self.features = nn.Sequential(
            nn.Conv2d(1, 128, 4, stride = 2, padding = 1, bias = use_bias),
            nn.LeakyReLU(.2, True),

            nn.Conv2d(128, 256, 4, stride = 2, padding = 1, bias = use_bias),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(.2, True),

            nn.Conv2d(256, 512, 4, stride = 2, padding = 1, bias = use_bias),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(.2, True),
            
            nn.Conv2d(512, 1024, 4, stride = 2, padding = 1, bias = use_bias),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(.2, True),            
        )

        self.binary_out = nn.Sequential(
            nn.Conv2d(1024, 1, 4, stride = 1, padding = 0, bias = use_bias),
            nn.Sigmoid()
        )

        self.multi_out = nn.Sequential(
            nn.Conv2d(1024, 10, 4, stride = 1, padding = 0, bias = use_bias),
            nn.Softmax(dim = 1)
        )

    def forward(self, inp):
        inp = self.features(inp)
        # print inp
        # quit()
        if self.dual_output:
            bin_out = self.binary_out(inp)
            class_out = self.multi_out(inp)
            return (bin_out, class_out)
        else:
            bin_out = self.binary_out(inp)
            return bin_out