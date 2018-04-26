import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicDiscriminator(nn.Module):
    def __init__(self, output_size, use_bias = True, input_channels = 1, dual_output = False, class_output = False):
        super(BasicDiscriminator, self).__init__()
        self.dual_output = dual_output
        self.class_output = class_output

        self.input_channels = input_channels

        if dual_output:
            self.class_input = nn.Sequential(
                nn.ConvTranspose2d(10, 1, 64, stride = 1, padding = 0, bias = use_bias),
                nn.BatchNorm2d(1)
            )

            self.input_channels = 2
            

        self.features = nn.Sequential(
            # 64
            nn.Conv2d(self.input_channels, 128, 4, stride = 2, padding = 1, bias = use_bias),
            nn.LeakyReLU(.2, True),

            # 32
            nn.Conv2d(128, 256, 4, stride = 2, padding = 1, bias = use_bias),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(.2, True),

            # 16
            nn.Conv2d(256, 512, 4, stride = 2, padding = 1, bias = use_bias),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(.2, True),

            nn.Conv2d(512, 1024, 4, stride = 2, padding = 1, bias = use_bias),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(.2, True),  

            nn.Conv2d(1024, output_size, 4, stride = 1, padding = 0, bias = use_bias),
        )

        if class_output:
            self.final = nn.Softmax(dim = 1)
        else:
            self.final = nn.Sigmoid()

        # self.binary_out = nn.Sequential(
        #     nn.Conv2d(512, 1024, 4, stride = 2, padding = 1, bias = use_bias),
        #     nn.BatchNorm2d(1024),
        #     nn.LeakyReLU(.2, True),  

        #     nn.Conv2d(1024, 1, 4, stride = 1, padding = 0, bias = use_bias),
        #     nn.Sigmoid()
        # )

        # self.multi_out = nn.Sequential(
        #     nn.Conv2d(512, 1024, 4, stride = 2, padding = 1, bias = use_bias),
        #     nn.BatchNorm2d(1024),
        #     nn.LeakyReLU(.2, True),  

        #     nn.Conv2d(1024, 10, 4, stride = 1, padding = 0, bias = use_bias),
        #     nn.Softmax(dim = 1)
        # )

    def forward(self, inp, class_label = None):
        if self.dual_output:
            c = self.class_input(class_label)
            inp = torch.cat((inp, c), 1)
        inp = self.features(inp)
        inp = self.final(inp)
        return inp

        # if self.dual_output:
        #     bin_out = self.binary_out(inp)
        #     class_out = self.multi_out(inp)
        #     return (bin_out, class_out)
        # else:
        #     bin_out = self.binary_out(inp)
        #     return bin_out