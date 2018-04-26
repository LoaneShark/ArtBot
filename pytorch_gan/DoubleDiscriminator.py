import torch
import torch.nn as nn
import torch.nn.functional as F
from BasicDiscriminator import BasicDiscriminator

class DoubleDiscriminator(nn.Module):
    def __init__(self, use_bias = True):
        super(DoubleDiscriminator, self).__init__()
        
        self.D1 = BasicDiscriminator(1, use_bias=use_bias)
        self.D2 = BasicDiscriminator(10,use_bias=use_bias)        

    def forward(self, inp):
        return (self.D1(inp), self.D2(inp))