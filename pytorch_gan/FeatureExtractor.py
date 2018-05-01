import torch
import torch.nn as nn
from BasicDiscriminator import BasicDiscriminator
from BasicGenerator import BasicGenerator
import torchvision
from torchvision import utils
import shutil
from support import *
import numpy as np
import argparse
import textgenrnn

parser = argparse.ArgumentParser()
parser.add_argument('-b', '--batch_size', type = int, default = 64)
parser.add_argument('-n', '--noise_dim', type = int, default = 100)
parser.add_argument('-e', '--epochs', type = int, default = 5000)
parser.add_argument('-r', '--resume', type = str, default = None)
parser.add_argument('-v', '--verbosity', type = int, default = 1)
parser.add_argument('-d', '--dual_discrim', action = 'store_true')
parser.add_argument('-q', '--quick_load', action = 'store_true')

args = parser.parse_args()

def main():
	test = BasicDiscriminator(1,use_bias=True)
	test.apply(weights_init)

	print(len(test.features))

if __name__ == "__main__":
	main()