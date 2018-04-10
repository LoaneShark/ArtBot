####################################
# Santiago Loane & Brad Beyers
# CSC 249
# 
# v0.0		4/4/18
#-----------------------------------
#
# Required packages: numpy, scipy
#
####################################

# Reads in the paintings

# TODO: - everything

import numpy as np 
import scipy

from PIL import Image
#import pygame
import os, sys, io
#import zipfile


# establish path to wherever the kaggle database is
# stored locally (uncompressed)
kagglepath = "../kaggle/"

def main():

	# open an image, for example
	test = kagglepath + "train_1/1.jpg"
	img = Image.open(test)
	img.show()



if __name__ == "__main__":
	main()