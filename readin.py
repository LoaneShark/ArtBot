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
# 		- readin crashes around line 930 of metadata file

import numpy as np 
import scipy

from PIL import Image
#import pygame
import os, os.path, sys, io
#import zipfile


# establish path to wherever the kaggle database is
# stored locally (uncompressed)
kagglepath = "../kaggle/"

def main():

	# open an image, for example
	test = kagglepath + "train_1/1.jpg"
	img = Image.open(test)
	#img.show()

	N = 0
	# go through ALL train set folders
	for i in range(1,9):
		subpath = kagglepath + "train_" + str(i) + "/"
		Ni = len([name for name in os.listdir(subpath) if os.path.isfile(name)])
		N += Ni
	print("N: " + str(N))
	
	# parse through and store all metadata in a dictionary 
	M = {} 	# metadata (by author)
	F = {}	# metadata by filename
	metapath = kagglepath + "train_info.csv"
	metafile = open(metapath, "r")

	n = 0
	# store all metadata attribute vectors in M and F dicts
	## TODO: Fix unreadable character issue (@n = 930)
	for line in metafile:
		n += 1
		if n <= 79433:
			print(n)
			print(line)
			linedats = line.split(",")
			M[str(linedats[1])] = [linedats[0][3:]] + linedats[2:-1] + [linedats[-1][:-1]]
			F[str(linedats[0][3:])] = linedats[1:-1] + [linedats[-1][:-1]]







if __name__ == "__main__":
	main()