import numpy as np
#import sys
import keras
from Namer import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('seed_file',type=str,default = '/public/bbeyers/CSC249_Project/pytorch_gan/meta/seeds.txt')
parser.add_argument('output_file',type=str,default='/public/bbeyers/CSC249_Project/pytorch_gan/meta/fake_titles.txt')
parser.add_argument('-e','--epochs',type=int,default = 50)
parser.add_argument('-r','--resume',action='store_true')
parser.add_argument('-v','--verbosity',type=int,default=0)
args = parser.parse_args()
if args.verbosity >= 2:
	print "Loading Metadata..."
# load in metadata
with open("/home/sloane/ArtBot/pytorch_gan/meta/prefixes.txt") as prefile:
	prefixes = prefile.readlines()
	#prefixes = pretext.split("\n")
	prefixes = prefixes[1:]

with open("/home/sloane/ArtBot/pytorch_gan/meta/titles.txt") as titlefile:
	titles = titlefile.readlines()
	#titles = titletext.split("\n")
	titles = titles[1:]

if args.verbosity >= 1:
	print str(len(titles))+" prefixes and titles loaded in"

# find max prefix and title lengths
maxp = max([len(prefix) for prefix in prefixes])
maxt = max([len(title) for title in titles])
if args.verbosity >= 1:
	print "maxp: " + str(maxp) + "   maxt: " + str(maxt)

# pad data
data = []
for pre,title in zip(prefixes,titles):
	pstr = ""
	tstr = ""
	for i in range(len(pre),maxp):
		pstr = pstr + " "
	for i in range(len(title),maxt):
		tstr = tstr + "="
	pre = pre + pstr + "|"
	title = title + tstr
	data.append(pre+title)
	#print pre+title+"\n"
#data = [pre+title for pre,title in zip(prefixes,titles)]
#print data

#N = Namer()
if args.resume:
	print "Resuming training from stored weights..."
N,tools = make_N(names=data,n_epochs=args.epochs,resume=args.resume,maxvals=(maxp,maxt))


# load in the feature vector seeds
with open(args.seed_file, "r+") as seedfile:
	seeds = seedfile.split("\n")
	seeds = seeds[1:]

n_titles = len(seeds)
if args.verbosity >= 1:
	print "Generating %d titles from seed file" % n_titles
# generate titles
for seed in seeds:
	seed = predict_word(N,seed,tools,maxp)

with open(args.output_file,"w+") as outfile:
	for seed in seeds:
		outfile.write("\n"+seed)