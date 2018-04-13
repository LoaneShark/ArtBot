
# Neural net trained on metadata
# mostly for funsies

import numpy as np
import scipy
import sys
from textgenrnn import textgenrnn

#sys.setdefaultencoding('utf8')

def train(metadata, params, save=False):
	[author, style, genre, date] = params
	namegen = textgenrnn()

	# filter by author
	if author == -1:
		data = [att for sublist in metadata.values() for att in sublist]
	else:
		data = metadata[str(author)]

	

	# we don't care about filenames
	data = [att[1:] for att in data]
	# filter by date
	if date == -1:
		data = [att[:-1] for att in data]
	else:
		data = [att[:-1] for att in data if att[3] == str(date)]
	
	# filter by genre
	if genre == -1:
		data = [att[:-1] for att in data]
	else:
		data = [att[:-1] for att in data if att[2] == str(genre)]
	
	# filter by style
	if style == -1:
		data = [att[0] for att in data]
	else:
		data = [att[0] for att in data if att[1] == str(style)]
	
	# train neural net
	namegen.train_on_texts(data, verbose=0)
	print("Generating titles...\n")
	if save:
		if author != -1:
			ext += str(author) + "_"
		if style != -1:
			ext += str(style) + "_"
		if genre != -1:
			ext += str(genre) + "_"
		if date != -1:
			ext += str(date) + "_"
		namegen.save(filename)
	namegen.generate(4,temperature=0.420)