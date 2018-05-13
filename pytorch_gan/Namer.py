import numpy as np
import sys
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.models import load_model


class Namer(Sequential):
	def __init__(self, output_length=-1,xshape=[128,128],yshape=128,units=256,dropout=0.2):
		super(Namer,self).__init__()
		self.model = Sequential()

		# define the LSTM model
		self.model.add(LSTM(units, input_shape=(xshape[0], xshape[1]), return_sequences=True))
		self.model.add(Dropout(dropout))
		self.model.add(LSTM(units))
		self.model.add(Dropout(dropout))
		self.model.add(Dense(yshape, activation='softmax'))
		self.model.compile(loss='categorical_crossentropy', optimizer='adam')

def make_N(names, resume=False, weightpath=False,n_epochs=50,batchsize=64,maxvals=(0,0)):

	# establish dictionary
	flattext = ''.join(names)
	chars = sorted(list(set(flattext)))
	char_to_int = dict((c, i) for i, c in enumerate(chars))
	int_to_char = dict((i, c) for i, c in enumerate(chars))
	n_chars = len(flattext)
	n_vocab = len(chars)
	maxp, maxt = maxvals
	#print names
	#print len(names)
	#print names[0]

	seq_length = maxp/2
	dataX = []
	dataY = []
	# train on every character aof the title, with
	# the attribute delimeters included
	print maxp+maxt
	count = 0
	for sentence in names:
		count = count + 1
		if count < 20:
			print sentence,": ",len(sentence)
		#print sentence[:],len(sentence)
		#title = sentence.split("_")[-1]
		#st = len(sentence[:-len(title)])
		for i in range(maxp,maxp+maxt-1):
			#print i,
			seq_in = sentence[i-maxp:i]
			seq_out = sentence[i+1]

			dataX.append([char_to_int[char] for char in seq_in])
			dataY.append(char_to_int[seq_out])
	n_patterns = len(dataX)
	#print len(dataX), len(dataY)

	# reshape X to be [samples, time steps, features]
	X = np.reshape(dataX, (n_patterns, maxp, 1))
	# normalize
	X = X / float(n_vocab)
	# one hot encode the output variable
	y = np_utils.to_categorical(dataY)
	print X.shape, y.shape

	N = Namer(xshape=[X.shape[1],X.shape[2]],yshape=y.shape[1])

	# establish training checkpoints (only save best)
	if weightpath:
		filepath = weightpath
	else:
		filepath = "/home/sloane/ArtBot/pytorch_gan/namenet/name_weights.hdf5"
	if resume:
		N.model = load_model(filepath)

	checkpoint = ModelCheckpoint(filepath, monitor='loss', save_best_only=True, mode='min')
	callbacks_list = [checkpoint]

	N.model.fit(X, y, epochs = n_epochs, batch_size = batchsize, callbacks=callbacks_list)

	return N,(int_to_char,dataX,n_vocab)

def predict_word(N,seed,tools,maxp):
	int_to_char,dataX,n_vocab = tools

	for i in range(maxp):
		x = np.reshape(seed, (1, len(seed), 1))
		x = x / float(n_vocab)
		prediction = N.model.predict(x, verbose=0)
		index = np.argmax(prediction)
		result = int_to_char[index]
		seq_in = [int_to_char[value] for value in seed]
		#sys.stdout.write(result)
		seed.append(index)
		seed = seed[1:len(seed)]