# Model: LSTM (stacked)
# Task: Repeat-Copy Task
# Author: Vineet Jain 
# =============================================================

import numpy as np
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.layers import Dense, Activation, TimeDistributed
from keras.optimizers import Adam, RMSprop


def data_construction(N, length=5, size=8, repeats=5, max_repeats = 20, unary=False, end_marker=True):
	X = []
	Y = []
	for i in range(N):
		sequence = np.random.binomial(1, 0.5, (1, length, size))
		num_repeats_length = repeats if unary else 1
		example_input = np.zeros(((repeats + 1) * length + \
		    num_repeats_length + 1 + end_marker, size + 2))
		example_output = np.zeros(((repeats + 1) * length + \
		    num_repeats_length + 1 + end_marker, size + 2))

		example_input[:length, :size] = sequence
		for j in range(repeats):
		    example_output[(j + 1) * length + num_repeats_length + 1:\
		    (j + 2) * length + num_repeats_length + 1, :size] = sequence
		if unary:
		    example_input[length:length + repeats, -2] = 1
		else:
		    example_input[length, -2] = repeats / float(max_repeats)
		example_input[length + num_repeats_length, -1] = 1
		if end_marker:
		    example_output[-1, -1] = 1

		X.append(example_input)
		Y.append(example_output)

	X = np.asarray(X, dtype=np.float32)
	Y = np.asarray(Y, dtype=np.float32)
	return X, Y

def main():

	length = 5
	size = 8
	repeats = 3
	max_repeats = 10
	unary = False
	end_marker = True

	N_train = 120
	N_test = 40

	learn_rate = 3e-5
	momentum = 0.9
	H1 = 512
	H2 = 512
	H3 = 512

	X_train, Y_train = data_construction(N_train, length=length, size=size, repeats=repeats, max_repeats=max_repeats, unary=unary, end_marker=end_marker)
	X_test, Y_test = data_construction(N_test, length=length, size=size, repeats=repeats, max_repeats=max_repeats, unary=unary, end_marker=end_marker)

	model = Sequential()
	model.add(LSTM(H1, batch_input_shape=(1,X_train.shape[1],X_train.shape[2]), stateful=False, return_sequences=True, activation='sigmoid', recurrent_activation='tanh', use_bias=True))
	model.add(LSTM(H2, stateful=False, return_sequences=True, activation='sigmoid', recurrent_activation='tanh', use_bias=True))
	model.add(LSTM(H3, stateful=False, return_sequences=True, activation='sigmoid', recurrent_activation='tanh', use_bias=True))
	model.add(TimeDistributed(Dense(Y_train.shape[2])))
	model.add(Activation('softmax'))

	#opt = Adam(lr=learn_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
	opt = RMSprop(lr=learn_rate, rho=momentum, epsilon=1e-08, decay=0.0)
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

	model.fit(X_train, Y_train, nb_epoch=1, batch_size=1, verbose=2)

	scores = model.evaluate(X_test, Y_test, batch_size=1, verbose=2)

	print('\nValidation Accuracy: %lf%%' % (scores[1]*100))

	example_input, example_output = data_construction(1, length=length, size=size, repeats=repeats, max_repeats=max_repeats, unary=unary, end_marker=end_marker)

	predicted_output = model.predict(example_input)

	print('\nExample input:')
	print(example_input)
	print('\nExample output:')
	print(example_output)
	print('\nPredicted output:')
	print(predicted_output)

	return


main()
