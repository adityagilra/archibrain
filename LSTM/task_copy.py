# Model: LSTM (stacked)
# Task: Copy Task
# Author: Vineet Jain 
# =============================================================

import numpy as np
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.layers import Dense, Activation, TimeDistributed
from keras.optimizers import Adam, RMSprop


def data_construction(N, length=5, size=8, end_marker=True):
	X = []
	Y = []
	for i in range(N):
		sequence = np.random.binomial(1, 0.5, (1, length, size))
		example_input = np.zeros((2 * length + 1 + end_marker, \
		            size + 1))
		example_output = np.zeros((2 * length + 1 + end_marker, \
		            size + 1))

		example_input[:length, :size] = sequence
		example_input[length, -1] = 1
		example_output[length+1:2*length+1, :size] = sequence
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
	end_marker = True

	N_train = 1000
	N_test = 400

	learn_rate = 3e-5
	momentum = 0.9
	H1 = 256
	H2 = 256
	H3 = 256

	X_train, Y_train = data_construction(N_train, length=length, size=size, end_marker=end_marker)
	X_test, Y_test = data_construction(N_test, length=length, size=size, end_marker=end_marker)

	model = Sequential()
	model.add(LSTM(H1, batch_input_shape=(1,2*(length+1),size+1), stateful=False, return_sequences=True, activation='sigmoid', recurrent_activation='tanh', use_bias=False))
	model.add(LSTM(H2, stateful=False, return_sequences=True, activation='sigmoid', recurrent_activation='tanh', use_bias=False))
	model.add(LSTM(H3, stateful=False, return_sequences=True, activation='sigmoid', recurrent_activation='tanh', use_bias=False))
	model.add(TimeDistributed(Dense(size+1)))
	model.add(Activation('linear'))

	#opt = Adam(lr=learn_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
	opt = RMSprop(lr=learn_rate, rho=momentum, epsilon=1e-08, decay=0.0)
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

	model.fit(X_train, Y_train, epochs=1, batch_size=1, verbose=1)

	scores = model.evaluate(X_test, Y_test, batch_size=1, verbose=1)

	print('\nValidation Accuracy: %lf%%' % (scores[1]*100))

	example_input, example_output = data_construction(1, length=length, size=size, end_marker=end_marker)

	predicted_output = model.predict(example_input)

	print('\nExample input:')
	print(example_input)
	print('\nExample output:')
	print(example_output)
	print('\nPredicted output:')
	print(predicted_output)

	return


main()
