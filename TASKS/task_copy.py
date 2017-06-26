import numpy as np

def construct_batch(N, length=5, size=8, end_marker=True):
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

def data_construction(max_iters, batch_size=1, min_length=1, max_length=20, size=8, end_marker=True):
	X = []
	Y = []
	for i in range(max_iters):
		length = int(np.random.random(1)*(max_length - min_length)) + min_length
		x, y = construct_batch(batch_size, length=length, size=size, end_marker=end_marker)
		X.append(x)
		Y.append(y)

	return X, Y