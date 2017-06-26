import numpy as np

def construct_batch(N, length=5, size=8, repeats=5, max_repeats=20, end_marker=True):
	X = []
	Y = []
	for i in range(N):
		sequence = np.random.binomial(1, 0.5, (1, length, size))
		num_repeats_length = 1
		example_input = np.zeros(((repeats + 1) * length + \
		    num_repeats_length + 1 + end_marker, size + 2))
		example_output = np.zeros(((repeats + 1) * length + \
		    num_repeats_length + 1 + end_marker, size + 2))

		example_input[:length, :size] = sequence
		for j in range(repeats):
		    example_output[(j + 1) * length + num_repeats_length + 1:\
		    (j + 2) * length + num_repeats_length + 1, :size] = sequence
		example_input[length, -2] = repeats / float(max_repeats)
		example_input[length + num_repeats_length, -1] = 1
		if end_marker:
		    example_output[-1, -1] = 1

		X.append(example_input)
		Y.append(example_output)

	X = np.asarray(X, dtype=np.float32)
	Y = np.asarray(Y, dtype=np.float32)
	
	return X, Y

def data_construction(max_iters, batch_size=1, min_length=1, max_length=20, min_repeats=2, max_repeats=5, size=8, end_marker=True):
	X = []
	Y = []
	for i in range(max_iters):
		length = int(np.random.random(1)*(max_length - min_length)) + min_length
		repeats = int(np.random.random(1)*(max_repeats - min_repeats)) + min_repeats
		x, y = construct_batch(batch_size, length=length, size=size, repeats=repeats, max_repeats=max_repeats, end_marker=end_marker)
		X.append(x)
		Y.append(y)

	return X, Y