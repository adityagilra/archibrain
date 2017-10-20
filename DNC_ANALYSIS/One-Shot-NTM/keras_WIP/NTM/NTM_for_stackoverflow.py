from keras.models import Model,Sequential
from keras.layers import Input, Dense, LSTM, Lambda, Activation
from keras.layers import concatenate, Reshape, Flatten, Add, Multiply, Dot, add, multiply, dot
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import RMSprop, Adam
from keras.activations import softmax
from keras import backend as K
from keras.utils import plot_model

import numpy as np


def data_construction(N, length=5, size=8, end_marker=True):
	
	X = []
	Y = []

	previous_output = np.zeros((2 * length + 1 + end_marker, size + 1))

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

		x = np.concatenate([example_input, previous_output],axis=1)

		X.append(x)
		Y.append(example_output)
		previous_output = example_output


	X = np.asarray(X, dtype=np.float32)
	Y = np.asarray(Y, dtype=np.float32)
	return X, Y

N = 100

X_train, Y_train = data_construction(N, length=6, size=5, end_marker=True)		
np.set_printoptions(precision=3)

model = 'D'

## A
if model=='A':
	controller_input = Input(shape=(14,12),name='New_Input')

	MEMORY = Lambda(lambda x: K.zeros(shape=(1,120,40)),name='Memory_0')(controller_input)
	usage_weights = Lambda(lambda x: K.zeros(shape=(1,1,120)),name='Usage_Weights_0')(controller_input)
	read_weights = Lambda(lambda x: K.zeros(shape=(1,14,120)),name='Read_Weights_0')(controller_input)

	controller = LSTM(units=200, activation='tanh',stateful=False, return_sequences=True,name='LSTM_CONTROLLER')(controller_input)
	write_keys = Dense(40, activation='tanh',name='Write_Keys')(controller)
	read_keys = Dense(40, activation='tanh',name='Read_Keys')(controller)
	omegas = Dense(1, activation='sigmoid',name='Omegas')(controller)
	least_usage = Lambda(lambda x: K.one_hot(indices=K.argmax(-x),num_classes=120),name='Least_Usage')(usage_weights)
	omegas_tiled = Lambda(lambda x: K.tile(x,(1,1,120)))(omegas)
	compl_omegas = Lambda(lambda o:  K.ones(shape=(14,120)) - o)(omegas_tiled)
	rd_part = Multiply()([omegas_tiled, read_weights])
	us_part = Multiply()([compl_omegas, least_usage])
	write_weights = Add(name='Write_Weights')([rd_part,us_part])
	writing = Dot(axes=[1,1])([write_weights, write_keys])
	MEMORY = Add(name='Memory')([MEMORY, writing])
	cos_sim = Dot(axes=[2,2], normalize=True,name='Cosine_Similarity')([read_keys,MEMORY])  
	read_weights = Lambda(lambda x: softmax(x,axis=1),name='Read_Weights')(cos_sim)
	write_weights_summed = Lambda(lambda x: K.sum(x,axis=1,keepdims=True))(write_weights)
	read_weights_summed = Lambda(lambda x: K.sum(x,axis=1,keepdims=True))(read_weights)
	decay_usage = Lambda(lambda x: K.constant(0.99, shape=(1,120))*x)(usage_weights)
	usage_weights = Add(name='Usage_Weights')([decay_usage, read_weights_summed, write_weights_summed])
	retrieved_memory = Dot(axes=[2,1],name='Retrieved_Memories')([read_weights, MEMORY])
	controller_output = concatenate([controller, retrieved_memory],name='Controller_Output')
	main_output = Dense(6,activation='sigmoid',name='Final_Output')(controller_output)

	M = Model(inputs=controller_input, outputs=[main_output])	
	M.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=3e-5,rho=0.9,decay=0), metrics=['accuracy'])
	plot_model(M, to_file='NTM_model_A.png',show_shapes=False,show_layer_names=True)
	M.fit(X_train, Y_train, batch_size=1)

## B
if model=='B':
	controller_input = Input(shape=(14,12))
	MEMORY = Input(shape=(120,40))
	prev_usage_weights = Input(shape=(1,120))
	prev_read_weights = Input(shape=(14,120))

	controller = LSTM(units=200, activation='tanh',stateful=False,return_sequences=True)(controller_input)
	write_keys = Dense(40, activation='tanh')(controller)
	read_keys = Dense(40, activation='tanh')(controller)
	omegas = Dense(1, activation='sigmoid')(controller)
	least_usage = Lambda(lambda x: K.one_hot(indices=K.argmax(-x),num_classes=120))(prev_usage_weights)
	omegas_tiled = Lambda(lambda x: K.tile(x,(1,1,120)))(omegas)
	compl_omegas = Lambda(lambda o:  K.ones(shape=(14,120)) - o)(omegas_tiled)
	rd_part = Multiply()([omegas_tiled, prev_read_weights])
	us_part = Multiply()([compl_omegas, least_usage])
	write_weights = Add()([rd_part,us_part])
	writing = Dot(axes=[1,1])([write_weights, write_keys])
	NEW_MEMORY = Add()([MEMORY, writing])
	cos_sim = Dot(axes=[2,2], normalize=True)([read_keys,NEW_MEMORY])  
	read_weights = Lambda(lambda x: softmax(x,axis=1))(cos_sim)
	write_weights_summed = Lambda(lambda x: K.sum(x,axis=1,keepdims=True))(write_weights)
	read_weights_summed = Lambda(lambda x: K.sum(x,axis=1,keepdims=True))(read_weights)
	decay_usage = Lambda(lambda x: K.constant(0.99, shape=(1,120))*x)(prev_usage_weights)
	usage_weights = Add()([decay_usage, read_weights_summed, write_weights_summed])
	retrieved_memory = Dot(axes=[2,1])([read_weights, NEW_MEMORY])
	controller_output = concatenate([controller, retrieved_memory])
	main_output = Dense(6,activation='sigmoid')(controller_output)

	M1 = Model(inputs=[controller_input,MEMORY,prev_usage_weights,prev_read_weights],outputs=[main_output,NEW_MEMORY,usage_weights,read_weights])
	M1.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=3e-5,rho=0.9,decay=0), metrics=['accuracy'])

	M2 = Model(inputs=[controller_input,MEMORY,prev_usage_weights,prev_read_weights], outputs=[main_output])	
	M2.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=3e-5,rho=0.9,decay=0), metrics=['accuracy'])

	mem = np.zeros(shape=(1,120,40))
	w_r =  np.zeros(shape=(1,14,120))
	w_u = np.zeros(shape=(1,1,120))
	
	#M2.fit([X_train,mem,w_u,w_r],[Y_train])

	for i in np.arange(200):
		s = X_train[i:i+1,:,:]
		y = Y_train[i:i+1,:,:]
		o,new_mem,new_w_u,new_w_r = M1.predict([s,mem,w_u,w_r])
		M2.fit([s,mem,w_u,w_r],[y])
		mem=new_mem
		w_u=new_w_u
		w_r=new_w_r
		W = M2.get_weights()
		M1.set_weights(W)

if model=='C':

	model = Sequential()
	model.add(LSTM(256, input_shape=(14, 12), stateful=False, return_sequences=True, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True))
	model.add(LSTM(256, stateful=False, return_sequences=True, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True))
	model.add(LSTM(256, stateful=False, return_sequences=True, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True))
	model.add((Dense(6)))
	model.add(Activation('sigmoid'))

	opt = RMSprop(lr=3e-5, rho=0.9, epsilon=1e-08, decay=0.0)
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

	for i in np.arange(N):
		model.fit(X_train[i:i+1,:,:],Y_train[i:i+1,:,:])


if model=='D':

	MEMORY = K.variable(np.zeros((1,120,40)) )
	usage_weights = K.variable(np.zeros((1,1,120)) )
	read_weights = K.variable(np.zeros((1,14,120)) )

	controller_input = Input(shape=(14,12),name='New_Input')

	controller = LSTM(units=200, activation='tanh',stateful=False, return_sequences=True, name='LSTM_CONTROLLER')(controller_input)
	write_keys = Dense(40, activation='tanh',name='Write_Keys')(controller)
	read_keys = Dense(40, activation='tanh',name='Read_Keys')(controller)
	omegas = Dense(1, activation='sigmoid',name='Omegas')(controller)
	least_usage = Lambda(lambda x: K.one_hot(indices=K.argmax(-x),num_classes=120),name='Least_Usage')(usage_weights)
	omegas_tiled = Lambda(lambda x: K.tile(x,(1,1,120)))(omegas)
	compl_omegas = Lambda(lambda o:  K.ones(shape=(14,120)) - o)(omegas_tiled)
	rd_part = multiply([omegas_tiled, read_weights])
	us_part = multiply([compl_omegas, least_usage])
	write_weights = add([rd_part,us_part])
	writing = dot([write_weights, write_keys],axes=[1,1])
	MEMORY = add([MEMORY, writing])
	cos_sim = dot([read_keys,MEMORY],axes=[2,2], normalize=True)  
	read_weights = Lambda(lambda x: softmax(x,axis=1),name='Read_Weights')(cos_sim)
	write_weights_summed = Lambda(lambda x: K.sum(x,axis=1,keepdims=True))(write_weights)
	read_weights_summed = Lambda(lambda x: K.sum(x,axis=1,keepdims=True))(read_weights)
	decay_usage = Lambda(lambda x: K.constant(0.99, shape=(1,120))*x)(usage_weights)
	usage_weights = add([decay_usage, read_weights_summed, write_weights_summed])
	retrieved_memory = dot([read_weights, MEMORY],axes=[2,1])
	controller_output = concatenate([controller, retrieved_memory],name='Controller_Output')
	main_output = Dense(6,activation='sigmoid',name='Final_Output')(controller_output)

	M = Model(inputs=controller_input, outputs=main_output)	
	M.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=3e-5,rho=0.9,decay=0), metrics=['accuracy'])
	plot_model(M, to_file='NTM_model_D.png',show_shapes=False,show_layer_names=True)
	#M.fit(X_train, Y_train, batch_size=1)
