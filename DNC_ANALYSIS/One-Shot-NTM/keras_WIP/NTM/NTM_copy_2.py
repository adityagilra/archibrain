## CLASS FOR NTM MODEL in the implementation of One-shot Learning
##
## AUTHOR: Marco Martinolli
## DATE: 09.05.2017


import numpy as np
import activations as act

from keras.models import Model
from keras.layers import Input, Dense, LSTM, Lambda, Activation
from keras.layers import concatenate, Reshape, Flatten, Add, Multiply, Dot
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import RMSprop, Adam
from keras.activations import softmax
from keras import backend as K

import matplotlib 
matplotlib.use('GTK3Cairo') 
from matplotlib import pyplot as plt
from matplotlib import gridspec
import pylab
import gc 



class NTM():

	def __init__(self, S, O, H=200, N=128, M=40, num_heads=4, lr=0.0001, decay=0.95, momentum=0.9, gamma=0.99, minibatch=16, optim='Adam', depth_in_time=False):

		self.S = S
		self.H = H
		self.O = O

		self.N = N
		self.M = M

		if depth_in_time==False:
			self.n = num_heads
		else:
			self.n = 1

		self.minibatch = minibatch

		self.GAMMA = K.constant(gamma, shape=(1,self.N))

		#self.MEMORY = K.placeholder(shape=(self.minibatch,self.N,self.M))
		#self.usage_weights = K.placeholder(shape=(self.minibatch,1,self.N))
		#self.read_weights = K.placeholder(shape=(self.minibatch,self.S[0],self.N))
		#self.write_weights = K.placeholder(shape=(self.minibatch,self.S[0],self.N))

		controller_input = Input(shape=(self.S[0],self.S[1]))
	
		self.MEMORY = Lambda(lambda x: K.zeros(shape=(self.minibatch,self.N,self.M)))(controller_input)
		self.usage_weights = Lambda(lambda x: K.zeros(shape=(self.minibatch,1,self.N)))(controller_input)
		self.read_weights = Lambda(lambda x: K.zeros(shape=(self.minibatch,self.S[0],self.N)))(controller_input)
		self.write_weights = Lambda(lambda x: K.zeros(shape=(self.minibatch,self.S[0],self.N)))(controller_input)
		#print(self.MEMORY) 
		print('CHECK: ',K.is_keras_tensor(self.MEMORY))

		# LSTM controller		
		#controller = LSTM(units=self.H, activation='tanh',stateful=False, return_sequences=True)(controller_input)	
		controller1 = LSTM(self.H, stateful=False, return_sequences=True, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True)(controller_input)
		controller2 = LSTM(self.H, stateful=False, return_sequences=True, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True)(controller1)
		controller3 = LSTM(self.H, stateful=False, return_sequences=True, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True)(controller2)

		# key outputs for memory dynamics
		read_keys = Dense(self.M, activation='tanh')(controller3) 	# 1 x n*M
		print('READ KEYS: ', read_keys)				
		write_keys = Dense(self.M, activation='tanh')(controller3) 	# 1 x n*M
		print('WRITE KEYS: ', write_keys)			
		omegas = Dense(1, activation='sigmoid')(controller3) 		# 1 x n
		print('OMEGAS: ', omegas)	

		least_usage = Lambda(lambda x: K.one_hot(indices=K.argmax(-x),num_classes=self.N))(self.usage_weights)	# -1

		## writing to memory
		omegas_tiled = Lambda(lambda x: K.tile(x,(1,1,self.N)))(omegas)
		print('omegas tiled: ', omegas_tiled)
		print('prev read: ', self.read_weights)
		rd_part = Multiply()([omegas_tiled, self.read_weights])
		print('read part: ', rd_part)
		compl_omegas = Lambda(lambda o:  K.ones(shape=(self.S[0],self.N)) - o)(omegas_tiled)
		print('complementary omegas: ', compl_omegas)
		print('least usage: ', least_usage)
		us_part = Multiply()([compl_omegas, least_usage])
		print('usage part: ', us_part)			
		self.write_weights = Add()([rd_part,us_part])
		print('WRITE WEIGHTS: ', self.write_weights)

		print('MEMORY: ', self.MEMORY)
		writing = Dot(axes=[1,1])([self.write_weights, write_keys])
		print('writing: ', writing)
		print('shape? ',K.get_variable_shape(writing))
		self.MEMORY = Add()([self.MEMORY, writing])
		print('NEW MEMORY: ', self.MEMORY)
 
		### reading from memory
		print('READ KEYS: ',read_keys)
		cos_sim = Dot(axes=[2,2], normalize=True)([read_keys,self.MEMORY])  
		print('COSINE_SIMILARITY: ',cos_sim)
		self.read_weights = Lambda(lambda x: softmax(x,axis=1))(cos_sim) 
		print('READ WEIGHTS: ',self.read_weights)

		write_weights_summed = Lambda(lambda x: K.sum(x,axis=1,keepdims=True))(self.write_weights)
		print(write_weights_summed)
		read_weights_summed = Lambda(lambda x: K.sum(x,axis=1,keepdims=True))(self.read_weights)
		print(read_weights_summed)
		print(self.usage_weights)
		decay_usage = Lambda(lambda x: self.GAMMA*x)(self.usage_weights)
		usage_weights = Add()([decay_usage, read_weights_summed, write_weights_summed])
		print('USAGE WEIGHTS: ', self.usage_weights)

		retrieved_memory = Dot(axes=[2,1])([self.read_weights, self.MEMORY])
		print('RETRIEVED MEMORY: ',retrieved_memory)

		### concatenation
		controller_output = concatenate([controller3, retrieved_memory])
		print('CONCATENATED OUTPUT: ',controller_output)
		# classifier
		main_output = Dense(self.O[1],activation='sigmoid')(controller_output)
		#main_output = TimeDistributed(Dense(self.O[1],activation='sigmoid'))(controller_output)
		print('FINAL OUTPUT: ',main_output)

		loss_fct='binary_crossentropy' 

		if optim == 'RMSprop':
			opt = RMSprop(lr=lr,rho=momentum,decay=decay)
		elif optim == 'Adam':
			opt = Adam(lr=lr,decay=decay)


		# NTM model: stimulus->controller->concatenation->output

		self.NTM = Model(inputs=controller_input, outputs=[main_output])	
		self.NTM.compile(loss=loss_fct, optimizer=opt, metrics=['accuracy'])


		# model just used for the ckeck		
		
		#self.NTM_total = Model(inputs=[controller_input,MEMORY,prev_read,prev_usage,least_usage], outputs=[main_output, NEW_MEMORY,controller, retrieved_memory,controller_output,read_keys,cos_sim, read_weights,usage_weights,write_keys,write_weights])
		#self.NTM_total.compile(loss=loss_fct, optimizer=opt, metrics=['accuracy'])	

	def build_least_usage(self,w_u,k):

		w_lu = np.zeros(np.shape(w_u))
		w_u_sorted = np.argsort(w_u,axis=1)
		min_ind = w_u_sorted[0,:k,0]		
		w_lu[0,min_ind, 0] = 1

		return w_lu

	def compute_response(self,p,gate):
		
		if gate=='max':

			resp_ind = np.argmax(p)

			return resp_ind

		if gate=='prob':

			pr = np.squeeze(p)
			resp_ind = np.random.choice(a=len(pr), size=1, p=pr)
			
			return resp_ind


#####################################################################################################################################################################################
# TRAINING/TEST FOR COPY-COPYREPEAT TASKS			

	def training(self,S_train,Y_train,dic_label=None,verbose=0):

		self.NTM.fit(S_train,Y_train,batch_size=self.minibatch)

	def test(self,S_test,Y_test, dic_label=None, verbose=0):

		scores = self.NTM.evaluate(S_test,Y_test)
		
		return scores[0], scores[1]

###########################################################################################################################################################################
# DEBUGGING FUNCTIONS

	def check(self,S,Y,dic_label):
	
		N = np.shape(S)[0]

		self.MEMORY = np.zeros((1,self.N,self.M))
		if self.depth_in_time==False:
			self.read_weights = np.zeros((1,self.N, self.n))
		else:
			self.read_weights = np.zeros((1,self.S[0],self.N))		
		self.usage_weights = np.zeros((1,1,self.N))

		for i in np.arange(N):

			if isinstance(self.S,int):
				s = S[i:(i+1),:]
				s = np.reshape(s,(1,1,self.S) )
			else:
				s = S[i:(i+1),:,:]
				s = np.reshape(s,(1,self.S[0],self.S[1]) )

			if isinstance(self.O,int):
				y = Y[i:(i+1),:]
				if dic_label is not None:
					y_print = self.dic_label[np.argmax(y)]
			else:
				y = Y[i:(i+1),:,:]

			least_usage = self.build_least_usage(self.usage_weights, self.n)

			print('Read Weights (before):\n', self.read_weights)
			print('MEMORY (before):\n', self.MEMORY)
			pred, self.MEMORY, controller_state, retrieved_memory, concatenated_output, read_keys, cos_sim, self.read_weights, self.usage_weights, write_keys, write_weights = self.NTM_total.predict([s,self.MEMORY,self.read_weights,self.usage_weights,least_usage])
			self.NTM_to_fit.fit([s,self.MEMORY,self.read_weights,self.usage_weights,least_usage],y)
				
			W = self.NTM_to_fit.get_weights()
			self.NTM.set_weights(W)

			print('Controller State:\n', controller_state)
			print('Write Keys:\n', write_keys)
			print('Write Weights:\n', write_weights)
			print('MEMORY (after):\n', self.MEMORY)
			print('Read Keys:\n', read_keys)
			print('Cosine Similarity:\n', cos_sim)
			print('Read Weights (after):\n', self.read_weights)
			print('Retrieved_Memory:\n', retrieved_memory)
			print('Controller State:\n', controller_state)
			print('Concatenated Output:\n', concatenated_output)
			print('Predictions:\n', pred)
			print('----------------------------------------------')

def main():

	from TASKS.task_copy_2 import data_construction
	task = 'copy'

	X_train, Y_train = data_construction(200000, length=6, size=5, end_marker=True)	
	X_test, Y_test = data_construction(2000, length=6, size=5, end_marker=True)		
	
	S = [np.shape(X_train)[1], np.shape(X_train)[2]]
	H = 256
	N = 128
	M = 40
	O = [np.shape(Y_train)[1], np.shape(Y_train)[2]]
	
	num_heads = 1
	gamma =	0.99

	lr = 3e-5
	momentum = 0.9
	decay = 0
	minibatch = 1
		
	do_training = True
	do_test = True
	do_plots = False
	do_check = False

	verb = 1

	model = NTM(S, O, H, N, M, num_heads, lr, decay, momentum, gamma, minibatch, 'RMSprop',True)
	
	if do_check:
		np.set_printoptions(precision=2)
		X, Y = data_construction(3, length=3, size=4, end_marker=end_marker)
		print(X)
		print(np.shape(X))
		print(Y)
		print(np.shape(Y))
		print('---------------------------------------------------------------------------')
		model2 = NTM([np.shape(X)[1], np.shape(X)[2]], [np.shape(Y)[1], np.shape(Y)[2]], 10, 5, 4, 1, 3e-5, 0, 0.9, 0.99, 16,'RMSprop', True)		
		model2.check(X, Y, verb)
		
	if do_training:
		E = model.training(X_train, Y_train)
	
	if do_test:
		LOSS, ACC = model.test(X_test, Y_test)

		example_input, example_output = data_construction(1, length=6, size=5, end_marker=True)
		predicted_output = model.NTM.predict(example_input)

		print('\nExample input:')
		print(example_input)
		print('\nExample output:')
		print(example_output)
		print('\nPredicted output:')
		print(predicted_output)

	if do_plots:

		iters = np.arange(np.shape(ACC)[0])*mini_batch		

		fig = plt.figure(figsize=(20,8))
		plt.subplot(1,2,1)
		plt.plot(iters, ACC, 'b-', linewidth=2, alpha=0.8)
		plt.title('Training Accuracy (1L-NTM - Copy Task)')
		plt.ylabel('Accuracy')
		plt.xlabel('Iteration')	

		plt.subplot(1,2,2)
		plt.plot(iters, LOSS, 'r-', linewidth=2, alpha=0.8)
		plt.title('Training Loss (1L-NTM - Copy Task)')
		plt.ylabel('Loss')
		plt.xlabel('Iteration')
		plt.show()


main()
