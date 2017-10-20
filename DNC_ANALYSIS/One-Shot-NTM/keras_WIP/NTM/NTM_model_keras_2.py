## CLASS FOR NTM MODEL in the implementation of One-shot Learning
##
## AUTHOR: Marco Martinolli
## DATE: 09.05.2017


import numpy as np
import activations as act
from TASKS.omniglot_task_2 import construct_episode 

from keras.models import Model
from keras.layers import Input, Dense, LSTM, Lambda, Activation
from keras.layers import concatenate, Reshape, Flatten, Add, Multiply, Dot
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import RMSprop, Adam
from keras.activations import softmax
from keras import backend as K


class NTM():

	## Inputs
	# ------------
	# S: int, dimension of the input stimulus (concatenated stimulus x_t with label y_t-1)
	# H: int, number of units in the controller
	# R: int, sze of memory location 
	# N: int, number of memory locations 
	# num_heads: int, number of activity units
	# bias: real, number of activity units	

	# alpha: scalar, value for interpolation in the write weights (LRUA)
	# gamma: scalar, decay parameter of the usage vectors
	# lr: scalar, learning rate

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

		self.depth_in_time = depth_in_time

		if self.depth_in_time==False:

			print('Time depth OFF')

			controller_input = Input(shape=(1,self.S))
			prev_usage = Input(shape=(self.N,1))
			least_usage = Input(shape=(self.N,1))
			prev_read = Input(shape=(self.N,self.n))
			MEMORY = Input(shape=(self.N,self.M))

			self.GAMMA = K.constant(gamma, shape=(self.N,1))
	
			# LSTM controller		
			controller = LSTM(units=self.H, activation='tanh',stateful=False, return_sequences=False)(controller_input)	

			# key outputs for memory dynamics
			read_keys = Dense(self.n*self.M, activation='tanh')(controller) 	# 1 x n*M
			read_keys_reshaped =  Reshape((self.n,self.M))(read_keys)
			write_keys = Dense(self.n*self.M, activation='tanh')(controller) 	# 1 x n*M		
			write_keys_reshaped = Reshape((self.n,self.M))(write_keys) 
			omegas = Dense(self.n, activation='sigmoid')(controller) 		# 1 x n

			#least_usage = Lambda(lambda x: self.build_least_usage(x,self.n))(prev_usage)

			## writing to memory
			omegas_tiled = Lambda(lambda x: K.tile(K.expand_dims(x,axis=1),(1,self.N,1)))(omegas)
			rd_part = Multiply()([omegas_tiled, prev_read])
			compl_omegas = Lambda(lambda o:  K.ones(shape=(self.N,self.n)) - o)(omegas_tiled)
			least_usage_tiled = Lambda(lambda x: K.tile(x,(1,1,self.n)))(least_usage)
			us_part = Multiply()([compl_omegas, least_usage_tiled])
			write_weights = Add()([rd_part,us_part])
			print('WRITE WEIGHTS: ', write_weights)	

			print('MEMORY: ', MEMORY)
			writing = Dot(axes=[2,1])([write_weights, write_keys_reshaped])
			NEW_MEMORY = Add()([MEMORY, writing])
			print('NEW MEMORY: ',NEW_MEMORY)
 
			### reading from memory
			print('READ KEYS: ',read_keys_reshaped)
			cos_sim = Dot(axes=[2,2],normalize=True)([NEW_MEMORY, read_keys_reshaped])  
			print('COSINE_SIMILARITY: ',cos_sim)
			read_weights = Lambda(lambda x: softmax(x,axis=1))(cos_sim) 
			print('READ WEIGHTS: ',read_weights)

			write_weights_summed = Lambda(lambda x: K.sum(x,axis=2,keepdims=True))(write_weights)
			read_weights_summed = Lambda(lambda x: K.sum(x,axis=2,keepdims=True))(read_weights)
			decay_usage = Lambda(lambda x: self.GAMMA*x)(prev_usage)
			usage_weights = Add()([decay_usage, read_weights_summed, write_weights_summed])
			print('USAGE WEIGHTS: ', usage_weights)

			retrieved_memory = Dot(axes=[1,1])([read_weights, NEW_MEMORY])
			print('RETRIEVED MEMORY: ',retrieved_memory)
			retrieved_memory_reshaped = Reshape((self.n*self.M,))(retrieved_memory)
			print('RETRIEVED MEMORY_RESHAPED: ',retrieved_memory_reshaped)

			### concatenation
			controller_output = concatenate([controller, retrieved_memory_reshaped])
			print('CONCATENATED OUTPUT: ',controller_output)

			# classifier
			main_output = Dense(self.O,activation='softmax')(controller_output)
			print('FINAL OUTPUT: ',main_output)

			loss_fct='categorical_crossentropy' 

		else:
			print('Time depth ON')

			# time depth ON, where stimulus has a time dimension (e.g. copy/copy-repeat tasks)		
			controller_input = Input(shape=(self.S[0],self.S[1]))
			prev_usage = Input(shape=(1,self.N))
			least_usage = Input(shape=(1,self.N))
			prev_read = Input(shape=(self.S[0],self.N))
			MEMORY = Input(shape=(self.N,self.M))
	
			self.GAMMA = K.constant(gamma, shape=(1,self.N))

			# LSTM controller		
			controller = LSTM(units=self.H, activation='tanh',stateful=False, return_sequences=True)(controller_input)	
			#controller1 = LSTM(self.H, stateful=False, return_sequences=True, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True)(controller_input)
			#controller2 = LSTM(self.H, stateful=False, return_sequences=True, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True)(controller1)
			#controller3 = LSTM(self.H, stateful=False, return_sequences=True, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True)(controller2)

			# key outputs for memory dynamics
			read_keys = Dense(self.M, activation='tanh')(controller) 	# 1 x n*M
			print('READ KEYS: ', read_keys)				
			write_keys = Dense(self.M, activation='tanh')(controller) 	# 1 x n*M
			print('WRITE KEYS: ', write_keys)			
			omegas = Dense(1, activation='sigmoid')(controller) 		# 1 x n
			print('OMEGAS: ', omegas)	

			##### least_usage = Lambda(lambda x: self.build_least_usage(x,self.n))(prev_usage)

			## writing to memory
			omegas_tiled = Lambda(lambda x: K.tile(x,(1,1,self.N)))(omegas)
			print('omegas tiled: ', omegas_tiled)
			print('prev read: ', prev_read)
			rd_part = Multiply()([omegas_tiled, prev_read])
			print('read part: ', rd_part)
			compl_omegas = Lambda(lambda o:  K.ones(shape=(self.S[0],self.N)) - o)(omegas_tiled)
			print('complementary omegas: ', compl_omegas)
			print('least usage: ', least_usage)
			us_part = Multiply()([compl_omegas, least_usage])
			print('usage part: ', us_part)			
			write_weights = Add()([rd_part,us_part])
			print('WRITE WEIGHTS: ', write_weights)

			print('MEMORY: ', MEMORY)
			writing = Dot(axes=[1,1])([write_weights, write_keys])
			print('writing: ', writing)
			print('shape? ',K.get_variable_shape(writing))
			NEW_MEMORY = Add()([MEMORY, writing])
			print('NEW MEMORY: ', NEW_MEMORY)
 
			### reading from memory
			print('READ KEYS: ',read_keys)
			cos_sim = Dot(axes=[2,2], normalize=True)([read_keys,NEW_MEMORY])  
			print('COSINE_SIMILARITY: ',cos_sim)
			read_weights = Lambda(lambda x: softmax(x,axis=1))(cos_sim) 
			print('READ WEIGHTS: ',read_weights)

			write_weights_summed = Lambda(lambda x: K.sum(x,axis=1,keepdims=True))(write_weights)
			print(write_weights_summed)
			read_weights_summed = Lambda(lambda x: K.sum(x,axis=1,keepdims=True))(read_weights)
			print(read_weights_summed)
			print(prev_usage)
			decay_usage = Lambda(lambda x: self.GAMMA*x)(prev_usage)
			usage_weights = Add()([decay_usage, read_weights_summed, write_weights_summed])
			print('USAGE WEIGHTS: ', usage_weights)

			retrieved_memory = Dot(axes=[2,1])([read_weights, NEW_MEMORY])
			print('RETRIEVED MEMORY: ',retrieved_memory)

			### concatenation
			controller_output = concatenate([controller, retrieved_memory])
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

		self.NTM_to_fit = Model(inputs=[controller_input,MEMORY,prev_read,prev_usage,least_usage], outputs=main_output)
		self.NTM_to_fit.compile(loss=loss_fct, optimizer=opt, metrics=['accuracy'])

		self.NTM = Model(inputs=[controller_input,MEMORY,prev_read,prev_usage,least_usage], outputs=[main_output,NEW_MEMORY, read_weights,usage_weights])	
		self.NTM.compile(loss=loss_fct, optimizer=opt, metrics=['accuracy'])


		# model just used for the ckeck		
		
		self.NTM_total = Model(inputs=[controller_input,MEMORY,prev_read,prev_usage,least_usage], outputs=[main_output, NEW_MEMORY,controller, retrieved_memory,controller_output,read_keys,cos_sim, read_weights,usage_weights,write_keys,write_weights])
		self.NTM_total.compile(loss=loss_fct, optimizer=opt, metrics=['accuracy'])	

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



	def check_episodes(self,S_train,Y_train,dic_label):	
		N_ep = np.shape(S_train)[0]
		N_img = np.shape(S_train)[1]

		self.MEMORY = np.zeros((1,self.N,self.M))
		self.read_weights = np.random.random((1,self.N, self.n))
		self.usage_weights = np.zeros((1,self.N, 1))

		for ep in np.arange(N_ep):
			for i in np.arange(N_img):

				s = S_train[ep:(ep+1),i:(i+1),:]
				y = Y_train[ep:(ep+1),i:(i+1),:]
				y = np.reshape(y,(1,-1))
				y_print = dic_label[np.argmax(y)]

				print('CHECK INPUT: ',self.dic_label[np.argmax(s[0,0,-np.shape(y)[1]:])] )
				
				least_usage = self.build_least_usage(self.usage_weights, self.n)

				pred, self.MEMORY, controller_state, retrieved_memory, concatenated_output, read_keys, cos_sim, self.read_weights, self.usage_weights, write_keys, write_weights = self.NTM_total.predict([s,self.MEMORY,self.read_weights,self.usage_weights,least_usage])
				self.NTM_to_fit.fit([s,self.MEMORY,self.read_weights,self.usage_weights,least_usage],y,batch_size=self.minibatch)
				
				W = self.NTM_to_fit.get_weights()
				self.NTM.set_weights(W)

				print('Controller State:\n', controller_state)
				print('Read Keys:\n', read_keys)
				print('Write Keys:\n', write_keys)
				print('MEMORY:\n', self.MEMORY)
				print('Cosine Similarity:\n', cos_sim)
				print('Read Weights:\n', self.read_weights)
				print('Retrieved_Memory:\n', retrieved_memory)
				print('Write Weights:\n', write_weights)
				print('Concatenated Output:\n', concatenated_output)
				print('Predictions:\n', pred)

				r = self.compute_response(pred,'prob')	
				r_print = dic_label(r)		

				print('EPISODE ',ep,' - IMAGE ',i,'\t Y: ',y_print,'\t R: ',r_print,'\n\n')
			

	def training(self,S_train,Y_train,dic_label=None,verbose=0):

		N = np.shape(S_train)[0]
		E = np.zeros(N)

		self.MEMORY = np.zeros((self.minibatch,self.N,self.M))
		if self.depth_in_time==False:
			self.read_weights = np.zeros((self.minibatch,self.N, self.n))
		else:
			self.read_weights = np.zeros((self.minibatch,self.S[0],self.N))		
		self.usage_weights = np.zeros((self.minibatch,1,self.N))

		n_batches = np.floor(N/self.minibatch).astype(int)

		for i in np.arange(n_batches):

			print('TRAINING BATCH ', i+1,'/',n_batches,'\t (minibatch size ',self.minibatch,')')

			if isinstance(self.S,int):
				s = S_train[i*self.minibatch:(i+1)*self.minibatch,:]
				s = np.reshape(s,(self.minibatch,1,self.S) )
			else:
				s = S_train[i*self.minibatch:(i+1)*self.minibatch,:,:]
				s = np.reshape(s,(self.minibatch,self.S[0],self.S[1]) )

			if isinstance(self.O,int):
				y = Y_train[i*self.minibatch:(i+1)*self.minibatch,:]
				if dic_label is not None:
					y_print = self.dic_label[np.argmax(y)]
			else:
				y = Y_train[i*self.minibatch:(i+1)*self.minibatch,:,:]

			least_usage = self.build_least_usage(self.usage_weights,self.n)

			self.NTM_to_fit.fit([s,self.MEMORY, self.read_weights, self.usage_weights,least_usage],y,batch_size=self.minibatch)
			pred, self.MEMORY, self.read_weights, self.usage_weights = self.NTM.predict([s,self.MEMORY,self.read_weights,self.usage_weights,least_usage])
				
			W = self.NTM_to_fit.get_weights()
			self.NTM.set_weights(W)
						
			if dic_label is not None:
				r = self.compute_response(pred,'prob')							
				r_print = dic_label[np.argmax(r)]

			if dic_label is not None and r_print != y_print:
				E[i] = 1
		
			if dic_label is not None and verbose:
				print('Y:',y_print,'\t R:',r_print)

		return E,least_usage


	def training_episodes(self,N_ep,N_char,N_img,path,verbose=0):		

		E_tot = np.zeros(N_ep)

		self.MEMORY = np.zeros((1,self.N,self.M))
		self.read_weights = np.zeros((1,self.N, self.n))
		self.usage_weights = np.zeros((1,self.N, 1))

		for ep in np.arange(N_ep):
			S,Y,dic = construct_episode(N_char,N_img,path,0)
			E_ep = np.zeros((N_img),dtype=int)
			for i in np.arange(N_img):

				s = S[i:(i+1),:]
				s = np.reshape(s,(1,1,-1))
				y = Y[i:(i+1),:]
				y_print = dic[np.argmax(y)]
				#print('stim:\n',s)
				#print('out:\n',y)
				
				least_usage = self.build_least_usage(self.usage_weights, self.n)

				pred, self.MEMORY, self.read_weights, self.usage_weights = self.NTM.predict([s,self.MEMORY, self.read_weights, self.usage_weights,least_usage])
				self.NTM_to_fit.fit([s,self.MEMORY, self.read_weights, self.usage_weights,least_usage],y)
				
				W = self.NTM_to_fit.get_weights()
				self.NTM.set_weights(W)

				#print(pred)
				r = self.compute_response(pred,'prob')	
				#print(r)				
				r_print = dic[r[0]]							

				if r_print!=y_print:
					E_ep[i]=1
		
				if verbose:
					print('TRAINING EPISODE ',ep,' - IMAGE ',i,'\t Y: ',y_print,'\t R: ',r_print)

			E_tot[ep] = np.mean(E_ep)
			print('TRAINING EPISODE ',ep,':\t Predicion error ',E_tot[ep])	

		print('\n\n')
		for ep in np.arange(N_ep):
			print('TRAINING EPISODE ',ep,':\t Predicion error ',E_tot[ep])	

		return E_tot				



	def test(self,S_test,Y_test, dic_label=None, verbose=0):

		N = np.shape(S_test)[0]
		E = np.zeros(N)

		n_batches = np.floor(N/self.minibatch).astype(int)

		loss = np.zeros((n_batches))
		acc = np.zeros((n_batches))

		for i in np.arange(n_batches):

			print('TEST BATCH ', i+1,'/',n_batches,'\t (minibatch size ',self.minibatch,')')

			if isinstance(self.S,int):
				s = S_test[i*self.minibatch:(i+1)*self.minibatch,:]
				s = np.reshape(s,(self.minibatch,1,self.S) )
			else:
				s = S_test[i*self.minibatch:(i+1)*self.minibatch,:,:]
				s = np.reshape(s,(self.minibatch,self.S[0],self.S[1]) )

			if isinstance(self.O,int):
				y = Y_test[i*self.minibatch:(i+1)*self.minibatch,:]
				if dic_label is not None:
					y_print = self.dic_label[np.argmax(y)]
			else:
				y = Y_test[i*self.minibatch:(i+1)*self.minibatch,:,:]

			least_usage = self.build_least_usage(self.usage_weights,self.n)

			loss[i], acc[i] = self.NTM_to_fit.evaluate([s,self.MEMORY,self.read_weights,self.usage_weights,least_usage], y, batch_size=self.minibatch)
			pred, self.MEMORY, self.read_weights, self.usage_weights = self.NTM.predict([s,self.MEMORY,
self.read_weights,self.usage_weights,least_usage])

			print(loss[i],'\t',acc[i])
			

			if dic_label is not None:
				r = self.compute_response(pred,'prob')
				r_print = dic[np.argmax(r)]							

				if r_print != y_print:
					E[i] = 1
		
			if dic_label is not None and verbose:
				print('ITERATION ',i,'\t Y:',y_print,'\t R:',r_print)
		
		return loss, acc, least_usage	


	def test_episodes(self,N_ep,N_char,N_img,path_n,verbose=0):

		E_tot = np.zeros(N_ep)

		for ep in np.arange(N_ep):
			E_ep = np.zeros(N_img,dtype=int)
			S,Y,dic = construct_episode(N_char,N_img,path_n,1)
			for i in np.arange(N_img):

				s = S[i:(i+1),:]
				s = np.reshape(s,(1,1,-1))
				y = Y[i:(i+1),:]
				y_print = dic[np.argmax(y)]
				
				least_usage = self.build_least_usage(self.usage_weights, self.n)

				pred, self.MEMORY, self.read_weights, self.usage_weights = self.NTM.predict([s, self.MEMORY,self.read_weights,self.usage_weights,least_usage])

				r = self.compute_response(pred,'prob')								
				r_print = dic[np.argmax(r)]	

				if r_print!=y_print:
					E_ep[i]=1
		
				if verbose:
					print('TEST EPISODE ',ep,' - IMAGE ',i,'\t Y: ',y_print,'\t R: ',r_print)

			E_tot[ep] = np.mean(E_ep)
			print('TEST EPISODE ',ep,':\t Predicion error ',E_tot[ep])	

		print('\n\n')
		for ep in np.arange(N_ep):
			print('TEST EPISODE ',ep,':\t Predicion error ',E_tot[ep])	

		return E_tot		

# S = 410
# H = 200
# N = 128
# M = 16
# O = 10
	
# num_heads = 4
# lr = 0.0001
# gamma = 0.99
	
# model = NTM(S, O, H, N, M, num_heads, lr, gamma)	
