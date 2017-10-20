## CLASS FOR NTM MODEL in the implementation of One-shot Learning
##
## AUTHOR: Marco Martinolli
## DATE: 09.05.2017


import numpy as np
import activations as act

from keras.models import Model
from keras.layers import Input, Dense, LSTM, Lambda, Activation
from keras.layers import concatenate, Reshape, Flatten, Add, Multiply, Dot
from keras.optimizers import RMSprop
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

	def __init__(self, S, O, H=200, N=128, M=16, num_heads=4, lr=0.0001, alpha_init=0.5, gamma=0.99, dic_label=None):

		self.S = S
		self.H = H
		self.O = O

		self.N = N
		self.M = M

		self.n = num_heads
		self.alpha = alpha_init
		self.lr = lr

		self.dic_label = dic_label

		self.GAMMA = K.constant(gamma, shape=(self.N,1))
		self.MEMORY = K.zeros(shape=(self.N,self.M))
		self.usage_weights = K.zeros(shape=(self.N,1))
		self.read_weights = K.zeros(shape=(self.N,self.n))

		Add_layer = Add()
		Multiply_layer = Multiply()
		COS = Dot(axes=[2,2],normalize=True)
		Dot_layer_2_1 = Dot(axes=[2,1])
		Dot_layer_1_1 = Dot(axes=[1,1])
		Reshape_layer_n_M = Reshape((self.n,self.M))
		Reshape_layer_nM_1 = Reshape((self.n*self.M,))
		CONTROLLER = LSTM(units=self.H, activation='tanh')
		Dense_read_keys = Dense(self.n*self.M, activation='tanh')
		Dense_write_keys = Dense(self.n*self.M, activation='tanh')
		Dense_omegas = Dense(self.n, activation='sigmoid')
		Complement_layer = Lambda(lambda o:  K.ones(shape=(self.N,self.n)) - o)
		softmax_layer = Lambda(lambda x: softmax(x,axis=1))
		Sum_layer = Lambda(lambda x: K.sum(x,axis=2,keepdims=True))
		Output_layer = Dense(self.O,activation='softmax')

		# Initial input 
		controller_input = Input(shape=(1,self.S))
		least_usage = Input(shape=(self.N,1))
	
		# LSTM controller		
		controller = CONTROLLER(controller_input)

		# key outputs for memory dynamics
		read_keys = Dense_read_keys(controller) # 1 x n*M
		read_keys_reshaped = Reshape_layer_n_M(read_keys) 
		write_keys = Dense_write_keys(controller) # 1 x n*M		
		write_keys_reshaped = Reshape_layer_n_M(write_keys) 
		omegas = Dense_omegas(controller) # 1 x n

		#least_usage = Lambda(lambda x: self.build_least_usage(x,self.n))(prev_usage)

		## writing to memory
		omegas_tiled = Lambda(lambda x: K.tile(K.expand_dims(x,axis=1),(1,self.N,1)))(omegas)
		rd_part = Multiply_layer([omegas_tiled, self.read_weights])
		compl_omegas = Complement_layer(omegas_tiled)
		least_usage_tiled = Lambda(lambda x: K.tile(x,(1,1,self.n)))(least_usage)
		us_part = Multiply_layer([compl_omegas, least_usage_tiled])
		write_weights = Add_layer([rd_part,us_part])
		print('WRITE WEIGHTS: ', write_weights)	

		print('MEMORY: ', self.MEMORY)
		writing = Dot_layer_2_1([write_weights, write_keys_reshaped])
		self.MEMORY = Add_layer([self.MEMORY, writing])
		print('NEW MEMORY: ',self.MEMORY)
 
		### reading from memory
		print('READ KEYS: ',read_keys_reshaped)
		cos_sim = COS([self.MEMORY, read_keys_reshaped])  	# correct normalization? MEMORY OR NEW?
		print('COSINE_SIMILARITY: ',cos_sim)
		self.read_weights = softmax_layer(cos_sim) 			# correct softmax normalization?
		print('READ WEIGHTS: ',self.read_weights)

		write_weights_summed = Sum_layer(write_weights)
		read_weights_summed = Sum_layer(self.read_weights)
		decay_usage = Lambda(lambda x: self.GAMMA*x)(self.usage_weights)
		self.usage_weights = Add_layer([decay_usage, read_weights_summed, write_weights_summed])
		print('USAGE WEIGHTS: ', self.usage_weights)

		retrieved_memory = Dot_layer_1_1([self.read_weights, self.MEMORY])			# MEMORY OR NEW?
		print('RETRIEVED MEMORY: ',retrieved_memory)
		retrieved_memory_reshaped = Reshape_layer_nM_1(retrieved_memory)
		print('RETRIEVED MEMORY_RESHAPED: ',retrieved_memory_reshaped)

		### concatenation
		controller_output = concatenate([controller, retrieved_memory_reshaped])
		print('CONCATENATED OUTPUT: ',controller_output)

		# classifier
		main_output = Output_layer(controller_output)
		print('FINAL OUTPUT: ',main_output)

		if self.O == 2:
			loss_fct='binary_crossentropy'
		else:
			loss_fct='categorical_crossentropy' 

		# NTM model: stimulus->controller->concatenation->output
		self.NTM = Model(inputs=[controller_input,least_usage], outputs=[main_output])	
		self.NTM.compile(loss=loss_fct, optimizer=RMSprop(lr=self.lr), metrics=['accuracy'])


	def build_least_usage(self,w_u,k):

		w_u_np = K.eval(w_u)

		w_lu = np.zeros(np.shape(w_u_np))
		w_lu[np.argsort(w_u_np,axis=0)[:k], 0] = 1
	
		return w_lu

	def compute_response(self,p,gate):
		
		if gate=='max':

			resp_ind = np.argmax(p)
			r_print = self.dic_label[resp_ind]

			return r_print

	def training(self,S_train,Y_train,verbose=0):

		N = np.shape(S_train)[0]
		E = np.zeros(N)

		memory = np.zeros((self.N,self.M))
		prev_read = np.zeros((self.N, self.n))
		prev_usage = np.zeros((self.N, 1))

		for i in np.arange(N):

			#print('ITERATION: ',i)

			s = S_train[i:(i+1),:]
			s = np.reshape(s,(1,1,self.S))
			y = Y_train[i:(i+1),:]

			if i!=0:
				y_print = self.dic_label[np.argmax(y)]

			least_usage = self.build_least_usage(prev_usage,self.n)

			pred = self.NTM.predict([s,least_usage])			

			r_print = self.compute_response(pred,'max')							

			if i!=0 and r_print != y_print:
				E[i] = 1
		
			if verbose:

				if i==0:
					print('ITERATION ',i)
				else:
					print('ITERATION ',i,'\t Y:',y_print,'\t R:',r_print)
		
		return E


	def training_episodes(self,S_train,Y_train,verbose=0):

		N_ep = np.shape(S_train)[0]
		N_img = np.shape(S_train)[1]

		E_tot = np.zeros(N_ep,dtype=int)
		E_ep = np.zeros(N_img,dtype=int)

		for ep in np.arange(N_ep):
			for i in np.arange(N_img):

				s = S_train[ep:(ep+1),i:(i+1),:]
				y = Y_train[ep:(ep+1),i:(i+1),:]

				y_print = self.dic_label[np.argmax(y)]
				
				least_usage = self.build_least_usage(self.usage_weights, self.n)

				pred = self.NTM.predict([s,least_usage])
				self.NTM.fit([s,least_usage],y)			
				memory = MEMORY
				prev_read = read_weights
				prev_usage = usage_weights

				r_print = self.compute_response(pred,'max')								
 
				if i!=0 and r_print != y_print:
					E_ep[i] = 1
		
				if verbose:
					print('EPISODE ',ep,' - IMAGE ',i,'\t Y: ',y_print,'\t R: ',r_print)

			E_tot[ep] = np.mean(E_ep)
			print('EPISODE ',ep,':\t Predicion error ',E_tot[ep])	

		return E_tot				

	def test(self,S_test,Y_test,verbose=0):

		N = np.shape(S_test)[0]
		E = np.zeros(N)

		w_read = np.zeros((self.N, self.n))
		w_write = np.zeros((self.N, self.n))
		w_usage = np.zeros((self.N, 1))

		for i in np.arange(N):

			#print('ITERATION: ',i)

			s = S_train[i:(i+1),:]
			s = np.reshape(s,(1,1,self.S))
			y = Y_train[i:(i+1),:]

			if i!=0:
				y_print = self.dic_label[np.argmax(y)]

			key = self.Controller.predict(s)
			last = 2*self.n*self.M
			self.alpha = key[0,last]
			key = np.reshape(key[0,:last],(-1,2*self.n))
			#print('KEYS:\n', key)
			read_key = key[:,:self.n]			
			write_key = key[:,self.n:]	

			r, w_read, w_usage, w_write = self.memory_dynamics(read_key, write_key, w_read, w_usage, w_write)
			
			r = np.transpose(r)
			if self.n != 1:
				r = np.reshape(r,(1,-1))			

			self.NTM.fit([s,r],y,batch_size=1,verbose=0)
			main_weights = self.NTM.get_weights()
			controller_weights = self.Controller.get_weights()
			controller_weights[0:2] = main_weights[0:2]			# how to train W_hk???
			self.Controller.set_weights(controller_weights)

			pred = self.NTM.predict([s,r])
			#print(pred)
			r_print = self.compute_response(pred,'max')							

			if i!=0 and r_print != y_print:
				E[i] = 1
		
			if verbose:

				if i==0:
					print('ITERATION ',i)
				else:
					print('ITERATION ',i,'\t Y:',y_print,'\t R:',r_print)
		
		return E	


S = 410
H = 200
N = 128
M = 16
O = 10
	
num_heads = 4
lr = 0.0001
alpha_init = 0.5
gamma =	0.99
	
model = NTM(S, O, H, N, M, num_heads, lr, alpha_init, gamma)
	
