## CLASS FOR NTM MODEL in the implementation of One-shot Learning
##
## AUTHOR: Marco Martinolli
## DATE: 09.05.2017


import numpy as np
import activations as act

from keras.models import Model
from keras.layers import Input, Dense, LSTM, concatenate
from keras.optimizers import RMSprop
from keras import backend as K


def cosine_similarity(x, y):
	
	x_norm = K.l2_normalize(x,axis=1)
	y_norm = K.l2_normalize(y,axis=1)
	
	z = K.dot(y_norm, K.transpose(x_norm))

	return z

def build_least_usage(w_u, k):

	usage_vec = K.eval(w_u)
	w_lu = np.zeros(np.shape(usage_vec))
	
	max_usage = np.max(usage_vec)

	for i in np.arange(k):
		min_ind = np.argmin(usage_vec)
		usage_vec[min_ind] = max_usage
		w_lu[min_ind] = 1 

	w_lu = K.variable(w_lu)

	return w_lu

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
		self.gamma = gamma
		self.lr = lr

		self.dic_label = dic_label


		# initialization
		self.MEMORY = K.zeros(shape=(self.N, self.M))
		usage_weights = K.zeros(shape=(self.N,1))
		write_weights = K.zeros(shape=(self.N,self.n))
		read_weights = K.zeros(shape=(self.N,self.n))

		# Initial input 
		controller_input = Input(shape=(1,self.S,))
	
		# LSTM controller		
		controller = LSTM(units=self.H, activation='tanh')(controller_input)

		# key outputs for memory dynamics
		read_keys = Dense(self.n*self.M, activation='tanh')(controller) # 1 x n*M
		read_keys = K.reshape(read_keys,(self.n,self.M)) 
		write_keys = Dense(self.n*self.M, activation='tanh')(controller) # 1 x n*M
		write_keys = K.reshape(write_keys,(self.n,self.M))
		omegas = Dense(self.n, activation='sigmoid')(controller) # 1 x n

		# least usage vector before writing/reading
		least_usage = build_least_usage(usage_weights, self.n)

		# writing to memory
		omegas_tiled = K.tile(omegas,(self.N,1))
		compl_omegas = K.ones(shape=(self.N,self.n)) - omegas_tiled
		least_usage_tiled = K.tile(least_usage,(1,self.n))
		write_weights = K.update(write_weights, omegas_tiled*read_weights + compl_omegas*least_usage_tiled)
		
		self.MEMORY = K.update_add(self.MEMORY, K.dot(write_weights, write_keys))
 
		# reading from memory
		cos_sim = cosine_similarity(read_keys, self.MEMORY)
		read_weights = K.update(read_weights, K.transpose(K.softmax(K.transpose(cos_sim))))

		retrieved_memory = K.dot(K.transpose(read_weights), self.MEMORY)
		retrieved_memory = K.reshape(retrieved_memory, (1,self.n*self.M))

		# usage vector update
		usage_weights = K.update(usage_weights, self.gamma*usage_weights + K.sum(read_weights, axis=1,keepdims=True) + K.sum(write_weights, axis=1,keepdims=True) )

		# concatenation
		controller_output = K.concatenate([controller, retrieved_memory])
		
		# classifier
		main_output = Dense(self.O,activation='softmax')(controller_output)
 
		if self.O == 2:
			loss_fct='binary_crossentropy'
		else:
			loss_fct='categorical_crossentropy' 

		# NTM model: stimulus->controller->concatenation->output
		self.NTM = Model(inputs=controller_input, outputs=main_output)	
		self.NTM.compile(loss=loss_fct, optimizer=RMSprop(lr=self.lr), metrics=['accuracy'])


	def read_weights(self,k):

		#print('Key input:\n',np.transpose(k))
		K = self.cosine_similarity(k,self.MEMORY)
		#print('Cosine similarity:\n',np.transpose(K))	
		
		w_r = np.exp(K)
		w_r = w_r/np.sum(w_r)
		
		return w_r


	def compute_probabilities(self,o):
		
		z = np.dot(self.W_op,o)

		z_exp = np.exp(z)
		norm = np.sum(z_exp)

		return z_exp/norm


	def compute_response(self,p,gate):
		
		if gate=='max':

			resp_ind = np.argmax(p)
			r_print = self.dic_label[resp_ind]

			return r_print


	def memory_dynamics(self,k_r,k_w, w_r,w_u,w_w):

		r = np.zeros((self.M,self.n))

		# still from previous time step
		w_u_threshold = np.sort(w_u,axis=None)[self.n-1]
		
		if np.sum(np.where(w_u <= w_u_threshold,1,0))==self.n:

			w_lu = np.where(w_u <= w_u_threshold, 1, 0)
		else:

			w_lu = np.where(w_u < w_u_threshold, 1, 0)
			missed = self.n - np.sum(np.where(w_u < w_u_threshold,1,0))
			pos = np.where(w_u == w_u_threshold)
				
			for m in np.arange(missed):
				w_lu[pos[0][m]] = 1	 

		if  np.sum(np.where(w_lu==1,1,0))!=self.n:
			print('ERROR in least usage vector')
		#print(w_lu)

		# current time step
		interp_value = act.sigmoid(self.alpha)

		for head in np.arange(self.n):
			
			#print('-- HEAD ',head)			
			#print(np.shape(w_w[:,head:(head+1)]))
			#print(np.shape(w_r[:,head:(head+1)]))
			#print(np.shape(w_lu))

			w_lu_head = np.zeros((self.N,1))
			w_lu_head[np.where(w_lu == 1)[0][head], 0] = 1

			w_w[:,head:(head+1)] = interp_value*w_r[:,head:(head+1)] + (1-interp_value)*w_lu_head  

			w_r[:,head:(head+1)] = self.read_weights(k_r[:,head:(head+1)])
			r[:,head:(head+1)] = np.dot(np.transpose(self.MEMORY), w_r[:,head:(head+1)])
 	
		for head in np.arange(self.n):

			# write key to memory according to write weights
			self.MEMORY += np.dot(w_w[:,head:(head+1)], np.transpose(k_w[:,head:(head+1)]))

		w_u = self.gamma*w_u + np.reshape(np.sum(w_r,axis=1),(-1,1)) + np.reshape(np.sum(w_w,axis=1),(-1,1))

		#print('Write weights:\n',np.transpose(w_w))
		#print('Read weights:\n',np.transpose(w_r))
		#print('Usage weights:\n',np.transpose(w_u))
		#print('Memory:\n',self.MEMORY)

		return r, w_r, w_u, w_w	


	def training(self,S_train,Y_train,verbose=0):

		N = np.shape(S_train)[0]
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


	def training_episodes(self,S_train,Y_train,verbose=0):

		N_ep = np.shape(S_train)[0]
		N_img = np.shape(S_train)[1]

		E_tot = np.zeros(N_ep,dtype=int)
		E_ep = np.zeros(N_img,dtype=int)

		w_read = np.zeros((self.N, self.n))
		w_write = np.zeros((self.N, self.n))
		w_usage = np.zeros((self.N, 1))

		for ep in np.arange(N_ep):
			for i in np.arange(N_img):

				s = S_train[ep:(ep+1),i:(i+1),:]
				y = Y_train[ep:(ep+1),i:(i+1),:]

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
					E_ep[i] = 1
		
				if verbose:
					print('EPISODE ',ep,' - IMAGE ',i,'\t Y: ',y_print,'\t R: ',r_print)

			E_tot[ep] = np.mean(E_ep)
			print('EPISODE ',ep,':\t Predicion error ',E_tot[ep])	

		return E_tot				
				

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
