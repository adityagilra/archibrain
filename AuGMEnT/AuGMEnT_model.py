## CLASS FOR AuGMEnT MODEL
##
## The model and the equations for the implementation are taken from "How Attention
## Can Create Synaptic Tags for the Learning of Working Memories in Sequential Tasks"
## by J. Rombouts, S. Bohte, P. Roeffsema.
##
## AUTHOR: Marco Martinolli
## DATE: 22.03.2017

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras import backend as K
from keras.regularizers import l2
from keras.optimizers import SGD
from keras.engine.topology import Layer, Merge

from memory_block import WMBlock

import numpy as np

class HER_level():

	## Inputs
	# ------------
	# S: int, dimension of the input stimulus for both the instantaneous and transient units
	# R: int, number of neurons for the regular units
	# M: int, number of units in the memory layer
	# A: int, number of activity units
	# alpha: scalar, decay constant of synaptic tags (< 1)
	# beta: scalar, gain parameter for update rules
	# discount: scalar, discount dactor

	def __init__(self,S,R,M,A,alpha,beta,discount,reg_value=0,loss_fct='mse'):

		self.S = S
		self.R = R
		self.M = M
		self.A = A

		self.alpha = alpha
		self.beta = beta
		self.discount = discount

		# association branch = Instantaneous Sensory Units ----> Association Units
		self.association_branch = Sequential()
		self.association_branch.add( Dense(output_dim=self.R,input_dim=self.S,init='zero',activation='sigmoid',W_regularizer=l2(reg_value)))
		self.association_branch.compile(loss= loss_fct, optimizer='sgd', metrics=['accuracy'])

		# regular branch = association_branch + activity units
		self.regular_branch = Sequential()
		self.regular_branch.add( Dense(output_dim=self.R,input_dim=self.S,init='zero',activation='sigmoid',W_regularizer=l2(reg_value)))
		self.regular_branch.add( Dense(output_dim=self.A,init='zero',activation='linear',W_regularizer=l2(reg_value)))
		self.regular_branch.compile(loss= loss_fct, optimizer='sgd', metrics=['accuracy'])

		# memory_branch = Transient Sensory Units ---> Memory Units
		self.memory_branch = Sequential()
		self.memory_branch.add( WMBlock(output_dim=self.M, input_dim=2*self.S, init='zero',activation='sigmoid',W_regularizer=l2(reg_value)))
		self.memory_branch.compile(loss= loss_fct, optimizer='sgd', metrics=['accuracy'])

		# transient branch = memory_branch + activity units
		self.transient_branch = Sequential()
		self.transient_branch.add( WMBlock(output_dim=self.M, input_dim=2*self.S, init='zero',activation='sigmoid',W_regularizer=l2(reg_value)))
		self.transient_branch.add( Dense(output_dim=self.A, init='zero',activation='linear',W_regularizer=l2(reg_value)))
		self.transient_branch.compile(loss= loss_fct, optimizer='sgd', metrics=['accuracy'])

		# activity_branch = merge of regular and transient branches
		self.activity_branch = Sequential()
		self.activity_branch.add(Merge([self.regular_branch, self.transient_branch], mode='sum'))
		self.activity_branch.compile(loss= loss_fct, optimizer='sgd', metrics=['accuracy'])



	def compute_response(self, Qvec, epsilon=0.1):

		if np.random.random()<(1-epsilon):
			# greedy choice
			resp_ind = np.argmax(Qvec)
		else:
			# softmax probabilities
			P_vec = np.exp(Qvec)
			P_vec = P_vec/np.sum(P_vec)

			# response selection
			p_cum = 0
			random_value = np.random.random()
			for i,p_i in enumerate(P_vec):
				if (random_value <= (p_i + p_cum)):
					resp_ind = i
					break
				else:
					p_cum += p_u
		return resp_ind



	def compute_transient(self, s_inst, s_old):

			diff = s_inst - s_old
			s_plus = np.where(diff>0, diff, 0)
			s_minus = np.where(diff<0, -diff, 0)

			return np.append(s_plus,s_minus)

	def training(self,S_train,O_train,epsilon=0.1):

		N_stimuli = np.shape(S_train)[0]
		Tag_w_r = 0
		Tag_w_m = 0
		old_q = 0
		sTRACE = np.zeros((2*self.S,self.M))
		s_old = np.zeros((1,self.S))

		for n in np.arange(N_stimuli):

			s_inst = S_train[n:(n+1),:]
			s_trans = self.compute_transient(s_inst, s_old)
			s_old = s_inst
			sTRACE += np.tile(np.transpose(s_trans), (1,self.M))
		 	o = O_train[n]

			y_r = self.association_branch.predict(s_inst)
			y_m = self.memory_branch.predict(s_trans)
			Q = self.activity_branch.predict([s_inst,s_trans])

			resp_ind = self.compute_response(Q,epsilon)

			q = Q[resp_ind]
			z = zeros(np.shape(Q))
			z[resp_ind] = 1
			if resp_ind==o:
				r = 1
			else:
				r = 0

			RPE = (r + self.discount*q) - old_q  # Reward Prediction Error
			old_q = q

			weights_regular = self.regular_branch.get_weights()
			V_r = weights_regular[0]
			W_r = weights_regular[2]
			W_r_back = W_r

			weights_transient = self.transient_branch.get_weights()
			V_m = weights_transient[0]
			W_m = weights_transient[2]
			W_m_back = W_m

			Tag_w_r += -self.alpha*Tag_w_r + np.dot(np.transpose(y_r), z)
			Tag_w_r_back += np.transpose(Tag_w_r)
			Tag_w_m += -self.alpha*Tag_w_m + np.dot(np.transpose(y_m), z)
			Tag_w_m_back += np.transpose(Tag_w_m)
			W_r += self.beta*RPE*Tag_w_r
			W_m += self.beta*RPE*Tag_w_m
			W_r_back += self.beta*RPE*Tag_w_r_back
			W_m_back += self.beta*RPE*Tag_w_m_back

			Tag_v_r += -self.alpha*Tag_v_r + np.dot(np.transpose(s_inst), y_r*(1-y_r)*W_r_back[resp_ind,:])
			Tag_v_m += -self.alpha*Tag_v_m + sTRACE*y_m*(1-y_m)*W_m_back[resp_ind,:]
			V_r += self.beta*RPE*Tag_v_r
			V_m += self.beta*RPE*Tag_v_m

			weights_regular[0] = V_r
			weights_regular[2] = W_r
			self.regular_branch.set_weights(weights_regular)
			weights_transient[0] = V_m
			weights_transient[2] = W_m
			self.transient_branch.set_weights(weights_transient)

			self.association_branch.set_weights(weights_regular[:2])
			self.memory_branch.set_weights(weights_transient[:2])



def test(self,S_test,O_test,epsilon=0.1):

	N_stimuli = np.shape(S_train)[0]
	s_old = np.zeros((1,self.S))
	corr = 0
	for n in np.arange(N_stimuli):

		s_inst = S_test[n:(n+1),:]
		s_trans = self.compute_transient(s_inst, s_old)
		s_old = s_inst
		o = O_test[n]

		Q = self.activity_branch.predict([s_inst,s_trans])

		resp_ind = self.compute_response(Q,epsilon)

		if resp_ind==o:
				corr+=1

	print("Percentage of correct predictions: ", 100*corr/N_stimuli)
