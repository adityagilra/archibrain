## CLASS FOR SINGLE LAYER OF HER ARCHITECTURE
##
## The structure of each level includes:
##	- a Working Memory (WM), which registers the stimulus according to specific gating rules 
##	- a prediction block, that provides an (error) prediction
##	- an error block, which computes the prediction error (prediction - output)
##	- the outcome block, that presents the output from feedback/error at lower level
##
## Memory dynamics, error computation top-down and bottom-up steps are implemented as described in the Supplementary Material of the paper "Frontal cortex function derives from hierarchical
## predictive coding", W. Alexander, J. Brown, equations (2), (4), (7) and (8).
##
## Version 2.0: each layer is customized to handle top-down and bottom-ups.
## AUTHOR: Marco Martinolli
## DATE: 27.02.2017

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras import backend as K
from keras.regularizers import l2
from keras.optimizers import SGD
from keras.engine.topology import Layer, Merge

import numpy as np

from keras_HER import WMBlock, PredBlock, ModBlock


class HER_level():
	
	## Inputs
	# ------------
	# l: int, id number of the level in the HER architecture 
	# S: int, dimension of the input stimulus
	# P: int, number of neurons for the prediction/output vector
	# beta: real, gain parameter for the memory gating dynamics
	
	## Variables
	# ------------
	# X: 2-d array SxM, weight matrix for memory representation of the input stimulus
	# r: 1-d array Mx1, item representation in the WM
	# W: 2-d array MxP, weight matrix for error prediction
	# p: 1-d array Px1, prediction vector
	# o: 1-d array Px1, ouput vector
	# e: 1-d array Px1, prediction error
	# a: 1-d array Px1, activity filter based on the response


	# S is the dimension of the stimulus vector, i.e. the number of stimuli to present 
	# P is equal to the double of the number of categories, i.e. for a binary classification 		
	#   problem(C1,C2) the prediction/ouput will have the form 
	#   [C1_corr, C1_wrong, C2_corr, C2_wrong]
	
	def __init__(self,l,S,P,alpha,beta,elig_decay_const,reg_value=0.01,loss_fct='mse',mem_activ_fct='linear',pred_activ_fct='linear'):
		
		self.S = S
		self.P = P
		self.level = l
		self.alpha = alpha
		self.beta = beta
		self.elig_decay_const = elig_decay_const

		self.r = None
		
		# memory_branch = S --> V
		self.memory_branch = Sequential()
		self.memory_branch.add( WMBlock(output_dim=self.S, input_dim=self.S, init='zero',activation=mem_activ_fct,W_regularizer=l2(reg_value)))
		self.memory_branch.compile(loss= loss_fct, optimizer='sgd', metrics=['accuracy'])	
		
		# prediction_branch = WM --> PredBlock
		self.prediction_branch = Sequential()
		self.prediction_branch.add(PredBlock(output_dim=self.P,input_dim=self.S,init='zero',activation=pred_activ_fct,W_regularizer=l2(reg_value)))
		self.prediction_branch.compile(loss= loss_fct, optimizer=SGD(lr=self.alpha), metrics=['accuracy'])

		# modulator block WM ---> ModBlock
		self.modulator_branch = Sequential()
		self.modulator_branch.add(ModBlock(output_dim=self.P, input_dim=self.S, activation=pred_activ_fct, W_regularizer=l2(reg_value),trainable=None))
		self.modulator_branch.compile(loss= loss_fct, optimizer='sgd', metrics=['accuracy'])
		
		# modulated prediction = prediction + modulator
		self.modulated_prediction_branch = Sequential()
		self.modulated_prediction_branch.add(Merge([self.prediction_branch, self.modulator_branch], mode='sum'))		
		self.modulated_prediction_branch.compile(loss= loss_fct, optimizer=SGD(lr=self.alpha), metrics=['accuracy'])

	def empty_memory(self):
		self.r = None
	
			
	def memory_gating(self,s,bias=10,gate='softmax'):			
		
		if (self.r is None):
			self.r = s	
		else:
			v = self.memory_branch.predict(s)
			print(np.around(v,decimals=3))
			v_i = v[np.where(s==1)]
			v_j = v[np.where(self.r==1)]
			if(gate=='softmax'):
				p_storing = (np.exp(self.beta*v_i)+bias)/(np.exp(self.beta*v_i)+np.exp(self.beta*v_j)+bias)
				print(np.around(v_i,decimals=3),'   ',np.around(v_j,decimals=3),'   ',np.around(p_storing,decimals=3))
				if p_storing is not None and np.random.random()<=p_storing:
					self.r = s
				else:
					if v_i>=v_j:
						self.r = s 					
			elif(gate=='max'):
				if v_i>=v_j:
					self.r = s 
		return self.r                      

	def compute_error(self, p, o, resp_ind):
		
		len_tot = np.shape(o)[0]*np.shape(o)[1]
		a_prime = np.zeros((1,len_tot))
		action_ind = [ i for i in np.arange(len_tot) if i%(self.P/(self.level*self.S))==2*resp_ind or i%(self.P/(self.level*self.S))==2*resp_ind+1] 
		a_prime[0,action_ind] = 1
		#print('A_PRIME: (indeces :',action_ind,' on a total of ',len_tot,')\n', np.reshape(a_prime,(self.S,-1)))
		return a_prime*(o-p)

	def bottom_up(self, e):
		return (np.dot(np.transpose(self.r),e))

	def top_down(self, P_err):

		bias_vec = np.zeros((self.P,))
		ww = [P_err, bias_vec]	
		self.modulator_branch.set_weights(ww)


