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
## Version 3.0: Not using keras
## AUTHOR: Marco Martinolli
## DATE: 30.03.2017


import numpy as np
import math
import activations as act

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
	
	def __init__(self,l,S,P,alpha,beta,elig_decay_const):
		
		self.S = S
		self.P = P
		self.level = l
		self.alpha = alpha
		self.beta = beta
		self.elig_decay_const = elig_decay_const

		self.r = None
		
		self.X = np.zeros((self.S,self.S))
		self.W = np.zeros((self.S,self.P))

	def empty_memory(self):
		self.r = None
	
			
	def memory_gating(self,s,bias=0,gate='softmax'):			
		
		if (self.r is None):
			self.r = s	
		else:
			v = act.linear(s,np.transpose(self.X))
			v_i = v[np.where(s==1)]
			v_j = v[np.where(self.r==1)]

			random_value = np.random.random()
			if(gate=='softmax'):
				p_storing = ( np.exp(self.beta*v_i)+bias )/( np.exp(self.beta*v_i)+np.exp(self.beta*v_j)+bias )
				#print(v_i,'\t',v_j,'\t',p_storing,'\t',random_value)
				if math.isnan(p_storing)==False:    # because of the exponential and beta=12 , the probability value could be a NaN
					if random_value<=p_storing:
						self.r = s 
				else:        # if the storing probability is NaN I adopt the maximum criterion
					if v_i>=v_j:
						self.r = s 					
			elif(gate=='max'):
				if v_i>=v_j:
					self.r = s 
		return self.r                      

	def compute_error(self, p, o, a):
		return a*(o-p)

	def bottom_up(self, e):
		o_prime = np.dot(np.transpose(self.r),e)
		return np.reshape(o_prime,(1,-1))

	def top_down(self, P_err):
		#self.P_prime = np.transpose(np.reshape(P_err,(-1,self.S)))
		self.P_prime = np.reshape(P_err,(self.S,-1))

