## CLASS OF HER ARCHITECTURE WITH MULTIPLE LEVELS
## The structure of each level is implemented in HER_level.py
## The predictive coding is based on two pathways between subsequent HER levels:
## BOTTOM-UP: the error signal is transmitted from the lower level to the superior one;
## TOP-DOWN: the error prediction of the upper level is passed downwards to modulate the weight matrix of the inferior level.
## AUTHOR: Marco Martinolli
## DATE: 27.02.2017

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.regularizers import l2
from keras.engine.topology import Layer
from HER_level import HER_level, HER_base

import numpy as np

class HER_arch():

	## Attribute
	# ------------
	# H: 1-d array NLx1, list which contain every HER_level
	# NL: int, number of levels of the hierarchical predictive coding

	def __init__(self, NL, s_dim, M_dim, pred_dim, beta, gamma):

		self.NL = NL
		self.s = s_dim
		self.M = M_dim
		self.P = pred_dim
		self.H = []

		L0 = HER_base(0,s_dim, M_dim, pred_dim, beta, gamma)
		self.H.append(L0)
		for i in np.arange(NL-1):	
			L = HER_level(i+1,s_dim, M_dim, pred_dim*(M_dim**(i+1)), beta)
			self.H.append(L)

	
	def training(self, S_train, O_train, N_iter):
		
		N_samples = np.shape(S_train)[0]
		r_l = [None]*self.NL
		p_l = [None]*self.NL
		m_l = [None]*(self.NL-1)
		e_l = [None]*self.NL
		o_l = [None]*self.NL
		
		for i in np.arange(N_samples):
					
			print('TRAINING SAMPLE N.',i+1)
			s = S_train[i:(i+1),:]		
			o = O_train[i:(i+1),:]	
			o_l[0] = o			

			for l in np.arange(self.NL):
				r_l[l] = self.H[l].memory_gating(s)
						
			for t in np.arange(N_iter):

				p_l[self.NL-1] = self.H[self.NL-1].prediction_branch.predict(r_l[l])
	
				self.H[self.NL-2].top_down(np.reshape(p_l[self.NL-1],(self.M,-1))) 
				
				# TOP-DOWN --> MODULATED PREDICTIONS
				for l in (np.arange(self.NL-1)[::-1]):
					
					p_l[l] = self.H[l].prediction_branch.predict(r_l[l])			
					m_l[l] = self.H[l].modulated_prediction_branch.predict([r_l[l],r_l[l]]) 				
					if (l!=0):
						self.H[l-1].top_down(np.reshape(m_l[l],(self.M,-1))) 
	

				for l in np.arange(self.NL):
	
					self.H[l].memory_branch.fit(s,r_l[l],nb_epoch=1,batch_size=1,verbose=0)
					self.H[l].prediction_branch.fit(r_l[l],o_l[l],nb_epoch=1,batch_size=1,verbose=0)

					# prediction error computation
					if l==0:
						resp_i = self.H[l].compute_response(m_l[l])
						e_l[l] = self.H[l].compute_error(p_l[l], o_l[l], resp_i)
					else:
						e_l[l] = self.H[l].compute_error(p_l[l], o_l[l])  ## activity of layer>1??
				
					# BOTTOM-UP
					if l!=(self.NL-1):
						o_l[l+1]=np.reshape(self.H[l].bottom_up(e_l[l]),(1,-1))			
		

	def test(self, S_test, O_test):

		N_samples = np.shape(S_test)[0]
		r_l = [None]*self.NL
		p_l = [None]*self.NL
		m_l = [None]*(self.NL-1)
		
		Feedback_table = np.zeros((2,2))

		for l in np.arange(self.NL):
			self.H[l].empty_memory()

		for i in np.arange(N_samples):
					
			print('TEST SAMPLE N.',i+1)
			s = S_test[i:(i+1),:]		
			o = O_test[i:(i+1),:]	
			
			
			for l in np.arange(self.NL):
				r_l[l] = self.H[l].memory_gating(s)
			
			p_l[self.NL-1] = self.H[self.NL-1].prediction_branch.predict(r_l[l])	
			self.H[self.NL-2].top_down(np.reshape(p_l[self.NL-1],(self.M,-1))) 

			for l in (np.arange(self.NL-1)[::-1]):
 			
				m_l[l] = self.H[l].modulated_prediction_branch.predict([r_l[l],r_l[l]]) 				
				if (l!=0):
					self.H[l-1].top_down(np.reshape(m_l[l],(self.M,-1))) 
	
			resp_ind = self.H[0].compute_response(m_l[0])

			o_print = np.where( np.array_equiv(o,[[1,0,0,1]]), 'L', 'R')
			r_print = np.where( resp_ind==0, 'L', 'R' ) 			
			
			if (o_print=='L' and o_print==r_print):
				Feedback_table[0,0] +=1
			elif (o_print=='L' and o_print!=r_print):
				Feedback_table[0,1] +=1
			elif (o_print=='R' and o_print==r_print):
				Feedback_table[1,1] +=1
			elif (o_print=='R' and o_print!=r_print):
				Feedback_table[1,0] +=1
		
		print('TEST: \n', Feedback_table)
