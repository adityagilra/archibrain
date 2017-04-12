## CLASS OF HER ARCHITECTURE WITH MULTIPLE LEVELS
##
## The structure of each level is implemented in HER_level.py
## The predictive coding is based on two pathways between subsequent HER levels:
## BOTTOM-UP: the error signal is transmitted from the lower level to the superior one;
## TOP-DOWN: the error prediction of the upper level is passed downwards to modulate the weight matrix of the inferior level.
## General algorithm and equations for the learning  (equations (3) and (5)) derive from the Supplementary Material of the paper "Frontal cortex function derives from hierarchical 
## predictive coding", W. Alexander, J. Brown
##
## AUTHOR: Marco Martinolli
## DATE: 27.02.2017

from HER_level import HER_level
from HER_base import HER_base
import activations as act
import numpy as np

class HER_arch():

	## Attribute
	# ------------
	# H: 1-d list NLx1, list which contain every HER_level
	# NL: int, number of levels of the hierarchical predictive coding

	def __init__(self, NL, s_dim, pred_dim, learn_rate_vec, beta_vec, gamma,elig_decay_vec):

		self.NL = NL
		self.s = s_dim
		self.P = pred_dim
		self.H = []

		L0 = HER_base(0,s_dim, pred_dim, learn_rate_vec[0], beta_vec[0], gamma, elig_decay_vec[0])
		self.H.append(L0)
		for i in np.arange(NL-1):	
			L = HER_level(i+1,s_dim, pred_dim*(s_dim**(i+1)), learn_rate_vec[i+1], beta_vec[i+1],elig_decay_vec[i+1])
			self.H.append(L)

	
	def training(self, S_train, O_train, learning_WM='backprop', elig='inter',dic_stim=None, dic_resp=None):
		
		if self.NL==1:
			print('Mono-Level Structure')
			self.H[0].base_training(S_train, O_train, dic_stim, dic_resp)
		else:
			print('Multiple-Level Structure')
			
			N_samples = np.shape(S_train)[0]

			# variables initialization
			r_l = [None]*self.NL
			p_l = [None]*self.NL
			m_l = [None]*self.NL
			e_l = [None]*self.NL
			e_mod_l = [None]*self.NL
			o_l = [None]*self.NL
			a = [None]*self.NL
			d_l = [np.zeros((1,np.shape(S_train)[1]))]*self.NL		

			delete = ''

			bias_memory = [0,0,0]

			for i in np.arange(N_samples):
				print('\n-----------------------------------------------------------------------\n')
				s = S_train[i:(i+1),:]		
				o = O_train[i:(i+1),:]	
				o_l[0] = o	

				for l in np.arange(self.NL):
					
					if elig=='pre': 
						# eligibility trace dynamics					
						d_l[l] = d_l[l]*self.H[l].elig_decay_const
						d_l[l][0,(np.where(s==1))[1]] = 1
						#print('Eligibility trace: ',d_l[l])

					# memory gating dynamics
					r_l[l] = self.H[l].memory_gating(s,bias_memory[l],gate='softmax')

				if dic_stim is not None :
					if self.NL==2:			
						print('TRAINING ITER:',i+1,'\ts: ',dic_stim[repr(s.astype(int))],'\tr0: ', dic_stim[repr(r_l[0].astype(int))],'\tr1: ', dic_stim[repr(r_l[1].astype(int))],'\to: ',dic_resp[repr(o.astype(int))])
					elif self.NL==3:
						print('TRAINING ITER:',i+1,'\ts: ',dic_stim[repr(s.astype(int))],'\tr0:', dic_stim[repr(r_l[0].astype(int))],'\tr1:', dic_stim[repr(r_l[1].astype(int))],'\tr2:', dic_stim[repr(r_l[2].astype(int))])				
						
				
				# MODULATED PREDICTIONS ----> TOP-DOWN
				for l in (np.arange(self.NL)[::-1]):
						
					p_l[l] = act.linear(self.H[l].r,self.H[l].W )
					if l == self.NL-1:
						self.H[l].W_mod = self.H[l].W 	
						m_l[l] = p_l[l]    
					else:					
						self.H[l].W_mod = self.H[l].W + self.H[l].P_prime	
						m_l[l] = act.linear(self.H[l].r,self.H[l].W_mod)    				
					if (l!=0):
						self.H[l-1].top_down(m_l[l]) 
	
				for l in np.arange(self.NL):
					
					# prediction error computation: e_l is used for the bottom-up, e_mod_l for the training
					if l==0:					
						resp_i, p_resp = self.H[l].compute_response(m_l[l])	
						e_l[l], a[l] = self.H[l].compute_error(p_l[l], o_l[l], resp_i)
						e_mod_l[l], a[l] = self.H[l].compute_error(m_l[l], o_l[l], resp_i)
					
					# non-modulated prediction error					
					else:					
						a_ = np.dot(np.transpose(self.H[l-1].r),np.ones(np.shape(a[l-1])) )
						a[l] = np.reshape(a_,(1,-1))
						e_l[l] = self.H[l].compute_error(p_l[l], o_l[l], a[l])					
						e_mod_l[l] = self.H[l].compute_error(m_l[l], o_l[l], a[l])
					
					# BOTTOM-UP
					if l!=(self.NL-1):
						o_l[l+1]=self.H[l].bottom_up(e_l[l])

					if elig=='inter': 
						d_l[l][0,(np.where(s==1))[1]] = 1

					### UPDATING WEIGHTS: user can decide wheter to use the backprop or reinforcement learning to train the WM weights
					if learning_WM=='backprop':	

						deltaX = np.dot(e_mod_l[l], np.transpose(self.H[l].W_mod) )*r_l[l]							
						self.H[l].X += 0.1*np.dot(np.transpose(d_l[l]),deltaX)  	# dX = 0.1  d^T (e_mod W_mod^T * r)

						deltaW = np.dot(np.transpose(r_l[l]), e_mod_l[l]) 
						self.H[l].W += self.H[l].alpha*deltaW            	 # dW = alpha*(r^T e_mod)
						

					elif learning_WM == 'RL':
					# Reinforcement Learning using a factor delta computed as the difference between the reward and the probability of the selected action (as explained in the supplementary information "Extended example of the HER model" - equations (S1),(S2),(S3))
						if l==0:
							correct = (dic_resp[repr(o)]==dic_resp[repr(resp_i)])	
							RL_factor = correct - p_resp  
							print('Out: ',dic_resp[repr(o)],'\t Resp: ',dic_resp[repr(resp_i)],'\t',correct,'\t',RL_factor)		
						self.H[l].W += self.H[l].alpha*np.dot(np.transpose(r_l[l]), e_mod_l[l])  # dW = alpha*(r^T e_mod)
						self.H[l].X += RL_factor*np.dot(np.transpose(r_l[l]),d_l[l])	# dX = RL_factor * (r^T d)			
					
					if elig=='inter': 
						d_l[l] = d_l[l]*self.H[l].elig_decay_const
			
					if elig=='post': 
						# eligibility trace dynamics					
						d_l[l] = d_l[l]*self.H[l].elig_decay_const
						d_l[l][0,(np.where(s==1))[1]] = 1
						#print('Eligibility trace: ',d_l[l])
			
					if i==(N_samples-1):
							
						print('Final Memory Matrix (Level ',l,'): \n', self.H[l].X)
						print('Final Prediction Matrix (Level ',l,'): \n', self.H[l].W)		


	def test(self, S_test, O_test, dic_stim, dic_resp, verbose=0):

		if self.NL==1:
			self.H[0].base_test(S_test, O_test,dic_stim,dic_resp)
		else:

			N_samples = np.shape(S_test)[0]
			r_l = [None]*self.NL
			p_l = [None]*self.NL
			m_l = [None]*(self.NL-1)
		
			Feedback_table = np.zeros((2,2))
			RESP_list = list(dic_resp.values())		
			RESP_list = np.unique(RESP_list)
			RESP_list.sort()
			#print(RESP_list)

			binary = (len(RESP_list)==2)

			for l in np.arange(self.NL):
				self.H[l].empty_memory()

			for i in np.arange(N_samples):
					
				s = S_test[i:(i+1),:]		
				o = O_test[i:(i+1),:]	
			
				for l in np.arange(self.NL):
					r_l[l] = self.H[l].memory_gating(s)

				p_l[self.NL-1] = act.linear(self.H[self.NL-1].r,self.H[self.NL-1].W)	
				self.H[self.NL-2].top_down(p_l[self.NL-1]) 

				for l in (np.arange(self.NL-1)[::-1]):
					p_l[l] = act.linear(self.H[l].r, self.H[l].W)
					m_l[l] = p_l[l] + act.linear(self.H[l].r, self.H[l].P_prime) 
					if (l!=0):
						self.H[l-1].top_down(m_l[l]) 
	
				resp_ind,prob = self.H[0].compute_response(m_l[0])

				o_print = dic_resp[repr(o)]
				r_print = np.where(resp_ind==0, RESP_list[0], RESP_list[1] ) 	

				if verbose:
					if self.NL==2:			
						print('TEST ITER:',i+1,'\ts: ',dic_stim[repr(s.astype(int))],'\tr0: ', dic_stim[repr(r_l[0].astype(int))],'\tr1: ', dic_stim[repr(r_l[1].astype(int))],'\tO: ',o_print,'\tR: ',r_print)
					elif self.NL==3:
						print('TEST ITER:',i+1,'\ts: ',dic_stim[repr(s.astype(int))],'\tr0:', dic_stim[repr(r_l[0].astype(int))],'\tr1:', dic_stim[repr(r_l[1].astype(int))],'\tr2:', dic_stim[repr(r_l[2].astype(int))],'\tO: ',o_print,'\tR: ',r_print)	
			

				if (binary):

					if (o_print==RESP_list[0] and r_print==RESP_list[0]):
						Feedback_table[0,0] +=1
					elif (o_print==RESP_list[0] and r_print==RESP_list[1]):
						Feedback_table[0,1] +=1
					elif (o_print==RESP_list[1] and r_print==RESP_list[0]):
						Feedback_table[1,0] +=1
					elif (o_print==RESP_list[1] and r_print==RESP_list[1]):
						Feedback_table[1,1] +=1
		
			if (binary):	
				print('Table: \n', Feedback_table)
				print('Percentage of correct predictions: ', 100*(Feedback_table[0,0]+Feedback_table[1,1])/np.sum(Feedback_table),'%')

