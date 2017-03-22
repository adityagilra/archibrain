
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

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.regularizers import l2
from keras.engine.topology import Layer
from HER_level import HER_level
from HER_base import HER_base
import numpy as np

class HER_arch():

	## Attribute
	# ------------
	# H: 1-d array NLx1, list which contain every HER_level
	# NL: int, number of levels of the hierarchical predictive coding

	def __init__(self, NL, s_dim, pred_dim, learn_rate_vec, beta_vec, gamma,elig_decay_vec, reg_value=0.01,loss_fct='mse',mem_activ_fct='linear',pred_activ_fct='linear'):

		self.NL = NL
		self.s = s_dim
		self.P = pred_dim
		self.H = []

		L0 = HER_base(0,s_dim, pred_dim, learn_rate_vec[0], beta_vec[0], gamma, elig_decay_vec[0],reg_value,loss_fct,mem_activ_fct,pred_activ_fct)
		self.H.append(L0)
		for i in np.arange(NL-1):	
			L = HER_level(i+1,s_dim, pred_dim*(s_dim**(i+1)), learn_rate_vec[i+1], beta_vec[i+1],elig_decay_vec[i+1], reg_value,loss_fct,mem_activ_fct,pred_activ_fct)
			self.H.append(L)

	
	def training(self, S_train, O_train, dic_stim=None, dic_resp=None):
		
		if self.NL==1:
			print('Mono-Level Structure')
			self.H[0].base_training(S_train, O_train,dic_stim,dic_resp)
		else:
			print('Multiple-Level Structure')
			N_samples = np.shape(S_train)[0]
			r_l = [None]*self.NL
			p_l = [None]*self.NL
			m_l = [None]*(self.NL-1)
			e_l = [None]*self.NL
			e_mod_l = [None]*(self.NL-1)
			o_l = [None]*self.NL
			
			d_l = [np.zeros((1,np.shape(S_train)[1]))]*self.NL		

			delete = ''

			bias_memory = [1000,0]

			for i in np.arange(N_samples):
				print('\n-----------------------------------------------------------------------\n')
				s = S_train[i:(i+1),:]		
				o = O_train[i:(i+1),:]	
				o_l[0] = o			

				for l in np.arange(self.NL):
					# memory gating dynamics
					r_l[l] = self.H[l].memory_gating(s,bias_memory[l],gate='softmax')

				if dic_stim is not None :
					if self.NL==2:			
						print('TRAINING ITER:',i,'   s: ',dic_stim[repr(s.astype(int))],'   r0:', dic_stim[repr(r_l[0].astype(int))],'   r1:', dic_stim[repr(r_l[1].astype(int))])
					elif self.NL==3:
						print('TRAINING ITER:',i,'   s: ',dic_stim[repr(s.astype(int))],'   r0:', dic_stim[repr(r_l[0].astype(int))],'   r1:', dic_stim[repr(r_l[1].astype(int))],'   r2:', dic_stim[repr(r_l[2].astype(int))])				
						
				p_l[self.NL-1] = self.H[self.NL-1].prediction_branch.predict(r_l[l])
	
				self.H[self.NL-2].top_down(np.reshape(p_l[self.NL-1],(self.s,-1))) 
				
				# TOP-DOWN --> MODULATED PREDICTIONS
				for l in (np.arange(self.NL-1)[::-1]):
						
					p_l[l] = self.H[l].prediction_branch.predict(r_l[l])			
					m_l[l] = self.H[l].modulated_prediction_branch.predict([r_l[l],r_l[l]]) 				
					if (l!=0):
						self.H[l-1].top_down(np.reshape(m_l[l],(self.s,-1))) 
	
				for l in np.arange(self.NL):
					
					# prediction error computation: e_l is used for the bottom-up, e_mod_l for the training
					if l==0:					
						resp_i = self.H[l].compute_response(m_l[l])
						#target_wrong = (dic_resp[repr(o_l[0])]!=dic_resp[repr(resp_i)] and dic_resp[repr(o_l[0])]=='R')				
					e_l[l] = self.H[l].compute_error(p_l[l], o_l[l], resp_i)
					if( l!=self.NL-1):					
						e_mod_l[l] = self.H[l].compute_error(m_l[l], o_l[l], resp_i)

					### TRAINING OF PREDICTION WEIGHTS
					W_p = self.H[l].prediction_branch.get_weights()	
					W_m = self.H[l].memory_branch.get_weights()				
					if( l!=self.NL-1):							
						W_p[0] = W_p[0] + self.H[l].alpha*np.dot(np.transpose(r_l[l]), e_mod_l[l])
						W_m[0] = W_m[0] + np.dot(np.dot(W_p[0], np.transpose(e_mod_l[l]))*np.transpose(r_l[l]), d_l[l])
					else:
						W_p[0] = W_p[0] + self.H[l].alpha*np.dot(np.transpose(r_l[l]), e_l[l])
						W_m[0] = W_m[0] + np.dot(np.dot(W_p[0], np.transpose(e_l[l]))*np.transpose(r_l[l]), d_l[l])
					self.H[l].prediction_branch.set_weights(W_p)
					self.H[l].memory_branch.set_weights(W_m)					

					# BOTTOM-UP
					if l!=(self.NL-1):
						o_l[l+1]=np.reshape(self.H[l].bottom_up(e_l[l]),(1,-1))
											
					print('Memory Matrix (level ',l,'):\n', np.around(W_m[0],decimals=3))
					print('Prediction Matrix(level ',l,'):\n', np.around(W_p[0],decimals=3))

					# eligibility trace dynamics
					print('Eligibility trace: ',d_l[l])					
					d_l[l] = d_l[l]*self.H[l].elig_decay_const
					d_l[l][0,(np.where(s==1))[1]] = 1

				print('Output: ',dic_resp[repr(o_l[0])],'   Response: ', dic_resp[repr(resp_i)])
				print('Base Prediction: ', np.around(p_l[0],decimals=2))
				print('Prediction Error: ', np.around(e_l[0],decimals=2))
				print('Modulated Prediction: ', np.around(m_l[0],decimals=2))	
				print('Bottomed_up: \n',  np.around(np.reshape(o_l[1],(4,-1)),decimals=3))		
				print('Prediction (level 1): \n', np.around(np.reshape(p_l[1],(self.s,-1)) ,decimals=3))
				print('Error (level 1): \n', np.around(np.reshape(e_l[1],(self.s,-1)),decimals=3))

				#WIP = str(round(100*(i+1)/N_samples,2))
				#digits = len(WIP)
				# print state of the process
				#if (i==0):
					#print('TRAINING...',end="")			
				#print("{0}{1:{2}}".format(delete, WIP+'%', digits), end="",flush=True)
				#delete = "\b"*(digits+1)			

	def false_training(self, S_train, O_train, dic_stim=None, dic_resp=None):
		
		if self.NL==1:
			print('Mono-Level Structure')
			self.H[0].base_training(S_train, O_train,dic_stim,dic_resp)
		else:
			print('Multiple-Level Structure')
			N_samples = np.shape(S_train)[0]
			r_l = [None]*self.NL
			p_l = [None]*self.NL
			m_l = [None]*(self.NL-1)
			e_l = [None]*self.NL
			e_mod_l = [None]*(self.NL-1)
			o_l = [None]*self.NL
			
			d_l = [np.zeros((1,np.shape(S_train)[1]))]*self.NL		

			delete = ''

			bias_memory = 0

			for i in np.arange(N_samples):
				print('\n-----------------------------------------------------------------------\n')
				s = S_train[i:(i+1),:]		
				o = O_train[i:(i+1),:]	
				o_l[0] = o			

				for l in np.arange(self.NL):
		
					# eligibility trace dynamics
					d_l[l] = d_l[l]*self.H[l].elig_decay_const
					d_l[l][0,(np.where(s==1))[1]] = 1

					# memory gating dynamics
					if l==0:
						r_l[l] = self.H[l].memory_gating(s,bias_memory,gate='max')
					else:
						r_l[l] = self.H[l].memory_gating(s,bias_memory,gate='softmax')

				if dic_stim is not None :
					if self.NL==2:			
						print('TRAINING ITER:',i,'   s: ',dic_stim[repr(s.astype(int))],'   r0:', dic_stim[repr(r_l[0].astype(int))],'   r1:', dic_stim[repr(r_l[1].astype(int))])
					elif self.NL==3:
						print('TRAINING ITER:',i,'   s: ',dic_stim[repr(s.astype(int))],'   r0:', dic_stim[repr(r_l[0].astype(int))],'   r1:', dic_stim[repr(r_l[1].astype(int))],'   r2:', dic_stim[repr(r_l[2].astype(int))])				
				
				for l in np.arange(self.NL):		
					p_l[l] = self.H[l].prediction_branch.predict(r_l[l])
					if l==0:		
						resp_i = self.H[0].compute_response(p_l[0])

					e_l[l] = self.H[l].compute_error(p_l[l], o_l[l], resp_i)
					if l!=(self.NL-1):
						m_l[l] = p_l[l] + e_l[l]
						e_mod_l[l] = self.H[l].compute_error(m_l[l], o_l[l], resp_i)
						o_l[l+1]=np.reshape(self.H[l].bottom_up(e_l[l]),(1,-1))
					
					### TRAINING OF PREDICTION WEIGHTS
					W_p = self.H[l].prediction_branch.get_weights()	
					W_m = self.H[l].memory_branch.get_weights()
					if l!=(self.NL-1):										
						W_p[0] = W_p[0] + self.H[l].alpha*np.dot(np.transpose(r_l[l]), e_l[l])
						W_m[0] = W_m[0] + np.dot(np.dot(W_p[0], np.transpose(e_l[l]))*np.transpose(r_l[l]),d_l[l])
					else:
						W_p[0] = W_p[0] + self.H[l].alpha*np.dot(np.transpose(r_l[l]), e_l[l])
						W_m[0] = W_m[0] + np.dot(np.dot(W_p[0], np.transpose(e_l[l]))*np.transpose(r_l[l]),d_l[l])
					self.H[l].prediction_branch.set_weights(W_p)
					self.H[l].memory_branch.set_weights(W_m)
				
				resp_i = self.H[0].compute_response(m_l[0])

				print('Output: ',dic_resp[repr(o_l[0])],'   Response: ', dic_resp[repr(resp_i)])
				print('Base Prediction: ', np.around(p_l[0],decimals=2))
				print('Base Error: ', np.around(e_l[0],decimals=2))
				print('Modulated Prediction: ', np.around(m_l[0],decimals=2))	
				#print('Bottomed_up: \n',  np.around(np.reshape(o_l[1],(4,-1)),decimals=3))		
				#print('Prediction (level 1): \n', np.around(np.reshape(p_l[1],(self.s,-1)) ,decimals=3))
				#print('Error (level 1): \n', np.around(np.reshape(e_l[1],(self.s,-1)),decimals=3))

		

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
		
			RESP_list.sort()
			#print(RESP_list)

			binary = (len(RESP_list)/2==2)

			for l in np.arange(self.NL):
				self.H[l].empty_memory()

			for i in np.arange(N_samples):
					
				s = S_test[i:(i+1),:]		
				o = O_test[i:(i+1),:]	
			
				for l in np.arange(self.NL):
					r_l[l] = self.H[l].memory_gating(s)
				print('TEST ITER:',i,'   s: ',dic_stim[repr(s.astype(int))],'   r0:', dic_stim[repr(r_l[0].astype(int))],'   r1:', dic_stim[repr(r_l[1].astype(int))])
			
				p_l[self.NL-1] = self.H[self.NL-1].prediction_branch.predict(r_l[l])	
				self.H[self.NL-2].top_down(np.reshape(p_l[self.NL-1],(self.s,-1))) 

				for l in (np.arange(self.NL-1)[::-1]):
 			
					m_l[l] = self.H[l].modulated_prediction_branch.predict([r_l[l],r_l[l]]) 				
					if (l!=0):
						self.H[l-1].top_down(np.reshape(m_l[l],(self.s,-1))) 
	
				resp_ind = self.H[0].compute_response(m_l[0])

				o_print = dic_resp[repr(o)]
				r_print = np.where(resp_ind==0, RESP_list[0], RESP_list[2] ) 			
			

				if (binary):

					if (verbose):				
						print('TEST SAMPLE N.',i+1,'\t',dic_stim[repr(s.astype(int))],'\t',o_print,'\t',r_print,'\n')

					if (o_print==RESP_list[0] and r_print==RESP_list[0]):
						Feedback_table[0,0] +=1
					elif (o_print==RESP_list[0] and r_print==RESP_list[2]):
						Feedback_table[0,1] +=1
					elif (o_print==RESP_list[2] and r_print==RESP_list[0]):
						Feedback_table[1,0] +=1
					elif (o_print==RESP_list[2] and r_print==RESP_list[2]):
						Feedback_table[1,1] +=1
		
			if (binary):	
				print('Table: \n', Feedback_table)
				print('Percentage of correct predictions: ', 100*(Feedback_table[0,0]+Feedback_table[1,1])/np.sum(Feedback_table),'%')
