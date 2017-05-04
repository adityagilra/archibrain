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

	def __init__(self, NL, s_dim, pred_dim, learn_rate_vec, learn_rate_memory,beta_vec, gamma,elig_decay_vec,dic_stim=None,dic_resp=None,init='zero'):

		self.NL = NL
		self.s = s_dim
		self.P = pred_dim
		self.H = []

		self.dic_stim = dic_stim
		self.dic_resp = dic_resp

		np.random.seed(1234)

		L0 = HER_base(0,s_dim, pred_dim, learn_rate_vec[0], learn_rate_memory[0], beta_vec[0], gamma, elig_decay_vec[0],init,dic_resp)
		self.H.append(L0)
		for i in np.arange(NL-1):	
			L = HER_level(i+1,s_dim, pred_dim*(s_dim**(i+1)), learn_rate_vec[i+1], learn_rate_memory[i+1], beta_vec[i+1],elig_decay_vec[i+1],init)
			self.H.append(L)


	def print_HER(self,matrices=False):

		print('-----HER ARCHITECTURE-----')
		for l in np.arange(self.NL):
			print('Level: ', l)
			print('Size: ')
			print('\t -stimulus dimension: ', self.H[l].S)
			print('\t -output dimension: ', self.H[l].P)			
			print('Parameters:\n')
			print('\t\t -learning rate: ', self.H[l].alpha)
			print('\t\t -learning rate for WM: ', self.H[l].alpha_mem)
			print('\t\t -eligibility decay constant: ', self.H[l].elig_decay_const)
			print('\t\t -gain param for memory gating: ', self.H[l].beta)
			if l==0:			
				print('\t\t -temperature for response selection: ', self.H[l].gamma)
			if matrices:
				print('Memory matrix:\n', self.H[l].X)
				print('Prediction matrix:\n', self.H[l].W)
			
	
	def training(self, S_train, O_train, bias_memory,learning_WM='backprop',verbose=0,gt='softmax'):
		
		if self.NL==1:
			print('Mono-Level Structure')
			self.H[0].base_training(S_train, O_train)
		else:
			print('Multiple-Level Structure')
			
			N_samples = np.shape(S_train)[0]
			E = np.zeros((N_samples))

			# variables initialization

			p_l = [None]*self.NL
			m_l = [None]*self.NL
			e_l = [None]*self.NL
			e_mod_l = [None]*self.NL
			o_l = [None]*self.NL
			a = [None]*self.NL

			d_l = [np.zeros((1,np.shape(S_train)[1]))]*self.NL		

			correct = 0
			criterion = False
			string = None
			conv_iter = np.array([0])

			for i in np.arange(N_samples):

				print('\n-----------------------------------------------------------------------------\n')
				s = S_train[i:(i+1),:]	
				o = O_train[i:(i+1),:]

				s_print = self.dic_stim[repr(s.astype(int))]
				o_print = self.dic_resp[repr(o.astype(int))]

				o_l[0] = o	

				for l in np.arange(self.NL):
					
					# eligibility trace dynamics					
					d_l[l] = d_l[l]*self.H[l].elig_decay_const
					d_l[l][0,(np.where(s==1))[1]] = 1
					#print('Eligibility trace: ', d_l[l])

					# memory gating dynamics
					self.H[l].memory_gating(s,bias_memory[l],gt)

				# MODULATED PREDICTIONS ----> TOP-DOWN
				for l in (np.arange(self.NL)[::-1]):
						
					p_l[l] = act.linear(self.H[l].r,self.H[l].W)
					if l==self.NL-1:
						self.H[l].W_mod = self.H[l].W 	
						m_l[l] = p_l[l]    
					else:					
						self.H[l].W_mod = self.H[l].W + self.H[l].P_prime	
						m_l[l] = act.linear(self.H[l].r,self.H[l].W_mod)
						m_l[l] = np.where(m_l[l]>1,1,m_l[l])
						m_l[l] = np.where(m_l[l]<0,0,m_l[l])	
					if (l!=0):
						self.H[l-1].top_down(m_l[l])
	
				for l in np.arange(self.NL):
					
					# prediction error computation: e_l is used for the bottom-up, e_mod_l for the training
					if l==0:			
		
						resp_i, p_resp = self.H[l].compute_response(m_l[l])
						r_print = self.dic_resp[repr(resp_i)]
						feedback = (o_print==r_print)
						if feedback:
							correct += 1
						else:
							correct = 0
							E[i] = 1	
						
						e_l[l], a[l] = self.H[l].compute_error(p_l[l], o_l[l], resp_i,feedback)
						e_mod_l[l], a[l] = self.H[l].compute_error(m_l[l], o_l[l], resp_i,feedback)
					
					# non-modulated prediction error					
					else:					
						#a_ = np.dot(np.transpose(self.H[l-1].r),np.ones(np.shape(a[l-1])) )
						#a[l] = np.reshape(a_,(1,-1))
						a[l] = np.where(o_l[l]!=0,1,0)						
						e_l[l] = self.H[l].compute_error(p_l[l], o_l[l], a[l])					
						e_mod_l[l] = self.H[l].compute_error(m_l[l], o_l[l], a[l])
									

					# BOTTOM-UP
					if l!=(self.NL-1):
						o_l[l+1]=self.H[l].bottom_up(e_l[l])

					### UPDATING WEIGHTS: user can decide wheter to use the backprop or reinforcement learning to train the WM weights
					if learning_WM=='backprop':	
						
						dX = np.dot(e_mod_l[l], np.transpose(self.H[l].W_mod) )*self.H[l].r					
						#deltaX = np.dot(np.transpose(dX),d_l[l])  
						deltaX = np.dot(np.transpose(d_l[l]),dX)  	
						self.H[l].X += self.H[l].alpha_mem*deltaX	# dX = 0.1 d^T (e_mod W_mod^T * r)					

						deltaW = np.dot(np.transpose(self.H[l].r), e_mod_l[l]) 
						self.H[l].W += self.H[l].alpha*deltaW            	 # dW = alpha*(r^T e_mod)
						
					elif learning_WM == 'RL':
					# Reinforcement Learning using a factor delta computed as the difference between the reward and the probability of the selected action (as explained in the supplementary information "Extended example of the HER model" - equations (S1),(S2),(S3))
						if l==0:
							correct = (self.dic_resp[repr(o)]==self.dic_resp[repr(resp_i)])	
							RL_factor = correct - p_resp  
							print('Out: ',self.dic_resp[repr(o)],'\t Resp: ',self.dic_resp[repr(resp_i)],'\t',correct,'\t',RL_factor)		
						self.H[l].W += self.H[l].alpha*np.dot(np.transpose(self.H[l].r), e_mod_l[l])  # dW = alpha*(r^T e_mod)
						self.H[l].X += self.H[l].alpha*RL_factor*np.dot(np.transpose(d_l[l]),self.H[l].r)	# dX = 0.1 * RL_factor * (d^T r)			

					if l==0 and self.dic_stim is not None :
						if self.NL==2:			
							print('TRAINING ITER:',i+1,'\ts: ',s_print,'\t\tr0: ', self.dic_stim[repr(self.H[0].r.astype(int))],'\tr1: ', self.dic_stim[repr(self.H[1].r.astype(int))],'\t\to: ',o_print,'\t\tresp: ',r_print)
						elif self.NL==3:
							print('TRAINING ITER:',i+1,'\ts: ',s_print,'\t\tr0:', self.dic_stim[repr(self.H[0].r.astype(int))],'\tr1:', self.dic_stim[repr(self.H[1].r.astype(int))],'\tr2:', self.dic_stim[repr(self.H[2].r.astype(int))],'\t\tout: ',o_print,'\t\tresp: ',r_print)				
					
					if verbose and l==0 and s_print=='X':
						print('Base prediction: ', p_l[l])
						print('Modulated prediction: ', m_l[l])
						print('Filter (Level ',l,'):\n',a[l])
						print('Modualated error(Level ',l,'):\n', np.reshape(e_mod_l[l],(self.H[l].S**(l),-1)))
						print('Modulated Weight Matrix:', self.H[l].W_mod)
						print('WM weight variation (Level ',l,'):\n', deltaX)
						print('Memory Matrix (Level ',l,'): \n', self.H[l].X)
						#print('Pred weight variation (Level ',l,'):\n', deltaW)
						#print('Prediction Matrix (Level ',l,'): \n', self.H[l].W)	
		
					if i==(N_samples-1):
							
						print('Final Memory Matrix (Level ',l,'): \n', self.H[l].X)
						print('Final Prediction Matrix (Level ',l,'): \n', self.H[l].W)	

				if correct==1000 and criterion==False:
					criterion = True
					conv_iter = np.array([i])
					string = 'SIMULATION MET CRITERION AT ITERATION: '+str(i)
			
			if string is not None:			
				print(string)		

			return E,conv_iter

	def test(self, S_test, O_test, bias, verbose=0,gt='softmax'):

		if self.NL==1:
			self.H[0].base_test(S_test, O_test) 
		else:

			N_samples = np.shape(S_test)[0]
			p_l = [None]*self.NL
			m_l = [None]*self.NL
		
			Feedback_table = np.zeros((2,2))
			RESP_list = list(self.dic_resp.values())		
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
					self.H[l].memory_gating(s,bias[l],gt)

				for l in (np.arange(self.NL)[::-1]):
					p_l[l] = act.linear(self.H[l].r, self.H[l].W)
					if l==self.NL-1:
						m_l[l] = p_l[l]
					else:
						m_l[l] = p_l[l] + act.linear(self.H[l].r, self.H[l].P_prime) 
					if (l!=0):
						self.H[l-1].top_down(m_l[l]) 
	
				resp_ind,prob = self.H[0].compute_response(m_l[0])

				o_print = self.dic_resp[repr(o)]
				r_print = np.where(resp_ind==0, RESP_list[0], RESP_list[1] ) 	

				if verbose:
					if self.NL==2:			
						print('TEST ITER:',i+1,'\ts: ',self.dic_stim[repr(s.astype(int))],'\tr0: ', self.dic_stim[repr(self.H[0].r.astype(int))],'\tr1: ', self.dic_stim[repr(self.H[1].r.astype(int))],'\tO: ',o_print,'\tR: ',r_print)
					elif self.NL==3:
						print('TEST ITER:',i+1,'\ts: ',self.dic_stim[repr(s.astype(int))],'\tr0:', self.dic_stim[repr(self.H[0].r.astype(int))],'\tr1:', self.dic_stim[repr(self.H[1].r.astype(int))],'\tr2:', self.dic_stim[repr(self.H[2].r.astype(int))],'\tO: ',o_print,'\tR: ',r_print)	
			

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



	def try_again(self,tr,s,o,d_l,phase,gt='softmax'):

		print('TRY AGAIN')

		p_l = [None]*self.NL
		m_l = [None]*self.NL
		e_l = [None]*self.NL
		e_mod_l = [None]*self.NL
		o_l = [None]*self.NL
		a = [None]*self.NL

		o_l[0]=o
		s_print = self.dic_stim[repr(s.astype(int))]	
		o_print = self.dic_resp[repr(o.astype(int))]	
	
		for l in np.arange(self.NL):
		
			# eligibility trace dynamics					
			d_l[l] = d_l[l]*self.H[l].elig_decay_const
			d_l[l][0,(np.where(s==1))[1]] = 1
			#print('Eligibility trace: ',d_l[l])
						
			# memory gating dynamics
			self.H[l].memory_gating(s,0,gt)			
						
				
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
				r_print = self.dic_resp[repr(resp_i)]
	
				e_l[l], a[l] = self.H[l].compute_error(p_l[l], o_l[l], resp_i)
				e_mod_l[l], a[l] = self.H[l].compute_error(m_l[l], o_l[l], resp_i)
											
			else:					
				#a_ = np.dot(np.transpose(self.H[l-1].r),np.ones(np.shape(a[l-1])) )
				#a[l] = np.reshape(a_,(1,-1))
				a[l] = np.where(o_l[l]!=0,1,0)	
				e_l[l] = self.H[l].compute_error(p_l[l], o_l[l], a[l])					
				e_mod_l[l] = self.H[l].compute_error(m_l[l], o_l[l], a[l])
						
			# BOTTOM-UP
			if l!=(self.NL-1):
				o_l[l+1]=self.H[l].bottom_up(e_l[l])
				#print('Bottomed up:\n',o_l[l+1])
		

			# UPDATE!
			dX = np.dot(e_mod_l[l], np.transpose(self.H[l].W_mod) )*self.H[l].r					
			#deltaX = np.dot(np.transpose(dX),d_l[l])  
			deltaX = np.dot(np.transpose(d_l[l]),dX)  	
			self.H[l].X += self.H[l].alpha_mem*deltaX	# dX = 0.1 d^T (e_mod W_mod^T * r)					

			deltaW = np.dot(np.transpose(self.H[l].r), e_mod_l[l]) 
			self.H[l].W += self.H[l].alpha*deltaW            	 # dW = alpha*(r^T e_mod)
		
							
							
		if self.dic_stim is not None:
				print('TRIAL:',tr,'-',phase,'\t\to: ',o_print,'\t\tresp: ',r_print)

		return r_print


	def training_saccade(self, N_trial, S_train, O_train, bias_memory,gt='softmax'):
		
		if self.NL==1:
			print('Mono-Level Structure')
			self.H[0].base_training(S_train, O_train)
		else:
			print('Multiple-Level Structure')
			
			N_samples = np.shape(S_train)[0]

			# variables initialization
			p_l = [None]*self.NL
			m_l = [None]*self.NL
			e_l = [None]*self.NL
			e_mod_l = [None]*self.NL
			o_l = [None]*self.NL
			a = [None]*self.NL
			d_l = [np.zeros((1,np.shape(S_train)[1]))]*self.NL		

			delete = ''

			E_fix = np.zeros(N_trial)
			E_go = np.zeros(N_trial)
			tr = -1

			phase = 'start'
			fix = 0
			delay = 0
			r = None 
			abort = False
			resp = False

			cue_fix = None
			cue_loc = None
			convergence = False
			conv_iter = np.array([0])


			trial_PL = np.zeros(50)
			trial_PR = np.zeros(50)
			trial_AL = np.zeros(50)
			trial_AR = np.zeros(50)	

			num_PL = 0
			num_PR = 0
			num_AL = 0
			num_AR = 0

			prop_PL = 0
			prop_PR = 0
			prop_AL = 0
			prop_AR = 0

			for i in np.arange(N_samples):
				if abort==True and jump!=0:
					jump-=1
					if jump==0:
						abort = False
				else:
					#print('\n-----------------------------------------------------------------------\n')
					s = S_train[i:(i+1),:]		
					o = O_train[i:(i+1),:]	
					o_l[0] = o

					s_print = self.dic_stim[repr(s.astype(int))]
					o_print = self.dic_resp[repr(o.astype(int))]

					if s_print=='empty' and phase!='delay':			# empty screen, begin of the trial
						phase = 'start'
						tr += 1
						resp=False
						#print('TRIAL N.',tr)

					elif  phase=='start' and (s_print=='P' or s_print=='A'):   	# fixation mark appers
						phase = 'fix'	
						num_fix = 0
						attempts = 0
						cue_fix = s_print
					elif (s_print=='L' or s_print=='R'): 			# location cue
						phase = 'cue'
						cue_loc = s_print
					elif (s_print=='P' or s_print=='A') and phase=='cue':	   	# memory delay
						phase = 'delay'
					elif s_print=='empty' and phase=='delay':	 # go = solve task
						phase = 'go'
						attempts = 0
						resp = False
	
					for l in np.arange(self.NL):
					
						# eligibility trace dynamics					
						d_l[l] = d_l[l]*self.H[l].elig_decay_const
						d_l[l][0,(np.where(s==1))[1]] = 1
						#print('Eligibility trace: ',d_l[l])
						
						# memory gating dynamics
						self.H[l].memory_gating(s,bias_memory[l],gt)			
						
				
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
				
						if o_print!='None':

							# prediction error computation: e_l is used for the bottom-up, e_mod_l for the training
							if l==0:
								attempts += 1					
								resp_i, p_resp = self.H[l].compute_response(m_l[l])
								r_print = self.dic_resp[repr(resp_i)]
								if phase=='fix':
									if r_print=='F':
										feedback = 1
										num_fix += 1
									else:
										feedback = 0
										num_fix = 0
								elif phase=='go':
									if r_print!='F':
										resp = True
									if r_print == o_print:
										feedback = 1
									else:
										feedback = 0
	
								e_l[l], a[l] = self.H[l].compute_error(p_l[l], o_l[l], resp_i)
								e_mod_l[l], a[l] = self.H[l].compute_error(m_l[l], o_l[l], resp_i)
						
							# non-modulated prediction error					
							else:					
								#a_ = np.dot(np.transpose(self.H[l-1].r),np.ones(np.shape(a[l-1])) )
								#a[l] = np.reshape(a_,(1,-1))
								a[l] = np.where(o_l[l]!=0,1,0)	
								e_l[l] = self.H[l].compute_error(p_l[l], o_l[l], a[l])					
								e_mod_l[l] = self.H[l].compute_error(m_l[l], o_l[l], a[l])
						
							# BOTTOM-UP
							if l!=(self.NL-1):
								o_l[l+1]=self.H[l].bottom_up(e_l[l])

							# UPDATE!
							dX = np.dot(e_mod_l[l], np.transpose(self.H[l].W_mod) )*self.H[l].r					
							#deltaX = np.dot(np.transpose(dX),d_l[l])  
							deltaX = np.dot(np.transpose(d_l[l]),dX)  	
							self.H[l].X += self.H[l].alpha_mem*deltaX	# dX = 0.1 d^T (e_mod W_mod^T * r)					

							deltaW = np.dot(np.transpose(self.H[l].r), e_mod_l[l]) 
							self.H[l].W += self.H[l].alpha*deltaW            	 # dW = alpha*(r^T e_mod)
							
						if l==0 and o_print!='None' and feedback==0 and self.dic_stim is not None:
							if self.NL==2:
								print('TRAINING TRIAL:',tr+1,'-',phase,'\ts: ',s_print,'\t\tr0: ', self.dic_stim[repr(self.H[0].r.astype(int))],'\tr1: ',self.dic_stim[repr(self.H[1].r.astype(int))],'\t\to: ',o_print,'\t\tresp: ',r_print)
		
							elif self.NL==3:			
								print('TRAINING TRIAL:',tr+1,'-',phase,'\ts: ',s_print,'\t\tr0: ', self.dic_stim[repr(self.H[0].r.astype(int))],'\tr1: ', self.dic_stim[repr(self.H[1].r.astype(int))],'\tr2: ', self.dic_stim[repr(self.H[2].r.astype(int))],'\t\to: ',o_print,'\t\tresp: ',r_print)
		
						if i==(N_samples-1):
							
							print('Final Memory Matrix (Level ',l,'): \n', self.H[l].X)
							print('Final Prediction Matrix (Level ',l,'): \n', self.H[l].W)	
				
					#print('Base prediction:\n', p_l[0])
					#print('Modulated prediction:\n',m_l[0])	
				
					if phase=='fix' and num_fix<2 and attempts==2:

						while num_fix<2 and attempts<10:
							r_print = self.try_again(tr,s,o,d_l,phase,gt)
							attempts += 1
							if r_print!='F':
								num_fix = 0
							else:
								num_fix += 1
						if attempts==10:
							E_fix[tr] = 1 
							E_go[tr] = 1		# go automatically fails
							print('No fixation. ABORT')
							abort = True
							jump = 4  # four steps to skip before next trial

					if phase=='go' and resp==False:
						while resp==False and attempts<8:
							r_print = self.try_again(tr,s,o,d_l,phase,gt)
							attempts += 1
							if r_print!='F':
								resp=True
						if attempts==8 and resp==False:
							E_go[tr] = 1
					

					if phase=='go' and convergence==False:
						if (cue_fix=='P' and cue_loc=='L'):
							num_PL += 1
							if r_print==o_print:
								trial_PL[(i-1)%50] = 1
							else:
								trial_PL[(i-1)%50] = 0
							prop_PL = np.mean(trial_PL)
						elif (cue_fix=='P' and cue_loc=='R'):
							num_PR += 1
							if r_print==o_print:
								trial_PR[(i-1)%50] = 1
							else:
								trial_PR[(i-1)%50] = 0
							prop_PR = np.mean(trial_PR)
						elif (cue_fix=='A' and cue_loc=='L'):
							num_AL += 1
							if r_print==o_print:
								trial_AL[(i-1)%50] = 1
							else:
								trial_AL[(i-1)%50] = 0
							prop_AL = np.mean(trial_AL)
						elif (cue_fix=='A' and cue_loc=='R'):
							num_AR += 1
							if r_print==o_print:
								trial_AR[(i-1)%50] = 1
							else:
								trial_AR[(i-1)%50] = 0
							prop_AR = np.mean(trial_AR)

						if prop_PL>=0.9 and prop_PR>=0.9 and prop_AL>=0.9 and prop_AR>=0.9:
							conv_iter = np.array([tr])
							convergence=True

					if resp==True and r_print!=o_print:
						E_go[tr] = 1

		return E_fix,E_go,conv_iter	



	def test_saccade(self, N_trial, S_test, O_test, bias_memory,verbose,gt='softmax'):
		
		if self.NL==1:
			print('Mono-Level Structure')
			self.H[0].base_test(S_test, O_test)
		else:
			print('Multiple-Level Structure')
			
			N_samples = np.shape(S_test)[0]

			# variables initialization
			p_l = [None]*self.NL
			m_l = [None]*self.NL

			tr = -1

			phase = 'start'
			corr_fix = 0
			corr_go = 0

			for i in np.arange(N_samples):
				#print('\n-----------------------------------------------------------------------\n')
				s = S_test[i:(i+1),:]		
				o = O_test[i:(i+1),:]	
				
				s_print = self.dic_stim[repr(s.astype(int))]
				o_print = self.dic_resp[repr(o.astype(int))]

				if s_print=='empty' and phase!='delay':			# empty screen, begin of the trial
					phase = 'start'
					tr += 1
					#print('TRIAL N.',tr)

				elif  phase=='start' and (s_print=='P' or s_print=='A'):   	# fixation mark appers
					phase = 'fix'
				elif (s_print=='L' or s_print=='R'): 			# location cue
					phase = 'cue'
				elif (s_print=='P' or s_print=='A') and phase=='cue':	   	# memory delay
					phase = 'delay'
				elif s_print=='empty' and phase=='delay':	 # go = solve task
					phase = 'go'
	
				for l in np.arange(self.NL):
					# memory gating dynamics
					self.H[l].memory_gating(s,bias_memory[l],gt)			
					
				
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
				
					if o_print!='None':

						# prediction error computation: e_l is used for the bottom-up, e_mod_l for the training
						if l==0:			
							resp_i, p_resp = self.H[l].compute_response(m_l[l])
							r_print = self.dic_resp[repr(resp_i)]
							if r_print == o_print:
								if phase=='fix':
									corr_fix += 1 	
								elif phase=='go':
									corr_go += 1

					if verbose and o_print!='None' and l==0 and self.dic_stim is not None: 	
						if self.NL==2:
							print('TEST TRIAL:',tr+1,'-',phase,'\ts: ',s_print,'\t\tr0: ', self.dic_stim[repr(self.H[0].r.astype(int))],'\tr1: ',self.dic_stim[repr(self.H[1].r.astype(int))],'\t\to: ',o_print,'\t\tresp: ',r_print)
						elif self.NL==3:			
							print('TEST TRIAL:',tr+1,'-',phase,'\ts: ',s_print,'\t\tr0: ', self.dic_stim[repr(self.H[0].r.astype(int))],'\tr1: ', self.dic_stim[repr(self.H[1].r.astype(int))],'\tr2: ', self.dic_stim[repr(self.H[2].r.astype(int))],'\t\to: ',o_print,'\t\tresp: ',r_print)



			print('Percentage of correct fix responses:', 100*corr_fix/(2*N_trial),'%')
			print('Percentage of correct go responses:', 100*corr_go/N_trial,'%')	
