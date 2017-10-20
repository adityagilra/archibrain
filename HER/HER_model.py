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
from task_12AX import data_construction

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


		L0 = HER_base(0,s_dim, pred_dim, learn_rate_vec[0], learn_rate_memory[0], beta_vec[0], gamma, elig_decay_vec[0],init,dic_resp)
		self.H.append(L0)
		for i in np.arange(NL-1):	
			L = HER_level(i+1,s_dim, pred_dim*(s_dim**(i+1)), learn_rate_vec[i+1], learn_rate_memory[i+1], beta_vec[i+1],elig_decay_vec[i+1],init)
			self.H.append(L)


	def training(self, N_trials, p_target, bias_memory,gt='softmax',stop=True,conv_criterion='human'):
		
		if self.NL==1:
			print('Mono-Level Structure')
			self.H[0].base_training(S_train, O_train)
		else:
			print('Multiple-Level Structure')
			
			E = np.zeros((N_trials))

			# variables initialization
			p_l = [None]*self.NL
			m_l = [None]*self.NL
			e_l = [None]*self.NL
			e_mod_l = [None]*self.NL
			o_l = [None]*self.NL
			a = [None]*self.NL

			d_l = [np.zeros((1,8))]*self.NL		

			correct = 0
			convergence = False
			string = None
			conv_iter = np.array([0])
	
			correct_target_trials = np.zeros(50)
			target_trial = 0
			acc = 0

			for tr in np.arange(N_trials):

				#print('TRIAL ',tr+1)

				S_train, O_train = data_construction(1,p_target)

				N_samples = np.shape(S_train)[0]

				self.reset_memory()

				valid_loop=False
				correct_trial_bool=True	

				for i in np.arange(N_samples):

					s = S_train[i:(i+1),:]	
					o = O_train[i:(i+1),:]

					s_print = self.dic_stim[repr(s.astype(int))]
					o_print = self.dic_resp[repr(o.astype(int))]

					if (s_print=='X' or s_print=='Y') and valid_loop==False:
						valid_loop = True
						target_trial += 1

					o_l[0] = o	

					for l in np.arange(self.NL):
					
						# eligibility trace dynamics					
						d_l[l] = d_l[l]*self.H[l].elig_decay_const
						d_l[l][0,(np.where(s==1))[1]] = 1
						#print('Eligibility trace: ', d_l[l])

						# memory gating dynamics
						self.H[l].memory_gating(s,s_print,bias_memory[l],gt)
	
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
								E[tr] += 1
								correct_trial_bool=False	
							
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

						dX = np.dot(e_mod_l[l], np.transpose(self.H[l].W_mod) )*self.H[l].r					
						deltaX = np.dot(np.transpose(d_l[l]),dX)  	
						self.H[l].X += self.H[l].alpha_mem*deltaX	# dX = 0.1 d^T (e_mod W_mod^T * r)					

						deltaW = np.dot(np.transpose(self.H[l].r), e_mod_l[l]) 
						self.H[l].W += self.H[l].alpha*deltaW            	 # dW = alpha*(r^T e_mod)		

				if valid_loop==True:
				
					if correct_trial_bool==True:
						correct_target_trials[(target_trial-1)%50] = 1
					else:
						correct_target_trials[(target_trial-1)%50] = 0
					acc = np.mean(correct_target_trials)
					#print(acc)			
	
				if conv_criterion=='strong' and correct>=1000 and convergence==False:
					convergence= True
					conv_iter = np.array([tr])
					string = 'SIMULATION MET CRITERION AT ITERATION: '+str(conv_iter)	
					if stop:
						break
			
				if conv_criterion=='human' and acc>=0.9 and convergence==False:
					convergence= True
					conv_iter = np.array([tr])
					string = 'SIMULATION MET CRITERION AT ITERATION: '+str(conv_iter)	
					if stop:
						break
			
			if string is not None:			
				print(string)		

			return E,conv_iter

	def test(self, N_test, p_target, bias, gt='softmax'):

		if self.NL==1:
			self.H[0].base_test(N_test, p_target) 
		else:
			p_l = [None]*self.NL
			m_l = [None]*self.NL

			corr_ep = 0
		
			for tr in np.arange(N_test):

				S_test, O_test = data_construction(1,p_target)

				N_samples = np.shape(S_test)[0]

				self.reset_memory()

				corr_ep_bool = True

				for i in np.arange(N_samples):

					s = S_test[i:(i+1),:]	
					o = O_test[i:(i+1),:]
					s_print = self.dic_stim[repr(s.astype(int))]
					for l in np.arange(self.NL):
						self.H[l].memory_gating(s,s_print,bias[l],gt)

					for l in (np.arange(self.NL)[::-1]):
						p_l[l] = act.linear(self.H[l].r, self.H[l].W)
						if l==self.NL-1:
							m_l[l] = p_l[l]
						else:
							m_l[l] = p_l[l] + act.linear(self.H[l].r, self.H[l].P_prime) 
						if (l!=0):
							self.H[l-1].top_down(m_l[l]) 
	
					resp_ind,prob = self.H[0].compute_response(m_l[0])

					o_print = self.dic_resp[repr(o.astype(int))]
					r_print = self.dic_resp[repr(resp_ind)]
					
					if o_print!=r_print:
						corr_ep_bool = False		
				
				if corr_ep_bool:
					corr_ep += 1

			perc = 100*float(corr_ep)/float(N_test)

			print('Percentage of correct predictions: ', perc,'%')

			return perc

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


	def reset_memory(self):
		for l in np.arange(self.NL):
			self.H[l].r = np.zeros((1,self.H[l].S))


	def try_again(self,tr,s,o,d_l,phase,gt='softmax'):

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
						
			# memory gating dynamics
			self.H[l].memory_gating(s,s_print,0,gt)			
						
				
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
				a[l] = np.where(o_l[l]!=0,1,0)	
				e_l[l] = self.H[l].compute_error(p_l[l], o_l[l], a[l])					
				e_mod_l[l] = self.H[l].compute_error(m_l[l], o_l[l], a[l])
						
			# BOTTOM-UP
			if l!=(self.NL-1):
				o_l[l+1]=self.H[l].bottom_up(e_l[l])

			# UPDATE!
			dX = np.dot(e_mod_l[l], np.transpose(self.H[l].W_mod) )*self.H[l].r	
			deltaX = np.dot(np.transpose(d_l[l]),dX)  	
			self.H[l].X += self.H[l].alpha_mem*deltaX	# dX = 0.1 d^T (e_mod W_mod^T * r)					

			deltaW = np.dot(np.transpose(self.H[l].r), e_mod_l[l]) 
			self.H[l].W += self.H[l].alpha*deltaW            	 # dW = alpha*(r^T e_mod)
							
							
		print('\t\t OUT: ',o_print,'\t RESP: ',r_print)

		return r_print


	def training_saccade(self, S_train, O_train, bias_memory,gt='softmax', stop=True):
		
		if self.NL==1:
			print('Mono-Level Structure')
			self.H[0].base_training(S_train, O_train)
		else:
			print('Multiple-Level Structure')
			
			N_trials = np.shape(S_train)[0]
			N_timesteps = np.shape(S_train)[1]

			# variables initialization
			p_l = [None]*self.NL
			m_l = [None]*self.NL
			e_l = [None]*self.NL
			e_mod_l = [None]*self.NL
			o_l = [None]*self.NL
			a = [None]*self.NL
			d_l = [np.zeros((1,np.shape(S_train)[2]))]*self.NL		

			E_fix = np.zeros(N_trials)
			E_go = np.zeros(N_trials)

			phase = 'start'
			abort = False

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

			for tr in np.arange(N_trials):

				#print('TRIAL N.',tr+1)

				S_tr = S_train[tr,:,:]
				O_tr = O_train[tr,:,:]	

				self.reset_memory()	
				abort=False						
			
				for i in np.arange(N_timesteps):
					
					if abort==False:
			
						s = S_tr[i:(i+1),:]		
						o = O_tr[i:(i+1),:]	
						o_l[0] = o

						s_print = self.dic_stim[repr(s.astype(int))]
						o_print = self.dic_resp[repr(o.astype(int))]

						if s_print=='e' and phase!='delay':			# empty screen, begin of the trial
							phase = 'start'
							resp=False
						elif  phase=='start' and (s_print=='P' or s_print=='A'):   	# fixation mark appers
							phase = 'fix'	
							cue_fix = s_print
						elif (s_print=='L' or s_print=='R'): 			# location cue
							phase = 'cue'
							cue_loc = s_print
						elif (s_print=='P' or s_print=='A') and phase=='cue':	   	# memory delay
							phase = 'delay'
						elif s_print=='e' and phase=='delay':	 # go = solve task
							phase = 'go'
	
						for l in np.arange(self.NL):
					
							# eligibility trace dynamics					
							d_l[l] = d_l[l]*self.H[l].elig_decay_const
							d_l[l][0,(np.where(s==1))[1]] = 1
							# memory gating dynamics
							self.H[l].memory_gating(s,s_print,bias_memory[l],gt)			
						
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
							else:
								r_print=None
								
							if l==0:
								if phase=='start':
									ph = 'st'
								elif phase=='fix':
									ph = 'fx'
								elif phase=='cue':
									ph = 'lc'
								elif phase=='delay':
									ph = 'dl'
								elif phase=='go':
									ph = 'go'
	
								#if self.NL==2:
								#	print('\t ',ph,'\t s: ',s_print,'\t\t r0: ',self.dic_stim[repr(self.H[0].r.astype(int))],'\t r1: ',self.dic_stim[repr(self.H[1].r.astype(int))],'\t\t OUT: ',o_print,'\t RESP: ',r_print)

						if phase=='fix' and r_print!=o_print:
							E_fix[tr]+=1
							E_go[tr]+=1
							abort=True

						if phase=='go' and r_print!=o_print:
							E_go[tr]+=1

						if phase=='go':
							if (cue_fix=='P' and cue_loc=='L'):
								num_PL += 1
								if r_print==o_print:
									trial_PL[(num_PL-1)%50] = 1
								else:
									trial_PL[(num_PL-1)%50] = 0
								prop_PL = np.mean(trial_PL)
							elif (cue_fix=='P' and cue_loc=='R'):
								num_PR += 1
								if r_print==o_print:
									trial_PR[(num_PR-1)%50] = 1
								else:
									trial_PR[(num_PR-1)%50] = 0
								prop_PR = np.mean(trial_PR)
							elif (cue_fix=='A' and cue_loc=='L'):
								num_AL += 1
								if r_print==o_print:
									trial_AL[(num_AL-1)%50] = 1
								else:
									trial_AL[(num_AL-1)%50] = 0
								prop_AL = np.mean(trial_AL)
							elif (cue_fix=='A' and cue_loc=='R'):
								num_AR += 1
								if r_print==o_print:
									trial_AR[(num_AR-1)%50] = 1
								else:
									trial_AR[(num_AR-1)%50] = 0
								prop_AR = np.mean(trial_AR)
	
							if prop_PL>=0.9 and prop_PR>=0.9 and prop_AL>=0.9 and prop_AR>=0.9:
								conv_iter = np.array([tr])
								convergence=True
				if np.remainder(tr,100)==0:
					print('TRIAL ',tr,'\t PL:',prop_PL,' PR:',prop_PR,' AL:',prop_AL,' AR:',prop_AR)

				if convergence==True:
					if stop==True:			
						break

		print('SIMULATION CONVERGED AT TRIAL ', conv_iter)		

		return E_fix,E_go,conv_iter	



	def test_saccade(self, S_test, O_test, bias_memory,verbose,gt='softmax'):
		
		if self.NL==1:
			self.H[0].base_test(S_test, O_test)
		else:
			
			N_trials = np.shape(S_test)[0]
			N_timesteps = np.shape(S_test)[1]
			# variables initialization
			p_l = [None]*self.NL
			m_l = [None]*self.NL

			phase = 'start'
			corr_fix = 0
			corr_go = 0

			for tr in np.arange(N_trials):

				S_tst = S_test[tr,:,:]
				O_tst = O_test[tr,:,:]	

				self.reset_memory()	
				
				for i in np.arange(N_timesteps):
					
					s = S_tst[i:(i+1),:]		
					o = O_tst[i:(i+1),:]	

					s_print = self.dic_stim[repr(s.astype(int))]
					o_print = self.dic_resp[repr(o.astype(int))]

					if s_print=='e' and phase!='delay':			# empty screen, begin of the trial
						phase = 'start'
					elif  phase=='start' and (s_print=='P' or s_print=='A'):   	# fixation mark appers
						phase = 'fix'
					elif (s_print=='L' or s_print=='R'): 			# location cue
						phase = 'cue'
					elif (s_print=='P' or s_print=='A') and phase=='cue':	   	# memory delay
						phase = 'delay'
					elif s_print=='e' and phase=='delay':	 # go = solve task
						phase = 'go'
	
					for l in np.arange(self.NL):
						# memory gating dynamics
						self.H[l].memory_gating(s,s_print,bias_memory[l],gt)			
					
				
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

			perc_fix = 100*corr_fix/N_trials
			perc_go = 100*corr_go/N_trials

			return perc_fix, perc_go
