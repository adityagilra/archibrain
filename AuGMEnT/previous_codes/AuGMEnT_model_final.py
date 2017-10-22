## CLASS FOR AuGMEnT MODEL
##
## The model and the equations for the implementation are taken from "How Attention
## Can Create Synaptic Tags for the Learning of Working Memories in Sequential Tasks"
## by J. Rombouts, S. Bohte, P. Roeffsema.
##
## AUTHOR: Marco Martinolli
## DATE: 10.07.2017


import numpy as np
import activations as act

class AuGMEnT():

	## Inputs
	# ------------
	# S: int, dimension of the input stimulus for both the instantaneous and transient units
	# R: int, number of neurons for the regular units
	# M: int, number of units in the memory layer
	# A: int, number of activity units

	# alpha: scalar, decay constant of synaptic tags (< 1)
	# beta: scalar, gain parameter for update rules
	# discount: scalar, discount dactor
	# epsilon: scalar, response exploration parameter
	# gain: scalar, concentration parameter for response selection
	# memory_leak: scalar, leak of the memory dynamics
	
	# rew_rule: string, defining the rewarding system for correct and wrong predictions ('RL','PL','SRL','BRL')
	# dic_stim: dictionary, with associations stimulus-label
	# dic_resp: dictionary, with associations response-label
	# prop: string, propagation system ('std','BP','RBP','SRBP','MRBP')

	def __init__(self,S,R,M,A,alpha,beta,discount,eps,gain,leak,rew_rule='RL',dic_stim=None,dic_resp=None,prop='std'):
 
		self.S = S
		self.R = R
		self.M = M
		self.A = A

		self.alpha = alpha
		self.beta = beta
		self.discount = discount
		self.epsilon = eps
		self.gain = gain
		self.memory_leak = leak

		self.dic_stim = dic_stim
		self.dic_resp = dic_resp
	
		self.prop = prop

		self.define_reward_rule(rew_rule)


	def initialize_weights_and_tags(self):

		if self.prop=='std' or self.prop=='RBP' or self.prop=='SRBP' or self.prop=='MRBP':
			self.V_r = 0.5*np.random.random((self.S,self.R)) - 0.25
			self.W_r = 0.5*np.random.random((self.R,self.A)) - 0.25

			self.V_m = 0.5*np.random.random((2*self.S,self.M)) - 0.25
			self.W_m = 0.5*np.random.random((self.M,self.A)) - 0.25

			self.W_r_back = 0.5*np.random.random((self.A,self.R)) - 0.25
			self.W_m_back = 0.5*np.random.random((self.A,self.M)) - 0.25

		elif self.prop=='BP':
			self.V_r = 0.5*np.random.random((self.S,self.R)) - 0.25
			self.W_r = 0.5*np.random.random((self.R,self.A)) - 0.25

			self.V_m = 0.5*np.random.random((2*self.S,self.M)) - 0.25
			self.W_m = 0.5*np.random.random((self.M,self.A)) - 0.25
			
			self.W_r_back = np.transpose(self.W_r)
			self.W_m_back = np.transpose(self.W_m)

		self.reset_memory()
		self.reset_tags()

	def define_reward_rule(self,rew_rule):

		if rew_rule =='RL':
			self.rew_pos = 1
			self.rew_neg = 0
		elif rew_rule =='PL':
			self.rew_pos = 0
			self.rew_neg = -1
		elif rew_rule =='SRL':
			self.rew_pos = 1
			self.rew_neg = -1
		elif rew_rule =='BRL':
			self.rew_pos = 0.1
			self.rew_neg = -1

	def compute_response(self, Qvec, it=None):

		resp_ind = None
		P_vec = None

		if np.random.random()<=(1-self.epsilon):

			# greedy choice
			if len(np.where(np.squeeze(Qvec)==np.max(Qvec)) )==1:			
				resp_ind = np.argmax(Qvec)
			else:
				resp_ind = np.random.choice(np.arange(len(Qvec)),1).item()
		else:

			# softmax probabilities
			#if it is not None:
			#	g = self.gain*(1+ (8/np.pi)*np.arctan(it/2000))
			#else:
			g = self.gain
			tot = np.clip(a=g*Qvec,a_min=-500,a_max=500)
			
			P_vec = np.exp(tot)
			if (np.isnan(P_vec)).any()==True:
				resp_ind = np.argmax(Qvec)
			else:
				P_vec = P_vec/np.sum(P_vec)
				resp_ind = np.random.choice(np.arange(len(np.squeeze(Qvec))),1,p=np.squeeze(P_vec)).item()
		
		return resp_ind, P_vec


	def update_weights(self,RPE):

		self.W_r += self.beta*RPE*self.Tag_w_r
		self.V_r += self.beta*RPE*self.Tag_v_r

		self.W_m += self.beta*RPE*self.Tag_w_m
		self.V_m += self.beta*RPE*self.Tag_v_m
		if self.prop=='std' or self.prop=='BP':
			self.W_r_back += self.beta*RPE*np.transpose(self.Tag_w_r)
			self.W_m_back += self.beta*RPE*np.transpose(self.Tag_w_m)


	def reset_memory(self):
		self.cumulative_memory = 1e-6*np.ones((1,self.M))


	def reset_tags(self):

		self.sTRACE = 1e-6*np.ones((2*self.S, self.M))

		self.Tag_v_r = 1e-6*np.ones((self.S, self.R))
		self.Tag_v_m = 1e-6*np.ones((2*self.S, self.M))

		self.Tag_w_r = 1e-6*np.ones((self.R, self.A))
		self.Tag_w_m = 1e-6*np.ones((self.M, self.A))


	def update_tags(self,s_inst,s_trans,y_r,y_m,z,resp_ind):

		self.sTRACE = self.memory_leak*self.sTRACE + np.tile(np.transpose(s_trans), (1,self.M))

		# synaptic tags for W
		self.Tag_w_r += -self.alpha*self.Tag_w_r + np.dot(np.transpose(y_r), z)
		delta_r = self.W_r_back[resp_ind,:]
		self.Tag_w_m += -self.alpha*self.Tag_w_m + np.dot(np.transpose(y_m), z)
		delta_m = self.W_m_back[resp_ind,:]

		# synaptic tags for V using feedback propagation and synaptic traces
		self.Tag_v_r += -self.alpha*self.Tag_v_r + np.dot(np.transpose(s_inst), y_r*(1-y_r)*delta_r)
		self.Tag_v_m += -self.alpha*self.Tag_v_m + self.sTRACE*y_m*(1-y_m)*delta_m


	def feedforward(self,s_inst,s_trans):

		y_r = act.sigmoid(s_inst, self.V_r)
		y_m,self.cumulative_memory = act.sigmoid_acc_leaky(s_trans, self.V_m, self.cumulative_memory,self.memory_leak)

		y_tot = np.concatenate((y_r, y_m),axis=1)
		W_tot = np.concatenate((self.W_r, self.W_m),axis=0)
		Q = act.linear(y_tot, W_tot)

		return y_r, y_m, Q


	def define_transient(self, s,s_old):

		s_plus =  np.where(s<=s_old,0,1)
		s_minus = np.where(s_old<=s,0,1)
		s_trans = np.concatenate((s_plus,s_minus),axis=1)

		return s_trans

######### TRAINING + TEST FOR CPT TASKS LIKE 12AX

	def training(self,N,S_train,O_train,reset_case,conv_criterion='strong',stop=True,verbose=False):

		self.initialize_weights_and_tags()

		N_stimuli = np.shape(S_train)[0]
		zero = np.zeros((1,self.S))

		E = np.zeros((N))
		first = True
		s_old = zero

		correct = 0
		convergence = False
		conv_iter = np.array([0])

		ep = 0
		ep_corr=0
		cont = 0

		for n in np.arange(N_stimuli):
			s = S_train[n:(n+1),:]
			if self.dic_stim[repr(s.astype(int))] in reset_case:
				if verbose:
					print('RESET \n')
				ep += 1
				ep_corr += 1
				self.reset_memory()
				self.reset_tags()
			elif first==True:
				first = False	

			s_inst = s
			s_trans = self.define_transient(s_inst,s_old)
			s_old = s
			s_print = self.dic_stim[repr(s.astype(int))]			
			o = O_train[n:(n+1),:]
			o_print = self.dic_resp[repr(o.astype(int))]
 
			y_r,y_m,Q = self.feedforward(s_inst, s_trans)
			if (np.isnan(Q)).any()==True:
				conv_iter = np.array([-1])	
				break
			resp_ind,P_vec = self.compute_response(Q)
			q = Q[0,resp_ind]
			r_print = self.dic_resp[repr(resp_ind)]
	
			z = np.zeros(np.shape(Q))
			z[0,resp_ind] = 1
	
			if verbose:
				print('ITER: ',n+1,'\t STIM: ',s_print,'\t OUT: ',o_print,'\t RESP: ', r_print,'\t\t Q: ',Q)


			if first==False:
				RPE = (r + self.discount*q) - q_old  # Reward Prediction Error
				self.update_weights(RPE)
			self.update_tags(s_inst,s_trans,y_r,y_m,z,resp_ind)
			
			if r_print!=o_print:
				r = self.rew_neg
				E[ep-1] += 1
				correct = 0
				ep_corr=0
			else: 
				r = self.rew_pos
				correct += 1			
			q_old = q


			if conv_criterion=='strong':
				if correct==1000 and convergence==False:
					conv_iter = np.array([n]) 
					convergence = True
					self.epsilon = 0
					if stop==True:
						break
			elif conv_criterion=='lenient':
				if ep_corr==51 and convergence==False:
					conv_iter = np.array([n]) 
					convergence = True					
					if stop==True:
						break
		if convergence == True:
			print('SIMULATION MET CRITERION AT ITERATION', conv_iter,'\t (',conv_criterion,' criterion)')

		return E,conv_iter

	def test(self,S_test,O_test,reset_case,verbose=False):

		N_stimuli = np.shape(S_test)[0]
		corr = 0
		binary = (self.A==2)
		Feedback_table = np.zeros((2,2))

		RESP_list = list(self.dic_resp.values())
		RESP_list = np.unique(RESP_list)
		RESP_list.sort()

		zero = np.zeros((1,self.S))
		s_old = zero

		self.reset_memory()

		self.epsilon = 0

		for n in np.arange(N_stimuli):

			s = S_test[n:(n+1), :]

			if self.dic_stim[repr(s.astype(int))] in reset_case:
				self.reset_memory()

			s_inst = s			
			s_trans = self.define_transient(s_inst,s_old)
			o = O_test[n:(n+1), :]
			s_old = s
			s_print = self.dic_stim[repr(s.astype(int))]

			y_r, y_m, Q = self.feedforward(s_inst,s_trans)
			resp_ind,P_vec = self.compute_response(Q)

			o_print = self.dic_resp[repr(o.astype(int))]
			r_print = self.dic_resp[repr(resp_ind)]

			if r_print==o_print:
					corr+=1

			if (binary):

				if (verbose):
					print('TEST SAMPLE N.',n+1,'\t',s_print,'\t',o_print,'\t',r_print,'\n')

				if (o_print==RESP_list[0] and r_print==RESP_list[0]):
					Feedback_table[0,0] += 1
				elif (o_print==RESP_list[0] and r_print==RESP_list[1]):
					Feedback_table[0,1] += 1
				elif (o_print==RESP_list[1] and r_print==RESP_list[0]):
					Feedback_table[1,0] += 1
				elif (o_print==RESP_list[1] and r_print==RESP_list[1]):
					Feedback_table[1,1] += 1
			
			s_inst = zero
			s_trans = self.define_transient(s_inst,s_old)
			s_old = s_inst
			y_r, y_m, Q = self.feedforward(s_inst,s_trans)


		if binary:
			print("PERFORMANCE TABLE (output vs. response):\n",Feedback_table)

		print("Percentage of correct predictions: ", 100*corr/N_stimuli) 


######### TRAINING + TEST FOR SACCADE TASK 

	def training_saccade(self,N_trial,S_train,O_train,reset_case,verbose=False,shape_factor=0.2):

		self.initialize_weights_and_tags()

		N_stimuli = np.shape(S_train)[0]
		zero = np.zeros((1,self.S))
		zerozero = np.concatenate((zero,zero),axis=1)
		s_old = zero

		E_fix = np.zeros(N_trial)
		E_go = np.zeros(N_trial)
		tr = -1

		phase = 'start'
		fix = 0
		delay = 0
		r = None 
		abort = False
		resp = False
	
		cue_trial = None
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

		for n in np.arange(N_stimuli):	
			
			if abort==True and jump!=0:
					jump-=1
					if jump==0:
						abort = False
			else:
				s = S_train[n:(n+1),:]
				o = O_train[n:(n+1),:]
				s_print = self.dic_stim[repr(s.astype(int))]
				o_print = self.dic_resp[repr(o.astype(int))]
				s_inst = s
				s_trans = self.define_transient(s_inst, s_old)
				s_old = s_inst
							
				if s_print=='empty' and phase!='delay':			# empty screen, begin of the trial
					if verbose:
						print('RESET \n')
					self.reset_memory()
					self.reset_tags()
					phase = 'start'
					r = None
					tr += 1
					print('TRIAL N.',tr)

				elif  phase=='start' and (s_print=='P' or s_print=='A'):   	# fixation mark appers
					phase = 'fix'	
					num_fix = 0
					attempts = 0
				elif (s_print=='AL' or s_print=='AR' or s_print=='PL' or s_print=='PR'): 	# location cue
					cue_trial = s_print
					phase = 'cue'
				elif (s_print=='P' or s_print=='A') and phase=='cue':	   	# memory delay
					phase = 'delay'
				elif s_print=='empty' and phase=='delay':	 # go = solve task
					phase = 'go'
					attempts = 0
					resp = False

				y_r,y_m,Q = self.feedforward(s_inst, s_trans)
				
				resp_ind,P_vec = self.compute_response(Q)
				q = Q[0,resp_ind]
		
				z = np.zeros(np.shape(Q))
				z[0,resp_ind] = 1

				if o_print!='None':			
					r_print = self.dic_resp[repr(resp_ind)]
					print('ITER: ',n+1,'-',phase,'\t STIM: ',s_print,'\t OUT: ',o_print,'\t RESP: ', r_print,'\t\t Q: ',Q)
				else:
					r_print = 'None'
					print('ITER: ',n+1,'-',phase,'\t STIM: ',s_print)
	
				if phase!='start' and r is not None:
					RPE = (r + self.discount*q) - q_old  # Reward Prediction Error
					self.update_weights(RPE)
	
				self.update_tags(s_inst,s_trans,y_r,y_m,z,resp_ind)

				if phase=='cue' and phase!='delay':
					r = 0
				if phase=='fix':
					attempts+=1
					if o_print!=r_print:
						if self.rew_neg!=0:
							print('Negative reward!')
						r = self.rew_neg
						num_fix = 0    # no fixation	
					else: 
						num_fix += 1     # fixation
						if num_fix==2:
							if self.rew_pos!=0:
								print('Positive reward!')
							r = shape_factor*self.rew_pos		

				elif phase=='go':
					if r_print!='F':
						resp = True
					if o_print!=r_print:
						if self.rew_neg!=0:
							print('Negative reward!')
						r = self.rew_neg 		
					else: 
						if self.rew_pos!=0:
							print('Positive reward!')
						r = 1.5*self.rew_pos					

				q_old = q
				
				if phase=='fix' and num_fix<2 and attempts==2:
					while num_fix<2 and attempts<10:
						r,q_old,fix = self.try_again(s_inst,zerozero,o_print,r,q_old,phase)
						attempts += 1
						if fix==False:
							num_fix = 0
						else:
							num_fix += 1
							if num_fix==2:
								if self.rew_pos!=0:
									print('Positive reward!')
								r=shape_factor*self.rew_pos
					if attempts==10:
						E_fix[tr] = 1 	
						E_go[tr] = 1		# automatically also go fails
						if shape_factor!=0:
							print('No fixation. ABORT')
							abort = True
							jump = 4  # four steps to skip before next trial

				if phase=='go' and resp==False:
					attempts = 1
					while resp==False and attempts<8:
						r,q_old,resp = self.try_again(s_inst,zerozero,o_print,r,q_old,phase)
						attempts += 1
						if attempts==8 and resp==False:
							E_go[tr] = 1

				if resp==True:
					if r==self.rew_neg:
						E_go[tr] = 1
					r,q_old,resp = self.try_again(s_inst,zerozero,o_print,r,q_old,phase)
					resp = False
				
				if phase=='go':

					if cue_trial=='PL':
						num_PL += 1
						if r_print == o_print:
							trial_PL[(num_PL-1) % 50] = 1
						else:
							trial_PL[(num_PL-1) % 50] = 0
						prop_PL = np.mean(trial_PL)

					if cue_trial=='PR':
						num_PR += 1
						if r_print == o_print:
							trial_PR[(num_PR-1) % 50] = 1
						else:
							trial_PR[(num_PR-1) % 50] = 0
						prop_PR = np.mean(trial_PR)

					if cue_trial=='AL':
						num_AL += 1
						if r_print == o_print:
							trial_AL[(num_AL-1) % 50] = 1
						else:
							trial_AL[(num_AL-1) % 50] = 0
						prop_AL = np.mean(trial_AL)

					if cue_trial=='AR':
						num_AR += 1
						if r_print == o_print:
							trial_AR[(num_AR-1) % 50] = 1	
						else:
							trial_AR[(num_AR-1) % 50] = 0
						prop_AR = np.mean(trial_AR)


					if convergence==False and prop_PL>=0.9 and prop_PR>=0.9 and prop_AL>=0.9 and prop_AR>=0.9:
						convergence = True
						conv_iter = np.array([tr])
						#gt = 'max'

		return E_fix,E_go,conv_iter

	def try_again(self,s_i,s_t,o_print,r,q_old,phase):

		y_r,y_m,Q = self.feedforward(s_i, s_t)
		resp_ind,P_vec = self.compute_response(Q)
		q = Q[0,resp_ind]
		z = np.zeros(np.shape(Q))
		z[0,resp_ind] = 1
		r_print = self.dic_resp[repr(resp_ind)]

		RPE = (r + self.discount*q) - q_old  # Reward Prediction Error
		self.update_weights(RPE)					
		self.update_tags(s_i,s_t,y_r,y_m,z,resp_ind)
		
		resp = False
		print('OUT: ',o_print,'\t RESP: ', r_print)

		if phase=='fix':
			if o_print!=r_print:
				if self.rew_neg!=0:
					print('Negative reward!')
				r = self.rew_neg
				resp = False		# no fixation
			else: 	
				resp = True		# fixation
				r = 0	

		if phase=='go':
			if r_print!='F':
				resp = True	
			if o_print!=r_print:
				if self.rew_neg!=0:
					print('Negative reward!')
				r = self.rew_neg	
			else: 
				if self.rew_pos!=0:
					print('Positive reward!')
				r = 1.5*self.rew_pos	

		q_old = q

		return r,q_old,resp	

	def test_saccade(self,S_test,O_test,reset_case,verbose=False):

		N_stimuli = np.shape(S_test)[0]
		zero = np.zeros((1,self.S))
		zerozero = np.concatenate((zero,zero),axis=1)
		s_old = zero

		phase = 'start'
		N_fix = 0
		corr_fix = 0
		N_go = 0	
		corr_go = 0

		self.epsilon = 0
		
		for n in np.arange(N_stimuli):	
			
			s = S_test[n:(n+1),:]
			o = O_test[n:(n+1),:]
			s_print = self.dic_stim[repr(s.astype(int))]
			o_print = self.dic_resp[repr(o.astype(int))]
			s_inst = s
			s_trans = self.define_transient(s_inst, s_old)
			s_old = s_inst
							
			if s_print=='empty' and phase!='delay':			# empty screen, begin of the trial
				if verbose:
					print('RESET \n')
				self.reset_memory()
				phase = 'start'
			elif  phase=='start' and (s_print=='P' or s_print=='A'):   	# fixation mark appers
				phase = 'fix'	
			elif (s_print=='AL' or s_print=='AR' or s_print=='PL' or s_print=='PR'): 	# location cue
				phase = 'cue'
			elif (s_print=='P' or s_print=='A') and phase=='cue':	   	# memory delay
				phase = 'delay'
			elif s_print=='empty' and phase=='delay':	 # go = solve task
				phase = 'go'
			y_r,y_m,Q = self.feedforward(s_inst, s_trans)
				
			resp_ind,P_vec = self.compute_response(Q)
			q = Q[0,resp_ind]
		
			z = np.zeros(np.shape(Q))
			z[0,resp_ind] = 1

			if o_print!='None':			
				r_print = self.dic_resp[repr(resp_ind)]
				print('ITER: ',n+1,'-',phase,'\t STIM: ',s_print,'\t OUT: ',o_print,'\t RESP: ', r_print,'\t\t Q: ',Q)
			else:
				r_print = 'None'
				print('ITER: ',n+1,'-',phase,'\t STIM: ',s_print)
			if phase=='fix':
				N_fix += 1			
				if o_print==r_print:
					corr_fix+=1			
			if phase=='go':
				N_go += 1
				if o_print==r_print:
					corr_go+=1	

		print("Percentage of correct predictions during fix: ", 100*corr_fix/N_fix)		
		print("Percentage of correct predictions during go: ", 100*corr_go/N_go)


###################################################################################################################################################################
###################################################################################################################################################################
###################################################################################################################################################################

class deep_AuGMEnT(AuGMEnT):

	## Inputs
	# ------------
	# H_r: int, number of units for the hidden layer of the controller
	# H_m: int, number of units for the hidden layer of the memory branch

	# perc_active: scalar, percentage of units considered to be active in the memory layer (units specialization)
	
	def __init__(self,S,R,M,H_r,H_m,A,alpha,beta,discount,eps,gain,leak,active_perc,rew_rule='RL',dic_stim=None,dic_resp=None,prop='std'):

		super(deep_AuGMEnT,self).__init__(S,R,M,A,alpha,beta,discount,eps,gain,leak,rew_rule,dic_stim,dic_resp,prop)
		
		self.H_r = H_r
		self.H_m = H_m

		self.perc_active = active_perc
		if H_r!=0:
			self.num_active_reg = np.round(self.H_r*active_perc).astype(int)
		else:
			self.num_active_reg = H_r
		if H_m!=0:
			self.num_active_mem = np.round(self.H_m*active_perc).astype(int)
		else:
			self.num_active_mem = H_m		
	

	def initialize_weights_and_tags(self):	

		if self.prop=='std' or self.prop=='RBP':

			self.V_r = 0.5*np.random.random((self.S,self.R)) - 0.25
			if self.H_r!=0:
				self.W_r = 0.5*np.random.random((self.R,self.H_r)) - 0.25
				self.W_r_back = 0.5*np.random.random((self.H_r,self.R)) - 0.25
				self.W_h_r = 0.5*np.random.random((self.H_r,self.A)) - 0.25
				self.W_h_r_back = 0.5*np.random.random((self.A,self.H_r)) - 0.25
			else:
				self.W_r = 0.5*np.random.random((self.R,self.A)) - 0.25
				self.W_r_back = 0.5*np.random.random((self.A,self.R)) - 0.25			

			self.V_m = 0.5*np.random.random((2*self.S,self.M)) - 0.25
			if self.H_m!=0:
				self.W_m = 0.5*np.random.random((self.M,self.H_m)) - 0.25
				self.W_m_back = 0.5*np.random.random((self.H_m,self.M)) - 0.25
				self.W_h_m = 0.5*np.random.random((self.H_m,self.A)) - 0.25
				self.W_h_m_back = 0.5*np.random.random((self.A,self.H_m)) - 0.25
			else:
				self.W_m = 0.5*np.random.random((self.M,self.A)) - 0.25
				self.W_m_back = 0.5*np.random.random((self.A,self.M)) - 0.25

		elif self.prop=='BP':

			self.V_r = 0.5*np.random.random((self.S,self.R)) - 0.25
			if self.H_r!=0:
				self.W_r = 0.5*np.random.random((self.R,self.H_r)) - 0.25
				self.W_r_back = np.transpose(self.W_r)
				self.W_h_r = 0.5*np.random.random((self.H_r,self.A)) - 0.25
				self.W_h_r_back = np.transpose(self.W_h_r)
			else:
				self.W_r = 0.5*np.random.random((self.R,self.A)) - 0.25
				self.W_r_back = np.transpose(self.W_r)			

			self.V_m = 0.5*np.random.random((2*self.S,self.M)) - 0.25
			if self.H_m!=0:
				self.W_m = 0.5*np.random.random((self.M,self.H_m)) - 0.25
				self.W_m_back = np.transpose(self.W_m)
				self.W_h_m = 0.5*np.random.random((self.H_m,self.A)) - 0.25
				self.W_h_m_back = np.transpose(self.W_h_m)
			else:
				self.W_m = 0.5*np.random.random((self.M,self.A)) - 0.25
				self.W_m_back = np.transpose(self.W_m)

		elif self.prop=='SRBP':
			self.V_r = 0.5*np.random.random((self.S,self.R)) - 0.25
			if self.H_r!=0:
				self.W_r = 0.5*np.random.random((self.R,self.H_r)) - 0.25
				self.W_r_back = 0.5*np.random.random((self.A,self.R)) - 0.25
				self.W_h_r = 0.5*np.random.random((self.H_r,self.A)) - 0.25
				self.W_h_r_back = 0.5*np.random.random((self.A,self.H_r)) - 0.25
			else:
				self.W_r = 0.5*np.random.random((self.R,self.A)) - 0.25
				self.W_r_back = 0.5*np.random.random((self.A,self.R)) - 0.25			

			self.V_m = 0.5*np.random.random((2*self.S,self.M)) - 0.25
			if self.H_m!=0:
				self.W_m = 0.5*np.random.random((self.M,self.H_m)) - 0.25
				self.W_m_back = 0.5*np.random.random((self.A,self.M)) - 0.25
				self.W_h_m = 0.5*np.random.random((self.H_m,self.A)) - 0.25
				self.W_h_m_back = 0.5*np.random.random((self.A,self.H_m)) - 0.25
			else:
				self.W_m = 0.5*np.random.random((self.M,self.A)) - 0.25
				self.W_m_back = 0.5*np.random.random((self.A,self.M)) - 0.25

		elif self.prop=='MRBP':

			self.a = 0.5

			self.V_r = 0.5*np.random.random((self.S,self.R)) - 0.25
			if self.H_r!=0:
				self.W_r = 0.5*np.random.random((self.R,self.H_r)) - 0.25
				self.W_r_back = 0.5*np.random.random((self.H_r,self.R)) - 0.25
				self.W_r_back_skipped = 0.5*np.random.random((self.A,self.R)) - 0.25
				self.W_h_r = 0.5*np.random.random((self.H_r,self.A)) - 0.25
				self.W_h_r_back = 0.5*np.random.random((self.A,self.H_r)) - 0.25
			else:
				self.W_r = 0.5*np.random.random((self.R,self.A)) - 0.25
				self.W_r_back = 0.5*np.random.random((self.A,self.R)) - 0.25			

			self.V_m = 0.5*np.random.random((2*self.S,self.M)) - 0.25
			if self.H_m!=0:
				self.W_m = 0.5*np.random.random((self.M,self.H_m)) - 0.25
				self.W_m_back = 0.5*np.random.random((self.H_m,self.M)) - 0.25
				self.W_m_back_skipped = 0.5*np.random.random((self.A,self.M)) - 0.25
				self.W_h_m = 0.5*np.random.random((self.H_m,self.A)) - 0.25
				self.W_h_m_back = 0.5*np.random.random((self.A,self.H_m)) - 0.25

			else:
				self.W_m = 0.5*np.random.random((self.M,self.A)) - 0.25
				self.W_m_back = 0.5*np.random.random((self.A,self.M)) - 0.25

		self.reset_memory()
		self.reset_tags()

	def reset_memory(self):
		self.cumulative_memory = 1e-6*np.ones((1,self.M))


	def reset_tags(self):
		
		self.sTRACE = 1e-6*np.ones((2*self.S, self.M))

		self.Tag_v_r = 1e-6*np.ones((self.S, self.R))
		if self.H_r!=0:
			self.Tag_w_r = 1e-6*np.ones((self.R, self.H_r))
			self.Tag_w_h_r = 1e-6*np.ones((self.H_r, self.A))
		else:
			self.Tag_w_r = 1e-6*np.ones((self.R, self.A))
		

		self.Tag_v_m = 1e-6*np.ones((2*self.S, self.M))
		if self.H_m!=0:
			self.Tag_w_m = 1e-6*np.ones((self.M, self.H_m))
			self.Tag_w_h_m = 1e-6*np.ones((self.H_m, self.A))
		else:
			self.Tag_w_m = 1e-6*np.ones((self.M, self.A))
		
	def update_weights(self,RPE):

		if self.H_r!=0:
			self.W_h_r += self.beta*RPE*self.Tag_w_h_r
		self.W_r += self.beta*RPE*self.Tag_w_r
		self.V_r += self.beta*RPE*self.Tag_v_r

		if self.H_m!=0:
			self.W_h_m += self.beta*RPE*self.Tag_w_h_m
		self.W_m += self.beta*RPE*self.Tag_w_m
		self.V_m += self.beta*RPE*self.Tag_v_m
		if self.prop!='RBP' and self.prop!='SRBP' and self.prop!='MRBP':
			self.W_r_back += self.beta*RPE*np.transpose(self.Tag_w_r)
			self.W_m_back += self.beta*RPE*np.transpose(self.Tag_w_m)
			if self.H_r!=0:
				self.W_h_r_back += self.beta*RPE*np.transpose(self.Tag_w_h_r)
			if self.H_m!=0:
				self.W_h_m_back += self.beta*RPE*np.transpose(self.Tag_w_h_m)


	def update_tags(self,s_inst,s_trans,y_r,y_m,y_h_r_filtered,y_h_m_filtered,z,resp_ind):

		# synaptic trace for memory units
		self.sTRACE = self.sTRACE*self.memory_leak + np.tile(np.transpose(s_trans), (1,self.M))

		# synaptic tags for memory branch (deep and shallow)
		if self.H_m!=0:
			self.Tag_w_h_m += -self.alpha*self.Tag_w_h_m + np.dot(np.transpose(y_h_m_filtered), z)	
			delta_h_m = self.W_h_m_back[resp_ind,:]
			self.Tag_w_m += -self.alpha*self.Tag_w_m + np.dot(np.transpose(y_m), y_h_m_filtered*(1-y_h_m_filtered)*delta_h_m)
			if self.prop=='BP' or self.prop=='std':		
				delta_m = np.dot(y_h_m_filtered*(1-y_h_m_filtered)*delta_h_m,self.W_m_back)
			elif self.prop=='RBP':
				delta_m = np.dot(delta_h_m,self.W_m_back)
			elif self.prop=='SRBP':
				delta_m = self.W_m_back[resp_ind,:]			
			elif self.prop=='MRBP':
				delta_m = self.a*np.dot(delta_h_m,self.W_m_back) + (1-self.a)*self.W_m_back_skipped[resp_ind,:]
			self.Tag_v_m += -self.alpha*self.Tag_v_m + self.sTRACE*y_m*(1-y_m)*delta_m
		else:
			self.Tag_w_m += -self.alpha*self.Tag_w_m + np.dot(np.transpose(y_m), z)
			delta_m = self.W_m_back[resp_ind,:]
			self.Tag_v_m += -self.alpha*self.Tag_v_m + self.sTRACE*y_m*(1-y_m)*delta_m
		
		# synaptic tags for controller branch (deep and shallow)
		if self.H_r!=0:
			self.Tag_w_h_r += -self.alpha*self.Tag_w_h_r + np.dot(np.transpose(y_h_r_filtered), z)	
			delta_h_r = self.W_h_r_back[resp_ind,:]
			self.Tag_w_r += -self.alpha*self.Tag_w_r + np.dot(np.transpose(y_r), y_h_r_filtered*(1-y_h_r_filtered)*delta_h_r)
			if self.prop=='BP' or self.prop=='std':		
				delta_r = np.dot(y_h_r_filtered*(1-y_h_r_filtered)*delta_h_r,self.W_r_back)
			elif self.prop=='RBP':
				delta_r = np.dot(delta_h_r,self.W_r_back)
			elif self.prop=='SRBP':
				delta_r = self.W_r_back[resp_ind,:]			
			elif self.prop=='MRBP':
				delta_r = self.a*np.dot(delta_h_r,self.W_r_back) + (1-self.a)*self.W_r_back_skipped[resp_ind,:]
			self.Tag_v_r += -self.alpha*self.Tag_v_r + np.dot(np.transpose(s_inst), y_r*(1-y_r)*delta_r)
		else:
			self.Tag_w_r += -self.alpha*self.Tag_w_r + np.dot(np.transpose(y_r), z)	
			delta_r = self.W_r_back[resp_ind,:]
			self.Tag_v_r += -self.alpha*self.Tag_v_r + np.dot(np.transpose(s_inst), y_r*(1-y_r)*delta_r )

	
	def feedforward(self,s_inst,s_trans):

		y_r = act.sigmoid(s_inst, self.V_r)
		if self.H_r!=0:		
			y_h_r = act.sigmoid(y_r,self.W_r)
			if self.perc_active!=1:
				max_ind = np.argsort(-np.abs(y_h_r))[0,:self.num_active_reg]
				y_h_r_filtered = np.zeros(np.shape(y_h_r))
				y_h_r_filtered[0,max_ind] = y_h_r[0,max_ind]
			else:
				y_h_r_filtered = y_h_r
		else:		
			y_h_r_filtered = None

		y_m,self.cumulative_memory = act.sigmoid_acc_leaky(s_trans, self.V_m, self.cumulative_memory,self.memory_leak)
		if self.H_m!=0:		
			y_h_m = act.sigmoid(y_m,self.W_m)
			if self.perc_active!=1:
				max_ind = np.argsort(-np.abs(y_h_m))[0,:self.num_active_mem]
				y_h_m_filtered = np.zeros(np.shape(y_h_m))
				y_h_m_filtered[0,max_ind] = y_h_m[0,max_ind]
			else:
				y_h_m_filtered = y_h_m
		else:		
			y_h_m_filtered = None

		if self.H_r!=0 and self.H_m!=0:
			y_tot = np.concatenate((y_h_r, y_h_m),axis=1)
			W_tot = np.concatenate((self.W_h_r, self.W_h_m),axis=0)
		elif self.H_r==0 and self.H_m!=0:
			y_tot = np.concatenate((y_r, y_h_m),axis=1)
			W_tot = np.concatenate((self.W_r, self.W_h_m),axis=0)
		if self.H_r!=0 and self.H_m==0:
			y_tot = np.concatenate((y_h_r, y_m),axis=1)
			W_tot = np.concatenate((self.W_h_r, self.W_m),axis=0)
		Q = act.linear(y_tot, W_tot)


		return y_r, y_m, Q, y_h_r_filtered, y_h_m_filtered 


######## TRAINING + TEST FOR 12AX TASK

	def training(self,N,S_train,O_train,average_sample,reset_case,conv_criterion='strong',stop=True,verbose=False):

		self.initialize_weights_and_tags()

		N_stimuli = np.shape(S_train)[0]
		zero = np.zeros((1,self.S))

		E = np.zeros((N))
		first = True
		s_old = zero

		correct = 0
		convergence = False
		conv_iter = np.array([0])
		ep_corr = 0
		ep = 0	

		lung_samp = 0
		cont = 0

		if self.prop=='RBP' or self.prop=='SRBP' or self.prop=='MRBP':
			RBP_cond_M = np.zeros((N))
			RBP_cond_R = np.zeros((N))
			if self.H_m!=0:			
				RBP_cond_H_M = np.zeros((N))
			if self.H_r!=0:
				RBP_cond_H_R = np.zeros((N))

		for n in np.arange(N_stimuli):

			s = S_train[n:(n+1),:]
			if self.dic_stim[repr(s.astype(int))] in reset_case:
				if verbose:
					print('RESET \n')
				if ep!=0 and (self.prop=='RBP' or self.prop=='SRBP' or self.prop=='MRBP'):
					RBP_cond_M[ep-1]/=lung_samp
					RBP_cond_R[ep-1]/=lung_samp
					if self.H_m!=0:
						RBP_cond_H_M[ep-1]/=lung_samp
					if self.H_r!=0:
						RBP_cond_H_R[ep-1]/=lung_samp
				ep += 1
				ep_corr += 1
				self.reset_memory()
				self.reset_tags()
				lung_samp = 0	
			elif first==True:
				first = False
			lung_samp += 1	
	
			s_inst = s
			s_trans = self.define_transient(s_inst,s_old)
			s_old = s
			s_print = self.dic_stim[repr(s.astype(int))]			
			o = O_train[n:(n+1),:]
			o_print = self.dic_resp[repr(o.astype(int))]
 
			y_r, y_m, Q, y_h_r_filtered, y_h_m_filtered = self.feedforward(s_inst, s_trans)
			if (np.isnan(Q)).any()==True:
				conv_iter = np.array([-1])	
				break
			resp_ind,P_vec = self.compute_response(Q,n)
			q = Q[0,resp_ind]

			r_print = self.dic_resp[repr(resp_ind)]
	
			z = np.zeros(np.shape(Q))
			z[0,resp_ind] = 1


			if first==False:

				RPE = (r + self.discount*q) - q_old  # Reward Prediction Error
				if self.prop=='RBP' or self.prop=='SRBP' or self.prop=='MRBP':

					if self.H_r!=0:
						mod_BP_H_R = RPE*np.transpose(self.W_h_r[:,resp_ind])
						mod_FA_H_R = RPE*self.W_h_r_back[resp_ind,:]
						norm_H_R = np.linalg.norm(mod_BP_H_R)*np.linalg.norm(mod_FA_H_R)
						RBP_cond_H_R[ep-1] += np.dot(np.transpose(mod_BP_H_R), mod_FA_H_R)/(norm_H_R)

						mod_BP_R = np.dot(mod_BP_H_R*np.squeeze(y_h_r_filtered)*(1-np.squeeze(y_h_r_filtered)), np.transpose(self.W_r))
						if self.prop=='RBP':
							mod_FA_R = np.dot(mod_FA_H_R, self.W_r_back)
						elif self.prop=='SRBP':
							mod_FA_R = RPE*self.W_r_back[resp_ind,:]
						elif self.prop=='MRBP':
							mod_FA_R = self.a*np.dot(mod_FA_H_R, self.W_r_back) + (1-self.a)*RPE*self.W_r_back_skipped[resp_ind,:][resp_ind,:]						
						norm_R = np.linalg.norm(mod_BP_R)*np.linalg.norm(mod_FA_R)
						RBP_cond_R[ep-1] += np.dot(np.transpose(mod_BP_R), mod_FA_R)/(norm_R)
					else:
						mod_BP_R = RPE*np.transpose(self.W_r[:,resp_ind])
						mod_FA_R = RPE*self.W_r_back[resp_ind,:]
						norm_R = np.linalg.norm(mod_BP_R)*np.linalg.norm(mod_FA_R)
						RBP_cond_R[ep-1] += np.dot(np.transpose(mod_BP_R), mod_FA_R)/(norm_R)

					if self.H_m!=0:
						mod_BP_H_M = RPE*np.transpose(self.W_h_m[:,resp_ind])
						mod_FA_H_M = RPE*self.W_h_m_back[resp_ind,:]
						norm_H_M = np.linalg.norm(mod_BP_H_M)*np.linalg.norm(mod_FA_H_M)
						RBP_cond_H_M[ep-1] += np.dot(np.transpose(mod_BP_H_M), mod_FA_H_M)/(norm_H_M)

						mod_BP_M = np.dot(mod_BP_H_M*np.squeeze(y_h_m_filtered)*(1-np.squeeze(y_h_m_filtered)), np.transpose(self.W_m))
						if self.prop=='RBP':
							mod_FA_M = np.dot(mod_FA_H_M, self.W_m_back)
						elif self.prop=='SRBP':
							mod_FA_M = RPE*self.W_m_back[resp_ind,:]
						elif self.prop=='MRBP':
							mod_FA_M = self.a*np.dot(mod_FA_H_M, self.W_m_back) + (1-self.a)*RPE*self.W_m_back_skipped[resp_ind,:]
						norm_M = np.linalg.norm(mod_BP_M)*np.linalg.norm(mod_FA_M)
						RBP_cond_M[ep-1] += np.dot(np.transpose(mod_BP_M), mod_FA_M)/(norm_M)
					else:
						mod_BP_M = RPE*np.transpose(self.W_m[:,resp_ind])
						mod_FA_M = RPE*self.W_m_back[resp_ind,:]
						norm_M = np.linalg.norm(mod_BP_M)*np.linalg.norm(mod_FA_M)
						RBP_cond_M[ep-1] += np.dot(np.transpose(mod_BP_M), mod_FA_M)/(norm_M)					

				self.update_weights(RPE)

			self.update_tags(s_inst,s_trans,y_r,y_m,y_h_r_filtered,y_h_m_filtered,z,resp_ind)

			print('ITER: ',n+1,'\t STIM: ',s_print,'\t OUT: ',o_print,'\t RESP: ', r_print,'\t\t Q: ',Q)
			
			if r_print!=o_print:
				r = self.rew_neg
				E[ep-1] += 1
				correct = 0
				ep_corr=0
			else:
				r = self.rew_pos
				correct += 1		
			q_old = q


			if conv_criterion=='strong':
				if correct==1000 and convergence==False:
					conv_iter = np.array([n]) 
					convergence = True
					self.epsilon = 0
					if stop==True:
						break;
			elif conv_criterion=='lenient':
				if ep_corr==51 and convergence==False:
					conv_iter = np.array([n]) 
					convergence = True					
					if stop==True:
						break;
		if convergence == True:
			print('SIMULATION MET CRITERION AT ITERATION', conv_iter,'\t (',conv_criterion,' criterion)')

		if lung_samp!=0 and (self.prop=='RBP' or self.prop=='SRBP' or self.prop=='MRBP'):
			RBP_cond_M[ep-1]/=lung_samp
			RBP_cond_R[ep-1]/=lung_samp
			if self.H_m!=0:
				RBP_cond_H_M[ep-1]/=lung_samp
			else:
				RBP_cond_H_M = None
			if self.H_r!=0:
				RBP_cond_H_R[ep-1]/=lung_samp
			else:
				RBP_cond_H_R = None

			return E, conv_iter, RBP_cond_R, RBP_cond_M, RBP_cond_H_R, RBP_cond_H_M

		return E,conv_iter


	def test(self,S_test,O_test,reset_case,verbose=False):

		N_stimuli = np.shape(S_test)[0]
		corr = 0
		binary = (self.A==2)
		Feedback_table = np.zeros((2,2))

		RESP_list = list(self.dic_resp.values())
		RESP_list = np.unique(RESP_list)
		RESP_list.sort()

		zero = np.zeros((1,self.S))
		s_old = zero

		self.epsilon = 0

		for n in np.arange(N_stimuli):

			s = S_test[n:(n+1), :]
			if self.dic_stim[repr(s.astype(int))] in reset_case:
				s_old=zero
				self.reset_memory()

			s_inst = s			
			s_trans = self.define_transient(s_inst,s_old)
			o = O_test[n:(n+1), :]
			s_old = s
			s_print = self.dic_stim[repr(s.astype(int))]

			y_r, y_m, Q, y_h_r_filtered, y_h_m_filtered = self.feedforward(s_inst,s_trans)
			resp_ind, P_vec = self.compute_response(Q)

			o_print = self.dic_resp[repr(o.astype(int))]
			r_print = self.dic_resp[repr(resp_ind)]

			if r_print==o_print:
				corr+=1

			if (binary):

				if (o_print==RESP_list[0] and r_print==RESP_list[0]):
					Feedback_table[0,0] += 1
				elif (o_print==RESP_list[0] and r_print==RESP_list[1]):
					Feedback_table[0,1] += 1
				elif (o_print==RESP_list[1] and r_print==RESP_list[0]):
					Feedback_table[1,0] += 1
				elif (o_print==RESP_list[1] and r_print==RESP_list[1]):
					Feedback_table[1,1] += 1

		if binary:
			print("PERFORMANCE TABLE (output vs. response):\n",Feedback_table)
		print("Percentage of correct predictions: ", 100*corr/N_stimuli) 


######## TRAINING + TEST FOR SACCADE TASK

	def training_saccade(self,N_trial,S_train,O_train,reset_case,verbose=False,shape_factor=0.2):

		self.initialize_weights_and_tags()

		N_stimuli = np.shape(S_train)[0]
		zero = np.zeros((1,self.S))
		zerozero = np.concatenate((zero,zero),axis=1)
		s_old = zero

		E_fix = np.zeros(N_trial)
		E_go = np.zeros(N_trial)
		tr = -1

		phase = 'start'
		fix = 0
		delay = 0
		r = None 
		abort = False
		resp = False
	
		cue_trial = None
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

		for n in np.arange(N_stimuli):	
			
			if abort==True and jump!=0:
					jump-=1
					if jump==0:
						abort = False
			else:
				s = S_train[n:(n+1),:]
				o = O_train[n:(n+1),:]
				s_print = self.dic_stim[repr(s.astype(int))]
				o_print = self.dic_resp[repr(o.astype(int))]
				s_inst = s
				s_trans = self.define_transient(s_inst, s_old)
				s_old = s_inst
							
				if s_print=='empty' and phase!='delay':			# empty screen, begin of the trial
					phase = 'start'
					r = None
					tr += 1
					self.reset_memory()
					self.reset_tags()					
					print('TRIAL N.',tr)
				elif  phase=='start' and (s_print=='P' or s_print=='A'):   	# fixation mark appers
					phase = 'fix'	
					num_fix = 0
					attempts = 0
				elif (s_print=='AL' or s_print=='AR' or s_print=='PL' or s_print=='PR'): 	# location cue
					cue_trial = s_print
					phase = 'cue'
				elif (s_print=='P' or s_print=='A') and phase=='cue':	   	# memory delay
					phase = 'delay'
				elif s_print=='empty' and phase=='delay':	 # go = solve task
					phase = 'go'
					attempts = 0
					resp = False

				y_r, y_m, Q, y_h_r_filtered, y_h_m_filtered = self.feedforward(s_inst, s_trans)
				
				resp_ind,P_vec = self.compute_response(Q)
				q = Q[0,resp_ind]
		
				z = np.zeros(np.shape(Q))
				z[0,resp_ind] = 1

				if o_print!='None':			
					r_print = self.dic_resp[repr(resp_ind)]
					print('ITER: ',n+1,'-',phase,'\t STIM: ',s_print,'\t OUT: ',o_print,'\t RESP: ', r_print,'\t\t Q: ',Q)
				else:
					r_print = 'None'
					print('ITER: ',n+1,'-',phase,'\t STIM: ',s_print)
	
				if phase!='start' and r is not None:
					RPE = (r + self.discount*q) - q_old  # Reward Prediction Error
					self.update_weights(RPE)
	
				self.update_tags(s_inst,s_trans,y_r,y_m,y_h_r_filtered,y_h_m_filtered,z,resp_ind)

				if phase=='cue' and phase!='delay':
					r = 0
				if phase=='fix':
					attempts+=1
					if o_print!=r_print:
						if self.rew_neg!=0:
							print('Negative reward!')
						r = self.rew_neg
						num_fix = 0    # no fixation	
					else: 
						num_fix += 1     # fixation
						if num_fix==2:
							if self.rew_pos!=0:
								print('Positive reward!')
							r = shape_factor*self.rew_pos		

				elif phase=='go':
					if r_print!='F':
						resp = True
					if o_print!=r_print:
						if self.rew_neg!=0:
							print('Negative reward!')
						r = self.rew_neg 		
					else: 
						if self.rew_pos!=0:
							print('Positive reward!')
						r = self.rew_pos					

				q_old = q
				
				if phase=='fix' and num_fix<2 and attempts==2:
					while num_fix<2 and attempts<10:
						r,q_old,fix = self.try_again(s_inst,zerozero,o_print,r,q_old,phase)
						attempts += 1
						if fix==False:
							num_fix = 0
						else:
							num_fix += 1
							if num_fix==2:
								if self.rew_pos!=0:
									print('Positive reward!')
								r=shape_factor*self.rew_pos
					if attempts==10:
						E_fix[tr] = 1 	
						E_go[tr] = 1		# automatically also go fails
						if shape_factor!=0:
							print('No fixation. ABORT')
							abort = True
							jump = 4  # four steps to skip before next trial

				if phase=='go' and resp==False:
					attempts = 1
					while resp==False and attempts<8:
						r,q_old,resp = self.try_again(s_inst,zerozero,o_print,r,q_old,phase)
						attempts += 1
						if attempts==8 and resp==False:
							E_go[tr] = 1

				if resp==True:
					if r==self.rew_neg:
						E_go[tr] = 1
					r,q_old,resp = self.try_again(s_inst,zerozero,o_print,r,q_old,phase)
					resp = False
				
				if phase=='go':

					if cue_trial=='PL':
						num_PL += 1
						if r_print == o_print:
							trial_PL[(num_PL-1) % 50] = 1
						else:
							trial_PL[(num_PL-1) % 50] = 0
						prop_PL = np.mean(trial_PL)

					if cue_trial=='PR':
						num_PR += 1
						if r_print == o_print:
							trial_PR[(num_PR-1) % 50] = 1
						else:
							trial_PR[(num_PR-1) % 50] = 0
						prop_PR = np.mean(trial_PR)

					if cue_trial=='AL':
						num_AL += 1
						if r_print == o_print:
							trial_AL[(num_AL-1) % 50] = 1
						else:
							trial_AL[(num_AL-1) % 50] = 0
						prop_AL = np.mean(trial_AL)

					if cue_trial=='AR':
						num_AR += 1
						if r_print == o_print:
							trial_AR[(num_AR-1) % 50] = 1	
						else:
							trial_AR[(num_AR-1) % 50] = 0
						prop_AR = np.mean(trial_AR)


					if convergence==False and prop_PL>=0.9 and prop_PR>=0.9 and prop_AL>=0.9 and prop_AR>=0.9:
						convergence = True
						conv_iter = np.array([tr])
						self.beta = 0
						#gt = 'max'

		return E_fix,E_go,conv_iter

	def try_again(self,s_i,s_t,o_print,r,q_old,phase):

		y_r, y_m, Q, y_h_r_filtered, y_h_m_filtered = self.feedforward(s_i, s_t)
		resp_ind,P_vec = self.compute_response(Q)
		q = Q[0,resp_ind]
		z = np.zeros(np.shape(Q))
		z[0,resp_ind] = 1
		r_print = self.dic_resp[repr(resp_ind)]

		RPE = (r + self.discount*q) - q_old  # Reward Prediction Error
		self.update_weights(RPE)					
		self.update_tags(s_i,s_t,y_r,y_m,y_h_r_filtered,y_h_m_filtered,z,resp_ind)
		
		resp = False
		print('OUT: ',o_print,'\t RESP: ', r_print)

		if phase=='fix':
			if o_print!=r_print:
				if self.rew_neg!=0:
					print('Negative reward!')
				r = self.rew_neg
				resp = False		# no fixation
			else: 	
				resp = True		# fixation
				r = 0	

		if phase=='go':
			if r_print!='F':
				resp = True	
			if o_print!=r_print:
				if self.rew_neg!=0:
					print('Negative reward!')
				r = self.rew_neg	
			else: 
				if self.rew_pos!=0:
					print('Positive reward!')
				r = self.rew_pos	

		q_old = q

		return r,q_old,resp	


	def test_saccade(self,S_test,O_test,reset_case,verbose=False):

		N_stimuli = np.shape(S_test)[0]
		zero = np.zeros((1,self.S))
		zerozero = np.concatenate((zero,zero),axis=1)
		s_old = zero

		phase = 'start'
		N_fix = 0
		corr_fix = 0
		N_go = 0	
		corr_go = 0

		self.epsilon = 0
		
		for n in np.arange(N_stimuli):	
			
			s = S_test[n:(n+1),:]
			o = O_test[n:(n+1),:]
			s_print = self.dic_stim[repr(s.astype(int))]
			o_print = self.dic_resp[repr(o.astype(int))]
			s_inst = s
			s_trans = self.define_transient(s_inst, s_old)
			s_old = s_inst
							
			if s_print=='empty' and phase!='delay':			# empty screen, begin of the trial
				phase = 'start'
				self.reset_memory()
			elif  phase=='start' and (s_print=='P' or s_print=='A'):   	# fixation mark appers
				phase = 'fix'	
			elif (s_print=='AL' or s_print=='AR' or s_print=='PL' or s_print=='PR'): 	# location cue
				phase = 'cue'
			elif (s_print=='P' or s_print=='A') and phase=='cue':	   	# memory delay
				phase = 'delay'
			elif s_print=='empty' and phase=='delay':	 # go = solve task
				phase = 'go'
			y_r, y_m, Q, y_h_r_filtered, y_h_m_filtered = self.feedforward(s_inst, s_trans)
				
			resp_ind,P_vec = self.compute_response(Q)
			q = Q[0,resp_ind]
		
			z = np.zeros(np.shape(Q))
			z[0,resp_ind] = 1

			if o_print!='None':			
				r_print = self.dic_resp[repr(resp_ind)]
			else:
				r_print = 'None'
			if phase=='fix':
				N_fix += 1			
				if o_print==r_print:
					corr_fix+=1		
			if phase=='go':
				N_go += 1
				if o_print==r_print:
					corr_go+=1

		print("Percentage of correct predictions during fix: ", 100*corr_fix/N_fix)		
		print("Percentage of correct predictions during go: ", 100*corr_go/N_go)
	
###################################################################################################################################################################
###################################################################################################################################################################
###################################################################################################################################################################

class hierarchical_AuGMEnT(AuGMEnT):

	## Inputs
	# ------------
	# L: int, number of levels of the hierarchical memory architecture
	# alpha,beta and leak can be scalars, as usual, or lists, to indicate specific dynamics for each level


	def __init__(self,L,S,R,M,A,alpha,beta,discount,eps,gain,leak,rew_rule='RL',dic_stim=None,dic_resp=None,prop='std'):

		self.L = L

		if isinstance(alpha,list)==False:
			self.ALPHA = alpha
		else:
			if len(alpha)==self.L:
				self.ALPHA = np.reshape(alpha,(self.L,1,1))
				alpha = alpha[0]		
			else:			
				print('WARNING: number of decay parameters is not compatible with number of memory levels.')		

		if isinstance(beta,list)==False:
			self.BETA = beta
		else:
			if len(beta)==self.L:
				self.BETA = np.reshape(beta,(self.L,1,1))
				beta = beta[0]		
			else:			
				print('WARNING: number of learning parameters is not compatible with number of memory levels.')

		if isinstance(leak,list)==False:
			self.MEMORY_LEAK = leak
		else:
			if len(leak)==self.L:
				self.MEMORY_LEAK = np.reshape(leak,(self.L,1,1))
				leak = leak[0]		
			else:			
				print('WARNING: number of leak parameters is not compatible with number of memory levels.')

		super(hierarchical_AuGMEnT,self).__init__(S,R,M,A,alpha,beta,discount,eps,gain,leak,rew_rule,dic_stim,dic_resp,prop)


	def initialize_weights_and_tags(self):

		if self.prop=='std' or self.prop=='RBP' or  self.prop=='SRBP' or  self.prop=='MRBP':
			self.V_r = 0.5*np.random.random((self.S, self.R)) - 0.25
			self.W_r = 0.5*np.random.random((self.R, self.A)) - 0.25
			self.W_r_back = 0.5*np.random.random((self.A, self.R)) - 0.25

			self.V_m = 0.5*np.random.random((self.L, 2*self.S, self.M)) - 0.25
			self.W_m = 0.5*np.random.random((self.L, self.M, self.A)) - 0.25
			self.W_m_back = 0.5*np.random.random((self.L, self.A, self.M)) - 0.25

		elif  self.prop=='BP':
			self.V_r = 0.5*np.random.random((self.S, self.R)) - 0.25
			self.W_r = 0.5*np.random.random((self.R, self.A)) - 0.25
			self.W_r_back = np.transpose(self.W_r)

			self.V_m = 0.5*np.random.random((self.L, 2*self.S, self.M)) - 0.25
			self.W_m = 0.5*np.random.random((self.L, self.M, self.A)) - 0.25
			self.W_m_back = np.transpose(self.W_m)

		self.reset_memory()
		self.reset_tags()


	def update_weights(self,RPE):

		self.W_r += self.beta*RPE*self.Tag_w_r
		self.W_r_back += self.beta*RPE*np.transpose(self.Tag_w_r)
		self.V_r += self.beta*RPE*self.Tag_v_r

		self.W_m += self.BETA*RPE*self.Tag_w_m
		self.V_m += self.BETA*RPE*self.Tag_v_m
		
		if self.prop=='std' or self.prop=='BP':
			self.W_r_back += self.beta*RPE*np.transpose(self.Tag_w_r)
			self.W_m_back += self.BETA*RPE*np.transpose(self.Tag_w_m,(0,2,1))


	def reset_memory(self):
		self.cumulative_memory = 1e-6*np.ones((self.L,1,self.M))


	def reset_tags(self):

		self.sTRACE = 1e-6*np.ones((self.L, 2*self.S, self.M))

		self.Tag_v_r = 1e-6*np.ones((self.S, self.R))
		self.Tag_v_m = 1e-6*np.ones((self.L, 2*self.S, self.M))

		self.Tag_w_r = 1e-6*np.ones((self.R, self.A))
		self.Tag_w_m = 1e-6*np.ones((self.L, self.M, self.A))


	def update_tags(self,s_inst,s_trans,y_r,y_m,z,resp_ind):

		# memory branch 
		self.sTRACE += np.tile(np.transpose(s_trans), (1,self.M)) 
		self.Tag_w_m += -self.ALPHA*self.Tag_w_m + np.dot(np.transpose(y_m,(0,2,1)), z)
		self.Tag_v_m += -self.ALPHA*self.Tag_v_m + self.sTRACE*y_m*(1-y_m)*self.W_m_back[:,resp_ind:(resp_ind+1),:]

		# regular branch
		self.Tag_w_r += -self.alpha*self.Tag_w_r + np.dot(np.transpose(y_r), z)
		self.Tag_v_r += -self.alpha*self.Tag_v_r + np.dot(np.transpose(s_inst), y_r*(1-y_r)*self.W_r_back[resp_ind,:] )


	def feedforward(self,s_inst,s_trans):

		y_r = act.sigmoid(s_inst, self.V_r)

		y_m = np.zeros((self.L, 1, self.M))
		Q = act.linear(y_r, self.W_r)
		for l in np.arange(self.L):
			y_m[l,:,:], self.cumulative_memory[l,:,:] = act.sigmoid_acc(s_trans, self.V_m[l,:,:], self.cumulative_memory[l,:,:])
			Q += act.linear(y_m[l,:,:], self.W_m[l,:,:])

		return y_r, y_m, Q

