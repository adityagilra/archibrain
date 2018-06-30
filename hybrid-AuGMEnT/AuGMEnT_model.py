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
	# leak: scalar or list, leak of the memory dynamics (if a list, the memory units are divided in groups of same size with different leak rates)
	
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

		if isinstance(self.memory_leak, list):
			if np.remainder(self.M,len(self.memory_leak))!=0:
				raise ValueError('Size of the leak vector is not compliant to the number of memory units.')

		self.dic_stim = dic_stim
		self.dic_resp = dic_resp
	
		self.prop = prop

		self.define_reward_rule(rew_rule)

		self.initialize_weights_and_tags()


	def initialize_weights_and_tags(self):

		rang = 1

		if self.prop=='std' or self.prop=='RBP' or self.prop=='SRBP' or self.prop=='MRBP':
			
			self.V_r = rang*np.random.random((self.S,self.R)) - rang/2
			self.W_r = rang*np.random.random((self.R,self.A)) - rang/2

			self.V_m = rang*np.random.random((2*self.S,self.M)) - rang/2
			self.W_m = rang*np.random.random((self.M,self.A)) - rang/2

			self.W_r_back = rang*np.random.random((self.A,self.R)) - rang/2
			self.W_m_back = rang*np.random.random((self.A,self.M)) - rang/2

		elif self.prop=='BP':

			self.V_r = rang*np.random.random((self.S,self.R)) - rang/2
			self.W_r = rang*np.random.random((self.R,self.A)) - rang/2

			self.V_m = rang*np.random.random((2*self.S,self.M)) - rang/2
			self.W_m = rang*np.random.random((self.M,self.A)) - rang/2
			
			self.W_r_back = np.transpose(self.W_r)
			self.W_m_back = np.transpose(self.W_m)

		self.reset_memory()
		self.reset_tags()

	def define_reward_rule(self,rew_rule, seq_pred=False):

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


	def positive_reward_seq_pred(self,final,d=3):
		
		if final==True:
			return 1.5
		else:
			#return (d-1.5)/(d-1)
			return 0.75

	def compute_response(self, Qvec, policy='eps_greedy',stoc='unif',t_weighted=False,e_weighted=False,it=None):

		resp_ind = None
		P_vec = None
		
		if policy=='eps_greedy':
		
			if e_weighted==True and it is not None:
				eps = self.epsilon*(1-(2/np.pi)*np.arctan(it/2000))
			else:
				eps = self.epsilon
			# greedy
			if np.random.random()<=(1-eps):
				if len(np.where(np.squeeze(Qvec)==np.max(Qvec)) )==1:
					resp_ind = np.argmax(Qvec)
				else:
					resp_ind = np.random.choice(np.arange(len(Qvec)),1).item()
			else:
				if stoc=='soft':
					if t_weighted==True and it is not None:
						g = 1 + (10/np.pi)*np.arctan(it/2000)
					else:
						g = 1
					tot = np.clip(a=g*Qvec,a_min=-500,a_max=500)
					P_vec = np.exp(tot)
					if (np.isnan(P_vec)).any()==True:
						resp_ind = np.argmax(Qvec)
					else:
						P_vec = P_vec/np.sum(P_vec)
						resp_ind = np.random.choice(np.arange(len(np.squeeze(Qvec))),1,p=np.squeeze(P_vec)).item()
				elif stoc=='unif':
					resp_ind = np.random.choice(np.arange(len(np.squeeze(Qvec))),1).item()
		
		elif policy=='greedy':
			#print('GREEDY')
			if len(np.where(np.squeeze(Qvec)==np.max(Qvec)) )==1:
				resp_ind = np.argmax(Qvec)
			else:
				resp_ind = np.random.choice(np.arange(len(Qvec)),1).item()
		        
		elif policy=='softmax':
			#print('SOFTMAX')
			if t_weighted==True and it is not None:
				g = 1 + (10/np.pi)*np.arctan(it/2000)
			else:
				g = 1
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

		if isinstance(self.memory_leak, list):
			num_groups=int(self.M/len(self.memory_leak))
			leak_vec = np.repeat(self.memory_leak, num_groups)
			self.sTRACE = leak_vec*self.sTRACE + np.tile(np.transpose(s_trans), (1,self.M))
		else:
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

	
######### TRAINING FOR SEQUENCE PREDICTION TASK (prediction of final letter only)

	def training_seq_pred(self,N_trial,d,stop=True,verb=True,policy='eps_greedy',stoc='soft',t_weighted=True):
	
		from TASKS.task_seq_prediction import construct_trial, get_dictionary

		zero = np.zeros((1,self.S))
		
		corr = 0
		conv_tr = 0
		E = np.zeros(N_trial)
		dic_stim, dic_resp = get_dictionary(d)
		convergence=False

		for tr in np.arange(N_trial):

			if verb:
				print('TRIAL N.',tr+1,':\t', end="")

			self.reset_memory()
			self.reset_tags()

			S, O = construct_trial(d,0.5)
			o_print = dic_resp[np.argmax(O)]
			s_old = zero
			
			for i in np.arange(np.shape(S)[0]):
				 
				s_inst = S[i:(i+1),:]
				s_trans = self.define_transient(s_inst, s_old)
				s_old = s_inst
				s_print =  dic_stim[np.argmax(s_inst)]

				y_r,y_m,Q = self.feedforward(s_inst, s_trans)
				
				resp_ind,P_vec = self.compute_response(Q,policy,stoc,t_weighted)
				q = Q[0,resp_ind]
		
				z = np.zeros(np.shape(Q))
				z[0,resp_ind] = 1
				r_print = self.dic_resp[resp_ind]

				if verb:
					 print(s_print,end="-")
				
				self.update_tags(s_inst,s_trans,y_r,y_m,z,resp_ind)
				q_old = q
				
				if i==np.shape(S)[0]-1:
					if dic_resp[resp_ind]==o_print:
						r = self.rew_pos
						corr += 1
					else:
						r = self.rew_neg
						E[tr] = 1
						corr = 0
				else:
					r = 0
			
			RPE = r - q_old  # Reward Prediction Error
			self.update_weights(RPE)
			
			if verb:
				print(dic_resp[resp_ind],'\t(',corr,')')

			if corr>=100 and convergence==False:
				conv_tr = tr
				convergence=True
				if stop==True:
					break


		if conv_tr!=0:
			print('SIMULATION CONVERGED AT TRIAL ',conv_tr)		
		
		return E, conv_tr

	def test_seq_pred(self,N_test,d,verb=True):
	
		from TASKS.task_seq_prediction import construct_test_trial, get_dictionary

		zero = np.zeros((1,self.S))
		
		corr = 0.0
		dic_stim, dic_resp = get_dictionary(d)

		for tr in np.arange(N_test):

			if verb:
				print('TRIAL N.',tr+1,':\t', end="")

			self.reset_memory()
			self.reset_tags()

			S, O = construct_test_trial(d,0.5)
			o_print = dic_resp[np.argmax(O)]
			s_old = zero
			
			self.epsilon = 0
			self.beta = 0
			
			for i in np.arange(np.shape(S)[0]):
				 
				s_inst = S[i:(i+1),:]
				s_trans = self.define_transient(s_inst, s_old)
				s_old = s_inst
				s_print = dic_stim[np.argmax(s_inst)]

				y_r, y_m, Q = self.feedforward(s_inst, s_trans)
				
				resp_ind,_ = self.compute_response(Q,'greedy')
				q = Q[0,resp_ind]
		
				z = np.zeros(np.shape(Q))
				z[0,resp_ind] = 1
				r_print = self.dic_resp[resp_ind]

				if verb:
					 print(s_print,end="-")
				
				q_old = q
				
				if i==np.shape(S)[0]-1:
				    if verb:
				        print(dic_resp[resp_ind],'\n')
				    if dic_resp[resp_ind]==o_print:
				    	corr += 1
		    	
		perc = 100*float(corr/N_test)
        
		return perc

	# training variant for given training datasets
	def training_seq_pred_2(self,S_train,O_train,dic_stim, dic_resp, verb=True):

		zero = np.zeros((1,self.S))
		
		corr = 0
		conv_tr = 0
		convergence=False

		N_trial = np.shape(S_train)[0]
		L = np.shape(S_train)[1]

		Q_matrix = np.zeros((N_trial*L,self.A))
		resp_vec = np.zeros((N_trial*L))
		RPE_vec = np.zeros(N_trial)		

		for tr in np.arange(N_trial):

			if verb:
				print('TRIAL N.',tr+1,':\t', end="")

			self.reset_memory()
			self.reset_tags()

			S = S_train[tr, :, :]
			O = O_train[tr, :]

			o_print = dic_resp[np.argmax(O)]
			s_old = zero
			
			for i in np.arange(L):
				 
				s_inst = S[i:(i+1),:]
				s_trans = self.define_transient(s_inst, s_old)
				s_old = s_inst
				s_print =  dic_stim[np.argmax(s_inst)]

				y_r,y_m,Q = self.feedforward(s_inst, s_trans)
				Q_matrix[tr*L+i,:] = Q
				
				resp_ind,P_vec = self.compute_response(Q,policy,stoc,t_weighted,e_weighted)
				q = Q[0,resp_ind]
				resp_vec[tr*L+i] = resp_ind
				
				z = np.zeros(np.shape(Q))
				z[0,resp_ind] = 1
				r_print = self.dic_resp[resp_ind]

				if verb:
					 print(s_print,end="-")
				
				self.update_tags(s_inst,s_trans,y_r,y_m,z,resp_ind)
				q_old = q
				
				if i==np.shape(S)[0]-1:
					if dic_resp[resp_ind]==o_print:
						r = self.rew_pos
						corr += 1
					else:
						r = self.rew_neg
						corr = 0
				else:
					r = 0
			
			RPE = r - q_old  # Reward Prediction Error
			RPE_vec[tr] = RPE			
			self.update_weights(RPE)
			
			if verb:
				print(dic_resp[resp_ind],'\t(',corr,')')

			if corr>=100 and convergence==False:
				conv_tr = tr
				convergence=True

		if conv_tr!=0:
			print('SIMULATION CONVERGED AT TRIAL ',conv_tr)		
		
		return conv_tr, Q_matrix, RPE_vec, resp_vec


######### TRAINING FOR SEQUENCE PREDICTION TASK (CPT variant=response to each cue)

	def training_seq_pred_CPT(self,N_trial,d,stop=True,verb=True):
	
		from TASKS.task_seq_prediction_CPT import construct_trial, get_dictionary

		zero = np.zeros((1,self.S))
		
		corr = 0
		conv_tr = 0
		E = np.zeros(N_trial)
		dic_stim, dic_resp = get_dictionary(d)
		convergence=False

		for tr in np.arange(N_trial):

			if verb:
				print('TRIAL N.',tr+1,':\t', end="")

			first = True

			self.reset_memory()
			self.reset_tags()

			S, O = construct_trial(d,0.5)
			s_old = zero
			
			for i in np.arange(np.shape(S)[0]):
				 
				s_inst = S[i:(i+1),:]
				s_trans = self.define_transient(s_inst, s_old)
				s_old = s_inst
				s_print =  dic_stim[np.argmax(s_inst)]

				#print(self.sTRACE)

				o = O[i:(i+1),:]
				o_print = dic_resp[np.argmax(o)]

				y_r,y_m,Q = self.feedforward(s_inst, s_trans)
				
				resp_ind,P_vec = self.compute_response(Q,policy,stoc,t_weighted,e_weighted)
				q = Q[0,resp_ind]
				z = np.zeros(np.shape(Q))
				z[0,resp_ind] = 1
				r_print = dic_resp[resp_ind]

				if verb:
					if i==0:
						print(s_print,end="-")
						print(r_print,end="-")
					elif i==np.shape(S)[0]-1:
						print(r_print,'\t(',corr,')')
					else:
						print(r_print,end="-")
	
				if first!=True:
					RPE = (r + self.discount*q) - q_old  # Reward Prediction Error
					self.update_weights(RPE)
				else:
					first = False
				
				self.update_tags(s_inst,s_trans,y_r,y_m,z,resp_ind)
				q_old = q
				
				if r_print==o_print:
					final = (o_print==dic_resp[self.A-2] or o_print==dic_resp[self.A-1])
					r = self.positive_reward_seq_pred(final,self.A-2)
					corr += 1
				else:
					r = self.rew_neg
					E[tr] = 1
					corr = 0

				#print('\t',s_print," - ",r_print,' (',o_print,')','--->',r)
			
			RPE = r - q_old  # Reward Prediction Error
			self.update_weights(RPE)

			if corr>=100 and convergence==False:
				conv_tr = tr
				convergence=True
				if stop==True:
					break
		if conv_tr!=0:
			print('SIMULATION CONVERGED AT TRIAL ',conv_tr)		
		
		return E, conv_tr


######### TRAINING + TEST FOR SACCADE TASK 

	def training_saccade(self,N_trial,S_train,O_train,reset_case,policy='eps_greedy',stoc='unif',t_weighted=False,e_weighted=False,verbose=False,shape_factor=0.2,stop=True):

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
					self.reset_memory()
					self.reset_tags()
					phase = 'start'
					r = None
					tr += 1
					print('TRIAL: ', tr)

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
				
				resp_ind,P_vec = self.compute_response(Q,policy,stoc,t_weighted,e_weighted)
				q = Q[0,resp_ind]
		
				z = np.zeros(np.shape(Q))
				z[0,resp_ind] = 1
				r_print = self.dic_resp[str(resp_ind)]
				print('\t',s_print,'--->',r_print)

				if phase!='start' and r is not None:
					RPE = (r + self.discount*q) - q_old  # Reward Prediction Error
					self.update_weights(RPE)
	
				self.update_tags(s_inst,s_trans,y_r,y_m,z,resp_ind)

				if phase=='cue' and phase!='delay':
					r = 0
				if phase=='fix':
					attempts+=1
					if o_print!=r_print:
						r = self.rew_neg
						num_fix = 0    # no fixation	
					else: 
						num_fix += 1     # fixation
						if num_fix==2:
							r = shape_factor*self.rew_pos		

				elif phase=='go':
					if r_print!='F':
						resp = True
					if o_print!=r_print:
						r = self.rew_neg 		
					else: 
						r = 1.5*self.rew_pos					

				q_old = q
				
				if phase=='fix' and num_fix<2 and attempts==2:
					while num_fix<2 and attempts<10:
						r,q_old,fix = self.try_again(s_inst,zerozero,o_print,r,q_old,phase,policy,stoc,t_weighted)
						attempts += 1
						if fix==False:
							num_fix = 0
						else:
							num_fix += 1
							if num_fix==2:
								r=shape_factor*self.rew_pos
					if attempts==10:
						E_fix[tr] = 1 	
						E_go[tr] = 1		# automatically also go fails
						if shape_factor!=0:
							abort = True
							jump = 4  # four steps to skip before next trial

				if phase=='go' and resp==False:
					attempts = 1
					while resp==False and attempts<8:
						r,q_old,resp = self.try_again(s_inst,zerozero,o_print,r,q_old,phase,policy,stoc,t_weighted)
						attempts += 1
						if attempts==8 and resp==False:
							E_go[tr] = 1

				if resp==True:
					if r==self.rew_neg:
						E_go[tr] = 1
					RPE = r - q_old
					self.update_weights(RPE)
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
			
					if np.remainder(tr,1000)==0 and tr!=0:
						print('TRIAL ',tr,'\t PL:',prop_PL,'  PR:',prop_PR,'  AL:',prop_AL,'  AR:',prop_AR)

					if convergence==False and prop_PL>=0.9 and prop_PR>=0.9 and prop_AL>=0.9 and prop_AR>=0.9:
						convergence = True
						conv_iter = np.array([tr])
						#gt = 'max'
						if stop:
							break

		return E_fix,E_go,conv_iter

	def try_again(self,s_i,s_t,o_print,r,q_old,phase,policy='eps_greedy',stoc='unif',t_weighted=False,e_weighted=False):

		y_r,y_m,Q = self.feedforward(s_i, s_t)
		resp_ind,P_vec = self.compute_response(Q,policy,stoc,t_weighted,e_weighted)
		q = Q[0,resp_ind]
		z = np.zeros(np.shape(Q))
		z[0,resp_ind] = 1
		r_print = self.dic_resp[repr(resp_ind)]

		RPE = (r + self.discount*q) - q_old  # Reward Prediction Error
		self.update_weights(RPE)					
		self.update_tags(s_i,s_t,y_r,y_m,z,resp_ind)
		
		resp = False

		if phase=='fix':
			if o_print!=r_print:
				r = self.rew_neg
				resp = False		# no fixation
			else: 	
				resp = True		# fixation
				r = 0	

		if phase=='go':
			if r_print!='F':
				resp = True	
			if o_print!=r_print:
				r = self.rew_neg	
			else: 
				r = 1.5*self.rew_pos	

		q_old = q

		return r,q_old,resp	

	def test_saccade(self,S_test,O_test,reset_case,policy='eps_greedy',stoc='unif',t_weighted=False,e_weighted=False):

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
				
			resp_ind,P_vec = self.compute_response(Q,policy,stoc,t_weighted,e_weighted)
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

		perc_fix = 100*float(corr_fix)/float(N_fix)
		perc_go = 100*float(corr_go)/float(N_go)
		
		return perc_fix, perc_go


######### TRAINING + TEST FOR 12AX TASK 

	def training_12AX(self,N_trial,p_target,criterion='strong',stop=True,verb=False,policy='eps_greedy',stoc='unif',t_weighted=False,e_weighted=False):

		
		from TASKS.task_12AX import data_construction

		zero = np.zeros((1,self.S))
		s_old = zero

		E = np.zeros(N_trial)
		R = np.zeros(N_trial)

		convergence = False
		conv_iter = np.array([0])
		corr = 0
		corr_ep = 0
		corr_ep_bool = True

		for tr in np.arange(N_trial):

			if verb:
				print('TRIAL ', tr+1)

			S_train, O_train = data_construction(1,p_target)

			N_stimuli = np.shape(S_train)[0]

			self.reset_memory()
			self.reset_tags()
		
			first = True

			for n in np.arange(N_stimuli):	

				s = S_train[n:(n+1),:]
				o = O_train[n:(n+1),:]
				s_print = self.dic_stim[repr(s.astype(int))]
				o_print = self.dic_resp[repr(o.astype(int))]
				s_inst = s
				s_trans = self.define_transient(s_inst, s_old)
				s_old = s_inst

				y_r,y_m,Q = self.feedforward(s_inst, s_trans)
				
				resp_ind,_ = self.compute_response(Q,policy,stoc,t_weighted,e_weighted,tr)
				q = Q[0,resp_ind]
		
				z = np.zeros(np.shape(Q))
				z[0,resp_ind] = 1
				
				r_print = self.dic_resp[str(resp_ind)]
				
				if first==False:
					RPE = (r + self.discount*q) - q_old  # Reward Prediction Error
					self.update_weights(RPE)
				else:
					first=False
	
				self.update_tags(s_inst,s_trans,y_r,y_m,z,resp_ind)	

				if r_print==o_print:
					r = self.rew_pos
					corr += 1 		
				else: 
					r = self.rew_neg
					corr = 0
					corr_ep_bool = False
					E[tr] = 1					
				R[tr]+=r
				q_old = q
				
				if verb:
					print('\t s:',s_print,'\t o:', o_print,'\t r:', r_print, '\t\t corr_acc:',corr)
			
			RPE = r - q_old
			self.update_weights(RPE)
	
			if corr_ep_bool==True:
				corr_ep += 1 
				
			if convergence==False and criterion=='strong' and corr>=1000:
				convergence = True
				conv_iter = np.array([tr])
				if stop:
					break

			if convergence==False and criterion=='lenient' and corr_ep>=50:
				convergence = True
				conv_iter = np.array([tr])
				if stop:
					break

		return E,conv_iter,R


	# training variant for given training datasets
	def training_12AX_2(self,S_train,O_train,policy='eps_greedy',stoc='unif',t_weighted=False,e_weighted=False):

		zero = np.zeros((1,self.S))
		first = True
		s_old = zero	

		convergence = False
		conv_iter = np.array([0])
		corr = 0

		N_samples = np.shape(S_train)[0]

		Q_matrix = np.zeros((N_samples,self.A))
		RPE_vec =  np.zeros((N_samples-1))

		tr = 1
		#print('TRIAL N.',tr)

		for n in np.arange(N_samples):

			s = S_train[n:(n+1),:]
			o = O_train[n:(n+1),:]
			s_print = self.dic_stim[repr(s.astype(int))]
			o_print = self.dic_resp[repr(o.astype(int))]

			if (s_print=='1' or s_print=='2') and n!=0:

				
				RPE = r - q_old
				RPE_vec[n-1] = RPE
				self.update_weights(RPE)

				self.reset_memory()
				self.reset_tags()
				
				first = True
				s_old = zero	
				tr += 1	
				if np.remainder(tr,10000)==0:				
					print('TRIAL N.',tr)		

			s_inst = s
			s_trans = self.define_transient(s_inst, s_old)
			s_old = s_inst

			y_r,y_m,Q = self.feedforward(s_inst, s_trans)
			Q_matrix[n,:] = Q
				
			resp_ind,_ = self.compute_response(Q,policy,stoc,t_weighted,e_weighted,tr)
			q = Q[0,resp_ind]
			
			z = np.zeros(np.shape(Q))
			z[0,resp_ind] = 1
			r_print = self.dic_resp[str(resp_ind)]
	
			if first==False:
				RPE = (r + self.discount*q) - q_old  # Reward Prediction Error
				RPE_vec[n-1] = RPE
				self.update_weights(RPE)
			else:
				first=False
	
			self.update_tags(s_inst,s_trans,y_r,y_m,z,resp_ind)

			if r_print==o_print:
				r = self.rew_pos
				corr += 1 		
			else: 
				r = self.rew_neg
				corr = 0
		
			#print('\t S: ',s_print,' ---> R: ',r_print,' ---> RPE: ',RPE)	
			#print('\t S: ',s_print,' ---> R: ',r_print,'  Q:',Q,' (',corr,')')					

			q_old = q

			if convergence==False and corr>=1000:
				convergence = True
				conv_iter = np.array([tr])

		return conv_iter, Q_matrix, RPE_vec
		
		
    # training variant for given training datasets
	def training_12AX_3(self,S_train,O_train,policy='eps_greedy',stoc='unif',t_weighted=False,e_weighted=False):

		zero = np.zeros((1,self.S))
		first = True
		s_old = zero	

		convergence = False
		conv_iter = np.array([0])
		corr = 0

		N_samples = np.shape(S_train)[0]

		E = np.zeros((N_samples))
		
		num_checks=5
		check_i=-1
		H_m = np.zeros((num_checks, self.M))

		tr = 1
		#print('TRIAL N.',tr)

		for n in np.arange(N_samples):

			s = S_train[n:(n+1),:]
			o = O_train[n:(n+1),:]
			s_print = self.dic_stim[repr(s.astype(int))]
			o_print = self.dic_resp[repr(o.astype(int))]

			if (s_print=='1' or s_print=='2') and n!=0:
				
				RPE = r - q_old
				self.update_weights(RPE)

				self.reset_memory()
				self.reset_tags()
				
				first = True
				s_old = zero	
				tr += 1
				
				if (tr==10 or tr==100 or tr==1000 or tr==10000 or tr==50000):
				    done=False
					

			s_inst = s
			s_trans = self.define_transient(s_inst, s_old)
			s_old = s_inst

			y_r,y_m,Q = self.feedforward(s_inst, s_trans)
			
			if (tr==10 or tr==100 or tr==1000 or tr==10000 or tr==50000) and done==False:
			    check_i += 1
			    print(tr,'\t',check_i)
			    H_m[check_i,:] = y_m
			    done=True
				
			resp_ind,_ = self.compute_response(Q,policy,stoc,t_weighted,e_weighted,tr)
			q = Q[0,resp_ind]
			
			z = np.zeros(np.shape(Q))
			z[0,resp_ind] = 1
			r_print = self.dic_resp[str(resp_ind)]
	
			if first==False:
				RPE = (r + self.discount*q) - q_old  # Reward Prediction Error
				self.update_weights(RPE)
			else:
				first=False
	
			self.update_tags(s_inst,s_trans,y_r,y_m,z,resp_ind)

			if r_print==o_print:
				r = self.rew_pos
				corr += 1 		
			else: 
				r = self.rew_neg
				corr = 0
				E[n] = 1	

			q_old = q

			if convergence==False and corr>=1000:
				convergence = True
				conv_iter = np.array([tr])

		return E, conv_iter, H_m
				

	def test(self,N_test,p_target):

		from TASKS.task_12AX import data_construction

		zero = np.zeros((1,self.S))
		s_old = zero
	
		corr_ep_expl = 0
		corr_ep_no_expl = 0
		corr_ep_soft = 0

		#self.epsilon = 0

		for tr in np.arange(N_test):

			S_test, O_test = data_construction(1,p_target)

			N_stimuli = np.shape(S_test)[0]

			self.reset_memory()
	
			corr_ep_bool_expl = True
			corr_ep_bool_no_expl = True
			corr_ep_bool_soft = True

			for n in np.arange(N_stimuli):		
			
				s = S_test[n:(n+1),:]
				o = O_test[n:(n+1),:]
				s_print = self.dic_stim[repr(s.astype(int))]
				o_print = self.dic_resp[repr(o.astype(int))]
				s_inst = s
				s_trans = self.define_transient(s_inst, s_old)
				s_old = s_inst
							
				y_r,y_m,Q = self.feedforward(s_inst, s_trans)
				
				resp_ind_expl,P_vec_e = self.compute_response(Q,'eps_greedy','soft','False','False')
				resp_ind_no_expl,_ = self.compute_response(Q,'greedy')
				resp_ind_soft,P_vec_s = self.compute_response(Q,'softmax')
				
				#print('P_vec (eps): ',P_vec_e,'\t P_vec (soft):',P_vec_s)	
				
				r_print_expl = self.dic_resp[repr(resp_ind_expl)]
				r_print_no_expl = self.dic_resp[repr(resp_ind_no_expl)]
				r_print_soft = self.dic_resp[repr(resp_ind_soft)]
		
				if o_print!=r_print_expl:
					corr_ep_bool_expl = False
				if o_print!=r_print_no_expl:
					corr_ep_bool_no_expl = False
				if o_print!=r_print_soft:
					corr_ep_bool_soft = False					
					
			if corr_ep_bool_expl:
				corr_ep_expl += 1
			if corr_ep_bool_no_expl:
				corr_ep_no_expl += 1
			if corr_ep_bool_soft:
				corr_ep_soft += 1					

		perc_expl = 100*float(corr_ep_expl)/float(N_test)
		perc_no_expl = 100*float(corr_ep_no_expl)/float(N_test)
		perc_soft = 100*float(corr_ep_soft)/float(N_test)
		
		return perc_expl, perc_no_expl, perc_soft


####### TRAINING FOR TEMPORAL XOR TASK

	def training_tXOR(self,N_trial,stop=True,verb=True,policy='eps_greedy',stoc='unif',t_weighted=False):
	
		from TASKS.task_tXOR import construct_trial

		zero = np.zeros((1,self.S))
		
		corr = 0
		conv_tr = 0
		E = np.zeros(N_trial)
		convergence=False

		for tr in np.arange(N_trial):

			if verb:
				print('TRIAL N.',tr+1,':\t', end="")

			self.reset_memory()
			self.reset_tags()

			S, O = construct_trial()
			o_print = self.dic_resp[np.argmax(O)]
			s_old = zero
			
			for i in np.arange(3):
				 
				s_inst = S[i:(i+1),:]
				s_trans = self.define_transient(s_inst, s_old)
				s_old = s_inst
				if i!=2:
					s_print = self.dic_stim[np.argmax(s_inst)]

				y_r,y_m,Q = self.feedforward(s_inst, s_trans)
				
				resp_ind,_ = self.compute_response(Q,policy,stoc,t_weighted)
				q = Q[0,resp_ind]
		
				z = np.zeros(np.shape(Q))
				z[0,resp_ind] = 1
				r_print = self.dic_resp[resp_ind]

				if verb and i!=2:
					 print(s_print,end="-")
				
				self.update_tags(s_inst,s_trans,y_r,y_m,z,resp_ind)
				q_old = q
				
				if i==2:
					if self.dic_resp[resp_ind]==o_print:
						r = self.rew_pos
						corr += 1
					else:
						r = self.rew_neg
						E[tr] = 1
						corr = 0
				else:
					r = 0
			
			RPE = r - q_old  # Reward Prediction Error
			self.update_weights(RPE)
			
			if verb:
				print(self.dic_resp[resp_ind],'\t(',corr,')')

			if corr>=100 and convergence==False:
				conv_tr = tr
				convergence=True
				if stop==True:
					break

		if conv_tr!=0:
			print('SIMULATION CONVERGED AT TRIAL ',conv_tr)		
		
		return E, conv_tr

