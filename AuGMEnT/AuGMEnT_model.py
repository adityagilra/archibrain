## CLASS FOR AuGMEnT MODEL
##
## The model and the equations for the implementation are taken from "How Attention
## Can Create Synaptic Tags for the Learning of Working Memories in Sequential Tasks"
## by J. Rombouts, S. Bohte, P. Roeffsema.
##
## AUTHOR: Marco Martinolli
## DATE: 30.03.2017


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

	def __init__(self,S,R,M,A,alpha,beta,discount,eps,g,rew_rule='RL',dic_stim=None,dic_resp=None,init='random'):

		self.S = S
		self.R = R
		self.M = M
		self.A = A

		self.alpha = alpha
		self.beta = beta
		self.discount = discount
		self.epsilon = eps
		self.gain = g

		self.dic_stim = dic_stim
		self.dic_resp = dic_resp

		self.cumulative_memory = np.zeros((1,self.M))

		if init=='zeros':
			self.V_r = np.zeros((self.S,self.R))
			self.V_m = np.zeros((2*self.S,self.M))
			self.W_r = np.zeros((self.R,self.A))
			self.W_m = np.zeros((self.M,self.A))
			self.W_r_back = np.zeros((self.A,self.R))
			self.W_m_back = np.zeros((self.A,self.M))
		elif init=='random':
			self.V_r = 0.5*np.random.random((self.S,self.R)) - 0.25
			self.V_m = 0.5*np.random.random((2*self.S,self.M)) - 0.25
			self.W_r = 0.5*np.random.random((self.R,self.A)) - 0.25
			self.W_m = 0.5*np.random.random((self.M,self.A)) - 0.25
			self.W_r_back = 0.5*np.random.random((self.A,self.R)) - 0.25
			self.W_m_back = 0.5*np.random.random((self.A,self.M)) - 0.25

		self.Tag_v_r = np.zeros((self.S, self.R))
		self.Tag_v_m = np.zeros((2*self.S, self.M))

		self.Tag_w_r = np.zeros((self.R, self.A))
		self.Tag_w_m = np.zeros((self.M, self.A))

		self.sTRACE = np.zeros((2*self.S, self.M))

		if rew_rule =='RL':
			self.rew_pos = 1
			self.rew_neg = 0
		elif rew_rule =='PL':
			self.rew_pos = 0
			self.rew_neg = -1
		elif rew_rule =='SRL':
			self.rew_pos = 1
			self.rew_neg = -1


	def compute_response(self, Qvec):

		resp_ind = None
		P_vec = None

		if np.random.random()<=(1-self.epsilon):

			# greedy choice
			if len(np.where(np.squeeze(Qvec)==np.max(Qvec)) )==1:			
				resp_ind = np.argmax(Qvec)
				#print('Arg_max: ',resp_ind)
			else:
				resp_ind = np.random.choice(np.arange(len(Qvec)),1).item()	# not working for #actions>2!
				#print('Multiple maxima!')
		else:

			# softmax probabilities
			P_vec = np.exp(self.gain*Qvec)
			#print('Prob_vec',P_vec)
			if (np.isnan(P_vec)).any()==True:
				resp_ind = np.argmax(Qvec)
			else:
				P_vec = P_vec/np.sum(P_vec)

				# response selection
				resp_ind = np.random.choice(np.arange(len(np.squeeze(Qvec))),1,p=np.squeeze(P_vec)).item()
		
		return resp_ind, P_vec


	def update_weights(self,r,q,q_old):

		# NEUROMODULATOR SIGNAL
		RPE = (r + self.discount*q) - q_old  # Reward Prediction Error
		#print('RPE = ',RPE)

		# UPDATE weights
		self.W_r += self.beta*RPE*self.Tag_w_r
		self.W_m += self.beta*RPE*self.Tag_w_m

		self.W_r_back += self.beta*RPE*np.transpose(self.Tag_w_r)
		self.W_m_back += self.beta*RPE*np.transpose(self.Tag_w_m)

		self.V_r += self.beta*RPE*self.Tag_v_r
		self.V_m += self.beta*RPE*self.Tag_v_m


	def reset_memory(self):

		self.cumulative_memory = np.zeros((1,self.M))


	def reset_tags(self):

		self.sTRACE = np.zeros((2*self.S, self.M))

		self.Tag_v_r = np.zeros((self.S, self.R))
		self.Tag_v_m = np.zeros((2*self.S, self.M))

		self.Tag_w_r = np.zeros((self.R, self.A))
		self.Tag_w_m = np.zeros((self.M, self.A))


	def update_tags(self,s_inst,s_trans,y_r,y_m,z,resp_ind):

		# synaptic trace for memory units
		self.sTRACE += np.tile(np.transpose(s_trans), (1,self.M))

		# synaptic tags for W (and W')
		self.Tag_w_r += -self.alpha*self.Tag_w_r + np.dot(np.transpose(y_r), z)
		self.Tag_w_m += -self.alpha*self.Tag_w_m + np.dot(np.transpose(y_m), z)

		# synaptic tags for V using feedback propagation and synaptic traces
		self.Tag_v_r += -self.alpha*self.Tag_v_r + np.dot(np.transpose(s_inst), y_r*(1-y_r)*self.W_r_back[resp_ind,:] )
		self.Tag_v_m += -self.alpha*self.Tag_v_m + self.sTRACE*y_m*(1-y_m)*self.W_m_back[resp_ind,:]


	def feedforward(self,s_inst,s_trans):

		y_r = act.sigmoid(s_inst, self.V_r)
		#print('Cum = ',self.cumulative_memory)
		y_m,self.cumulative_memory = act.sigmoid_acc(s_trans, self.V_m, self.cumulative_memory)

		y_tot = np.concatenate((y_r, y_m),axis=1)
		W_tot = np.concatenate((self.W_r, self.W_m),axis=0)
		Q = act.linear(y_tot, W_tot)

		return y_r, y_m, Q


	def define_transient(self, s,s_old):

		s_plus =  np.where(s<=s_old,0,1)
		s_minus = np.where(s_old<=s,0,1)
		s_trans = np.concatenate((s_plus,s_minus),axis=1)

		return s_trans

	def training_ON_OFF(self,S_train,O_train,reset_case,verbose=False):

		N_stimuli = np.shape(S_train)[0]
		zero = np.zeros((1,self.S))

		E = np.zeros((N_stimuli))
		first = True
		s_old = zero
	
		for n in np.arange(N_stimuli):

			# STEP ON
			s = S_train[n:(n+1),:]
			if self.dic_stim[repr(s.astype(int))] in reset_case:
				if verbose:
					print('RESET \n')
				self.reset_memory()
				self.reset_tags()
				first = True
				
			s_inst = s
			s_trans = self.define_transient(s_inst,s_old)
			s_old = s			
			o = O_train[n:(n+1),:]

			y_r,y_m,Q = self.feedforward(s_inst, s_trans)
			resp_ind,P_vec = self.compute_response(Q)
			q = Q[0,resp_ind]
	
			z = np.zeros(np.shape(Q))
			z[0,resp_ind] = 1

			print('ITER: ',n+1,'\t STIM: ',self.dic_stim[repr(s.astype(int))],'\t OUT: ',self.dic_resp[repr(o)],'\t RESP: ', self.dic_resp[repr(resp_ind)],'\t\t Q: ',Q)
			if verbose:

				print(s_inst, s_trans)

				#print('TAG_V_R:\n',self.Tag_v_r)
				#print('V_R : \n', self.V_r)
				#print('Regular Units: ', y_r)
				#print('Activity Tags (1): \n', self.Tag_w_r)
				#print('Activity Weight (2): \n', self.W_r)
				
				#print('TAG_V_M:\n',self.Tag_v_m)
				#print('V_M:\n',self.V_m)
				print('Memory Units: ', y_m)
				#print('Activity Tags (2): \n', self.Tag_w_m)
				#print('Activity Weight (2): \n', self.W_m)

			if first==False:
				self.update_weights(r,q,q_old)
			self.update_tags(s_inst,s_trans,y_r,y_m,z,resp_ind)
			
			if self.dic_resp[repr(resp_ind)]!=self.dic_resp[repr(o)]:
				r = self.rew_neg
				E[n] = 1
			else: 
				r = self.rew_pos			
			q_old = q


			# STEP OFF
			s_inst = zero
			s_trans = self.define_transient(s_inst,s_old)
			o = np.array([[1,0]])
			s_old = s_inst

			y_r, y_m, Q = self.feedforward(s_inst,s_trans)
			resp_ind,P_vec = self.compute_response(Q,gain)
			q = Q[0,resp_ind]
			z = np.zeros(np.shape(Q))
			z[0,resp_ind] = 1

			if verbose:
				print(s_inst, s_trans)

				#print('TAG_V_R:\n',self.Tag_v_r)
				#print('V_R : \n', self.V_r)
				##print('Regular Units: ', y_r)
				#print('Activity Tags (1): \n', self.Tag_w_r)
				#print('Activity Weight (2): \n', self.W_r)
				
				#print('TAG_V_M:\n',self.Tag_v_m)
				#print('V_M:\n',self.V_m)
				print('Memory Units: ', y_m)
				#print('Activity Tags (2): \n', self.Tag_w_m)
				#print('Activity Weight (2): \n', self.W_m)
			
			self.update_weights(r,q,q_old)
			
			self.update_tags(s_inst,s_trans,y_r,y_m,z,resp_ind)

			if self.dic_resp[repr(resp_ind)]!=self.dic_resp[repr(o)]:
				r = self.rew_neg
			else: 
				r = self.rew_pos			
			q_old = q

		return E


	def try_again(self,s_i,s_t,o_print,r,q_old,phase):
		y_r,y_m,Q = self.feedforward(s_i, s_t)
		resp_ind,P_vec = self.compute_response(Q)
		q = Q[0,resp_ind]
		z = np.zeros(np.shape(Q))
		z[0,resp_ind] = 1
		r_print = self.dic_resp[repr(resp_ind)]

		self.update_weights(r,q,q_old)					
		self.update_tags(s_i,s_t,y_r,y_m,z,resp_ind)
		
		resp = False
		print('OUT: ',o_print,'\t RESP: ', r_print)

		if phase=='fix':
			if o_print!=r_print:
				print('Negative reward!')
				r = self.rew_neg
				resp = False		# no fixation
			else: 	
				resp = True		# fixation
				r = 0	

		if phase=='go':
			if o_print!=r_print:
				print('Negative reward!')
				r = self.rew_neg
				if r_print!='F':
					resp = True		
			else: 
				print('Positive reward!')
				r = self.rew_pos	
				resp = True

		return r,q_old,resp	



	def training_saccade(self,S_train,O_train,reset_case,verbose=False):

		N_stimuli = np.shape(S_train)[0]
		zero = np.zeros((1,self.S))
		zerozero = np.concatenate((zero,zero),axis=1)
		s_old = zero

		phase = 'start'
		fix = 0
		delay = 0
		r = None 
		abort = False
		resp = False
	
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
				elif  phase=='start' and (s_print=='P' or s_print=='A'):   	# fixation mark appers
					phase = 'fix'	
					num_fix = 0
					attempts = 0
				elif (s_print=='AL' or s_print=='AR' or s_print=='PL' or s_print=='PR'): 	# location cue
					phase = 'cue'
				elif (s_print=='P' or s_print=='A') and phase=='cue':	   	# memory delay
					phase = 'delay'
				elif s_print=='empty' and phase=='delay':	 # go = solve task
					phase = 'go'
					num_attempts = 1
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
				
				if verbose:

					print(s_inst, s_trans)
					#print('TAG_V_R:\n',self.Tag_v_r)
					#print('V_R : \n', self.V_r)
					#print('Regular Units: ', y_r)
					#print('Activity Tags (1): \n', self.Tag_w_r)
					#print('Activity Weight (2): \n', self.W_r)
					
					#print('TAG_V_M:\n',self.Tag_v_m)
					#print('V_M:\n',self.V_m)
					#print('Memory Units: ', y_m)
					#print('Activity Tags (2): \n', self.Tag_w_m)
					#print('Activity Weight (2): \n', self.W_m)
	
				if phase!='start' and r is not None:
					self.update_weights(r,q,q_old)
	
				self.update_tags(s_inst,s_trans,y_r,y_m,z,resp_ind)

				if phase=='fix':
					attempts+=1
					if o_print!=r_print:
						print('Negative reward!')
						r = self.rew_neg
						num_fix = 0    # no fixation	
					else: 
						num_fix+=1     # fixation
						if num_fix==2:
							print('Positive reward!')
							r = self.rew_pos		

				if phase=='go':
					if o_print!=r_print:
						print('Negative reward!')
						r = self.rew_neg
						if r_print!='F':
							resp = True		
					else: 
						print('Positive reward!')
						r = self.rew_pos	
						resp = True				

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
								print('Positive reward!')
								r=self.rew_pos
					if attempts==10 and r!=self.rew_pos:
						print('No fixation. ABORT')
						abort = True
						jump = 4  

				if phase=='go' and resp==False:
					attempts = 1
					while resp==False and attempts<8:
						r,q_old,resp = self.try_again(s_inst,zerozero,o_print,r,q_old,phase)
						attempts += 1
				if resp==True:
					r,q_old,resp = self.try_again(s_inst,zerozero,o_print,r,q_old,phase)
						
				

			
	def test_ON_OFF(self,S_test,O_test,reset_case,verbose=False):

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

		for n in np.arange(N_stimuli):

			s = S_test[n:(n+1), :]

			if self.dic_stim[repr(s.astype(int))] in reset_case:
				self.reset_memory()
				first = True

			s_inst = s			
			s_trans = self.define_transient(s_inst,s_old)
			o = O_test[n:(n+1), :]
			s_old = s

			y_r, y_m, Q = self.feedforward(s_inst,s_trans)
			resp_ind,P_vec = self.compute_response(Q)

			o_print = self.dic_resp[repr(o)]
			r_print = self.dic_resp[repr(resp_ind)]

			if r_print==o_print:
					corr+=1
			else:
				print('TEST SAMPLE N.',n+1,'\t',self.dic_stim[repr(s.astype(int))],'\t',o_print,'\t',r_print,'\tQ: ',Q,'\tP:',P_vec,'\n')

			if (binary):

				if (verbose):
					print('TEST SAMPLE N.',n+1,'\t',self.dic_stim[repr(s.astype(int))],'\t',o_print,'\t',r_print,'\n')

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
