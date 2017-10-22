## MAIN FILE FOR deep AuGMEnT TESTING
## Here are defined the settings for the AuGMEnT architecture and for the tasks to test.
## AUTHOR: Marco Martinolli
## DATE: 10.08.2017

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec
import pylab
import gc 

np.set_printoptions(precision=3)
import activations as act
from task_12AX import construct_trial

###################################################################################################################################################################
###################################################################################################################################################################
###################################################################################################################################################################

class deep_AuGMEnT():

	## Inputs
	# ------------
	# H_r: int, number of units for the hidden layer of the controller
	# H_m: int, number of units for the hidden layer of the memory branch

	def __init__(self,S,R,M,H_r,H_m,A,alpha,beta,discount,eps,gain,leak,rew_rule='RL',dic_stim=None,dic_resp=None,prop='std'):

		self.S = S
		self.R = R
		self.M = M
		self.A = A
		self.H_r = H_r
		self.H_m = H_m

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
	
		self.a = 0.8
		self.it_ref = 2000

		self.initialize_weights_and_tags()	

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
			if it is not None:
				g = self.gain*(1+ (8/np.pi)*np.arctan(it/self.it_ref))
			else:
				g = self.gain
			tot = g*Qvec
			tot -= np.max(tot)
			
			P_vec = np.exp(tot)
			if (np.isnan(P_vec)).any()==True:
				resp_ind = np.argmax(Qvec)
			else:
				P_vec = P_vec/np.sum(P_vec)
				resp_ind = np.random.choice(np.arange(len(np.squeeze(Qvec))),1,p=np.squeeze(P_vec)).item()
		
		return resp_ind, P_vec
		

	def initialize_weights_and_tags(self):

		range = 1

		self.V_r = range*np.random.random((self.S,self.R)) - range/2
		if self.H_r!=0:
			self.W_r = range*np.random.random((self.R,self.H_r)) - range/2
			self.W_h_r = range*np.random.random((self.H_r,self.A)) - range/2
		else:
			self.W_r = range*np.random.random((self.R,self.A)) - range/2	
			
		self.V_m = range*np.random.random((2*self.S,self.M)) - range/2
		if self.H_m!=0:
			self.W_m = range*np.random.random((self.M,self.H_m)) - range/2
			self.W_h_m = range*np.random.random((self.H_m,self.A)) - range/2
		else:
			self.W_m = range*np.random.random((self.M,self.A)) - range/2

		if self.prop=='std' or self.prop=='RBP':
			if self.H_r!=0:
				self.W_r_back = range*np.random.random((self.H_r,self.R)) - range/2
				self.W_h_r_back = range*np.random.random((self.A,self.H_r)) - range/2
			else:
				self.W_r_back = range*np.random.random((self.A,self.R)) - range/2
			if self.H_m!=0:
				self.W_m_back = range*np.random.random((self.H_m,self.M)) - range/2
				self.W_h_m_back = range*np.random.random((self.A,self.H_m)) - range/2
			else:
				self.W_m_back = range*np.random.random((self.A,self.M)) - range/2

		elif self.prop=='BP':

			self.W_r_back = np.transpose(self.W_r)
			self.W_m_back = np.transpose(self.W_m)
			if self.H_r!=0:
				self.W_h_r_back = np.transpose(self.W_h_r)
			if self.H_m!=0:
				self.W_h_m_back = np.transpose(self.W_h_m)

		elif self.prop=='SRBP':
			if self.H_r!=0:
				self.W_r_back = range*np.random.random((self.A,self.R)) - range/2
				self.W_h_r_back = range*np.random.random((self.A,self.H_r)) - range/2
			else:
				self.W_r_back = range*np.random.random((self.A,self.R)) - range/2			

			if self.H_m!=0:
				self.W_m_back = range*np.random.random((self.A,self.M)) - range/2
				self.W_h_m_back = range*np.random.random((self.A,self.H_m)) - range/2
			else:
				self.W_m_back = range*np.random.random((self.A,self.M)) - range/2

		elif self.prop=='MRBP':

			if self.H_r!=0:
				self.W_r_back = range*np.random.random((self.H_r,self.R)) - range/2
				self.W_r_back_skipped = range*np.random.random((self.A,self.R)) - range/2
				self.W_h_r_back = range*np.random.random((self.A,self.H_r)) - range/2
			else:
				self.W_r_back = range*np.random.random((self.A,self.R)) - range/2		

			if self.H_m!=0:
				self.W_m_back = range*np.random.random((self.H_m,self.M)) - range/2
				self.W_m_back_skipped = range*np.random.random((self.A,self.M)) - range/2
				self.W_h_m_back = range*np.random.random((self.A,self.H_m)) - range/2

			else:
				self.W_m_back = range*np.random.random((self.A,self.M)) - range/2

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


		
	def update_weights(self, RPE):

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


	def update_tags(self,s_inst,s_trans,y_r,y_m,y_h_r,y_h_m,z,resp_ind):

		# synaptic trace for memory units
		self.sTRACE = self.sTRACE*self.memory_leak + np.tile(np.transpose(s_trans), (1,self.M))

		# synaptic tags for memory branch (deep and shallow)
		if self.H_m!=0:
			self.Tag_w_h_m += -self.alpha*self.Tag_w_h_m + np.dot(np.transpose(y_h_m), z)	
			delta_h_m = self.W_h_m_back[resp_ind,:]
			self.Tag_w_m += -self.alpha*self.Tag_w_m + np.dot(np.transpose(y_m), y_h_m*(1-y_h_m)*delta_h_m)
			if self.prop=='BP' or self.prop=='std':		
				delta_m = np.dot(y_h_m*(1-y_h_m)*delta_h_m,self.W_m_back)
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
			self.Tag_w_h_r += -self.alpha*self.Tag_w_h_r + np.dot(np.transpose(y_h_r), z)	
			delta_h_r = self.W_h_r_back[resp_ind,:]
			self.Tag_w_r += -self.alpha*self.Tag_w_r + np.dot(np.transpose(y_r), y_h_r*(1-y_h_r)*delta_h_r)
			if self.prop=='BP' or self.prop=='std':		
				delta_r = np.dot(y_h_r*(1-y_h_r)*delta_h_r,self.W_r_back)
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
		else:		
			y_h_r = None

		y_m,self.cumulative_memory = act.sigmoid_acc_leaky(s_trans, self.V_m, self.cumulative_memory,self.memory_leak)
		if self.H_m!=0:		
			y_h_m = act.sigmoid(y_m,self.W_m)
		else:		
			y_h_m = None

		if self.H_r!=0 and self.H_m!=0:
			y_tot = np.concatenate((y_h_r, y_h_m),axis=1)
			W_tot = np.concatenate((self.W_h_r, self.W_h_m),axis=0)
		elif self.H_r==0 and self.H_m!=0:
			y_tot = np.concatenate((y_r, y_h_m),axis=1)
			W_tot = np.concatenate((self.W_r, self.W_h_m),axis=0)
		elif self.H_r!=0 and self.H_m==0:
			y_tot = np.concatenate((y_h_r, y_m),axis=1)
			W_tot = np.concatenate((self.W_h_r, self.W_m),axis=0)
		else:
			y_tot = np.concatenate((y_r, y_m),axis=1)
			W_tot = np.concatenate((self.W_r, self.W_m),axis=0)			
		#print(y_tot)
		Q = act.linear(y_tot, W_tot)


		return y_r, y_m, Q, y_h_r, y_h_m 



	def check_angle_cond(self,RPE,resp_ind,y_h_r,y_h_m):

		if self.H_r!=0:
			mod_BP_H_R = RPE*np.transpose(self.W_h_r[:,resp_ind])
			mod_FA_H_R = RPE*self.W_h_r_back[resp_ind,:]
			norm_H_R = np.linalg.norm(mod_BP_H_R)*np.linalg.norm(mod_FA_H_R)
			RBP_cond_H_R = np.dot(np.transpose(mod_BP_H_R), mod_FA_H_R)/(norm_H_R)
	
			mod_BP_R = np.dot(mod_BP_H_R*np.squeeze(y_h_r)*(1-np.squeeze(y_h_r)), np.transpose(self.W_r))
			#mod_BP_R = np.dot(mod_BP_H_R, np.transpose(self.W_r))
			if self.prop=='RBP':
				mod_FA_R = np.dot(mod_FA_H_R, self.W_r_back)
			elif self.prop=='SRBP':
				mod_FA_R = RPE*self.W_r_back[resp_ind,:]
			elif self.prop=='MRBP':
				mod_FA_R = self.a*np.dot(mod_FA_H_R, self.W_r_back) + (1-self.a)*RPE*self.W_r_back_skipped[resp_ind,:]						
			norm_R = np.linalg.norm(mod_BP_R)*np.linalg.norm(mod_FA_R)
			RBP_cond_R = np.dot(np.transpose(mod_BP_R), mod_FA_R)/(norm_R)
		else:	
			RBP_cond_H_R = 0
			mod_BP_R = RPE*np.transpose(self.W_r[:,resp_ind])
			mod_FA_R = RPE*self.W_r_back[resp_ind,:]
			norm_R = np.linalg.norm(mod_BP_R)*np.linalg.norm(mod_FA_R)
			RBP_cond_R = np.dot(np.transpose(mod_BP_R), mod_FA_R)/(norm_R)

		if self.H_m!=0:
			mod_BP_H_M = RPE*np.transpose(self.W_h_m[:,resp_ind])
			mod_FA_H_M = RPE*self.W_h_m_back[resp_ind,:]
			norm_H_M = np.linalg.norm(mod_BP_H_M)*np.linalg.norm(mod_FA_H_M)
			RBP_cond_H_M = np.dot(np.transpose(mod_BP_H_M), mod_FA_H_M)/(norm_H_M)
	
			mod_BP_M = np.dot(mod_BP_H_M*np.squeeze(y_h_m)*(1-np.squeeze(y_h_m)), np.transpose(self.W_m))
			#mod_BP_M = np.dot(mod_BP_H_M, np.transpose(self.W_m))
			if self.prop=='RBP':
				mod_FA_M = np.dot(mod_FA_H_M, self.W_m_back)
			elif self.prop=='SRBP':
				mod_FA_M = RPE*self.W_m_back[resp_ind,:]
			elif self.prop=='MRBP':
				mod_FA_M = self.a*np.dot(mod_FA_H_M, self.W_m_back) + (1-self.a)*RPE*self.W_m_back_skipped[resp_ind,:]
			norm_M = np.linalg.norm(mod_BP_M)*np.linalg.norm(mod_FA_M)
			RBP_cond_M = np.dot(np.transpose(mod_BP_M), mod_FA_M)/(norm_M)
		else:
			RBP_cond_H_M = 0
			mod_BP_M = RPE*np.transpose(self.W_m[:,resp_ind])
			mod_FA_M = RPE*self.W_m_back[resp_ind,:]
			norm_M = np.linalg.norm(mod_BP_M)*np.linalg.norm(mod_FA_M)
			RBP_cond_M = np.dot(np.transpose(mod_BP_M), mod_FA_M)/(norm_M)		
	

		return RBP_cond_H_R, RBP_cond_H_M, RBP_cond_R, RBP_cond_M

	def check_angle_cond_2(self):

		for_R = np.reshape(np.transpose(self.W_r),(-1,1))
		back_R = np.reshape(self.W_r_back,(-1,1))
		norm_R = np.linalg.norm(for_R)*np.linalg.norm(back_R)
		RBP_cond_R = np.dot(np.transpose(for_R), back_R)/(norm_R)

		for_M = np.reshape(np.transpose(self.W_m),(-1,1))
		back_M = np.reshape(self.W_m_back,(-1,1))
		norm_M = np.linalg.norm(for_M)*np.linalg.norm(back_M)
		RBP_cond_M = np.dot(np.transpose(for_M), back_M)/(norm_M)

		if self.H_r!=0:
			for_H_R = np.reshape(np.transpose(self.W_h_r),(-1,1))
			back_H_R = np.reshape(self.W_h_r_back,(-1,1))
			norm_H_R = np.linalg.norm(for_H_R)*np.linalg.norm(back_H_R)
			RBP_cond_H_R = np.dot(np.transpose(for_H_R), back_H_R)/(norm_H_R)
		else:
			RBP_cond_H_R = 0

		if self.H_m!=0:
			for_H_M = np.reshape(np.transpose(self.W_h_m),(-1,1))
			back_H_M = np.reshape(self.W_h_m_back,(-1,1))
			norm_H_M = np.linalg.norm(for_H_M)*np.linalg.norm(back_H_M)
			RBP_cond_H_M = np.dot(np.transpose(for_H_M), back_H_M)/(norm_H_M)
		else:
			RBP_cond_H_M = 0

		return RBP_cond_H_R, RBP_cond_H_M, RBP_cond_R, RBP_cond_M



	def define_transient(self, s,s_old):

		s_plus =  np.where(s<=s_old,0,1)
		s_minus = np.where(s_old<=s,0,1)
		s_trans = np.concatenate((s_plus,s_minus),axis=1)

		return s_trans

######## TRAINING + TEST FOR 12AX TASK

	def train(self,N,p_c,average_sample,conv_criterion='strong',stop=True,verbose=False):

		E = np.zeros((N))
		zero = np.zeros((1,self.S))
		s_old = zero

		correct = 0
		convergence = False
		conv_ep = np.array([0]) 
		ep_corr = 0

		min_loops = 1
		max_loops = 4

		if self.prop=='RBP':
			RBP_cond_M = np.zeros((N))
			RBP_cond_R = np.zeros((N))		
			RBP_cond_H_M = np.zeros((N))
			RBP_cond_H_R = np.zeros((N))

		for n_ep in np.arange(N):

			if verb:
				print('EPISODE ', n_ep+1)
			S, O = construct_trial(p_c,min_loops,max_loops)
			N_stimuli = np.shape(S)[0]
			s_old = zero

			self.reset_memory()
			self.reset_tags()

			ep_corr_bool = True
			
			for n in np.arange(N_stimuli):

				s = S[n:(n+1),:]	
				s_inst = s
				s_trans = self.define_transient(s_inst,s_old)
				s_old = s
				s_print = self.dic_stim[np.argmax(s)]
			
				o = O[n:(n+1),:]
				o_print = self.dic_resp[np.argmax(o)]	
 
				y_r, y_m, Q, y_h_r, y_h_m = self.feedforward(s_inst, s_trans)
				if (np.isnan(Q)).any()==True:
					conv_ep = np.array([-1])
					print('ERROR: NaN values. Reduce learning rate (',self.beta,')')	
					break

				resp_ind, _ = self.compute_response(Q,n_ep)
				q = Q[0,resp_ind]

				r_print = self.dic_resp[resp_ind]
	
				z = np.zeros(np.shape(Q))
				z[0,resp_ind] = 1


				if n_ep!=0:
	
					RPE = (r + self.discount*q) - q_old  # Reward Prediction Error
				
					if self.prop=='RBP':
						#rhr, rhm, rr, rm = self.check_angle_cond(RPE,resp_ind,y_h_r,y_h_m)
						rhr, rhm, rr, rm = self.check_angle_cond_2()
						RBP_cond_H_R[n_ep] += rhr
						RBP_cond_H_M[n_ep] += rhm
						RBP_cond_R[n_ep] += rr
						RBP_cond_M[n_ep] += rm		

					self.update_weights(RPE)

				self.update_tags(s_inst,s_trans,y_r,y_m,y_h_r,y_h_m,z,resp_ind)

			
				if r_print!=o_print:
					r = self.rew_neg
					E[n_ep] += 1
					correct = 0
					ep_corr_bool = False
				else:
					r = self.rew_pos
					correct += 1	
				q_old = q
				if verb:
					print('\t\t STIM: ',s_print,'\t OUT: ',o_print,'\t RESP: ', r_print,'\t\t Q: ',Q,'\t\t corr',correct)

			RPE = r-q_old

			if ep_corr_bool==True:
				ep_corr += 1
			else:
				ep_corr = 0			

			if self.prop=='RBP':
				rhr, rhm, rr, rm = self.check_angle_cond_2()

				RBP_cond_H_R[n_ep] = (RBP_cond_H_R[n_ep]+rhr)/N_stimuli
				RBP_cond_H_M[n_ep] = (RBP_cond_H_M[n_ep]+rhm)/N_stimuli
				RBP_cond_R[n_ep] = (RBP_cond_R[n_ep]+rr)/N_stimuli
				RBP_cond_M[n_ep] = (RBP_cond_M[n_ep]+rm)/N_stimuli

			
			self.update_weights(RPE)

	
			if conv_criterion=='strong':
				if correct>=1000 and convergence==False:
					conv_ep = np.array([n_ep+1]) 
					convergence = True
					self.epsilon = 0
					if stop==True:
						break;
			elif conv_criterion=='lenient':
				if ep_corr==50 and convergence==False:
					conv_ep = np.array([n_ep]) 
					convergence = True					
					if stop==True:
						break;

		print('SIMULATION MET CRITERION AT EPISODE', conv_ep,'\t (',conv_criterion,' criterion)')

		E = np.mean(np.reshape(E,(-1,average_sample)),axis=1)

		if self.prop=='RBP':

			RBP_cond_R = np.mean(np.reshape(RBP_cond_R,(-1,average_sample)),axis=1)
			RBP_cond_M = np.mean(np.reshape(RBP_cond_M,(-1,average_sample)),axis=1)
			RBP_cond_H_R = np.mean(np.reshape(RBP_cond_H_R,(-1,average_sample)),axis=1)
			RBP_cond_H_M = np.mean(np.reshape(RBP_cond_H_M,(-1,average_sample)),axis=1)

			return E, conv_ep, RBP_cond_R, RBP_cond_M, RBP_cond_H_R, RBP_cond_H_M

		return E, conv_ep


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

###########################################################################################################################################################################################################################################

task = '12-AX'	
cues_vec = ['1','2','A','B','C','X','Y','Z']
cues_vec_tot = ['1+','2+','A+','B+','C+','X+','Y+','Z+','1-','2-','A-','B-','C-','X-','Y-','Z-']
pred_vec = ['L','R']

dic_stim = {0:'1',1:'2',2:'A',3:'B',4:'C',5:'X',6:'Y',7:'Z'}
dic_resp = {0:'L',1:'R'}

N = 200000
p_c = 0.5

## CONSTRUCTION OF THE AuGMEnT NETWORK
S = 8       		     # dimension of the input = number of possible stimuli
R = 10			     # dimension of the regular units
M = 10			     # dimension of the memory units
A = 2			     # dimension of the activity units = number of possible responses

H_m = 10
H_r = 10

if H_m!=0 and H_r!=0:
	deep_mode ='DCM'
elif H_m==0 and H_r!=0:
	deep_mode ='DC'
elif H_m!=0 and H_r==0:
	deep_mode='DM'
else:
	deep_mode='flat'

N_sim = 1

# value parameters
lamb = 0.5  			# synaptic tag decay 
beta = 0.15			# weight update coefficient
discount = 0.9			# discount rate for future rewards
alpha = 1-lamb*discount 	# synaptic permanenc
eps = 0.025			# fraction of softmax modality for activity selection
g = 1
leak = 0.7

rew = 'BRL'
prop_system = ['std','BP','RBP','SRBP','MRBP']
prop = 'std'

print('\nDeep_mode',deep_mode,'\t H_R=',H_r,'\t H_M=',H_m)
print('Decay=', lamb,'\t learn_rate=', beta,'\t discount=',discount)
print('Leak=', leak, '\t Expl_rate=',eps,' \t gain=',g)
print('REWARD: ', rew)
print('PROPAGATION: ',prop,'\n\n')

verb = 0	
do_training = True
do_test = False
do_plots = True
		
## TRAINING
conv_ep = np.zeros(N_sim)

data_folder = 'DATA'
image_folder = 'IMAGES'
if do_training:
		
	average_sample=500	
	
	for n in np.arange(N_sim):

		print('SIMULATION N.',n+1)

		model = deep_AuGMEnT(S,R,M,H_r,H_m,A,alpha,beta,discount,eps,g,leak,rew,dic_stim,dic_resp,prop)

		if prop=='RBP':

			E,conv_ep[n],RBP_R,RBP_M,RBP_H_R,RBP_H_M = model.train(N,p_c,average_sample,'strong',True,verb)

			#np.savetxt(data_folder+'/'+deep_mode+'_'+prop+'_'+task+'_error.txt', E)
			#np.savetxt(data_folder+'/'+deep_mode+'_'+prop+'_'+task+'_conv.txt', conv_ep)
			np.savetxt(data_folder+'/'+deep_mode+'_'+prop+'_'+task+'_RBP_r.txt', RBP_R)
			np.savetxt(data_folder+'/'+deep_mode+'_'+prop+'_'+task+'_RBP_m.txt', RBP_M)
			if H_r!=0:
				np.savetxt(data_folder+'/'+deep_mode+'_'+prop+'_'+task+'_RBP_hr.txt', RBP_H_R)
			if H_m!=0:
				np.savetxt(data_folder+'/'+deep_mode+'_'+prop+'_'+task+'_RBP_hm.txt', RBP_H_M)
		else:
			E,conv_ep[n] = model.train(N,p_c,average_sample,'strong',True,verb)

			#np.savetxt(data_folder+'/'+deep_mode+'_'+prop+'_'+task+'_error.txt', E)
			#np.savetxt(data_folder+'/'+deep_mode+'_'+prop+'_'+task+'_conv.txt', conv_ep)

np.savetxt(data_folder+'/NEW_'+deep_mode+'_'+prop+'_'+task+'_conv.txt', conv_ep)
	


## TEST
if do_test:
	print('TEST...\n')
	model.test(S_test,O_test,0)
	

## PLOTS
fontTitle = 26
fontTicks = 22
fontLabel = 22
	
if do_plots:

	xs = (np.arange(np.shape(E)[0])+1)*average_sample


	if prop=='RBP':

		figRBP = plt.figure(figsize=(20,25))	
		if H_r!=0:
			plt.subplot(2,2,1)
			plt.plot(xs,np.arccos(RBP_H_R)*180/np.pi,'r')
			#plt.plot(xs,RBP_H_R,'r')
			plt.xlabel('Training Episodes',fontsize=fontLabel)
			plt.ylabel('Angle [degrees]',fontsize=fontLabel)
			plt.title('Feedback Alignment: Hidden Regular Units',fontsize=fontTitle)
			plt.xlim((0,conv_ep[n]))
		if H_m!=0:		
			plt.subplot(2,2,2)			
			plt.plot(xs,np.arccos(RBP_H_M)*180/np.pi,'g')
			#plt.plot(xs,RBP_H_M,'g')
			plt.xlabel('Training Episodes',fontsize=fontLabel)
			plt.ylabel('Angle [degrees]',fontsize=fontLabel)
			plt.title('Feedback Alignment: Hidden Memory Units',fontsize=fontTitle)	
			plt.xlim((0,conv_ep[n]))		
		plt.subplot(2,2,3)
		plt.plot(xs,np.arccos(RBP_R)*180/np.pi,'b')
		#plt.plot(xs,RBP_R,'b')
		plt.xlabel('Training Episodes',fontsize=fontLabel)
		plt.ylabel('Angle [degrees]',fontsize=fontLabel)
		plt.title('Feedback Alignment: Regular Units',fontsize=fontTitle)
		plt.xlim((0,conv_ep[n]))
		plt.subplot(2,2,4) 
		plt.plot(xs,np.arccos(RBP_M)*180/np.pi,'k')
		#plt.plot(xs,RBP_M,'k')
		plt.xlabel('Training Episodes',fontsize=fontLabel)
		plt.ylabel('Angle [degrees]',fontsize=fontLabel)
		plt.title('Feedback Alignment: Memory Units',fontsize=fontTitle)
		plt.xlim((0,conv_ep[n]))

		plt.show()
		saveRBPcond = image_folder+'/AuG_'+deep_mode+'_'+prop+'_'+task+'_RBP_cond.png'	
		figRBP.savefig(saveRBPcond)

	figE = plt.figure(figsize=(10,8))		
	plt.plot(xs,E,color='green')
	plt.axvline(x=4492/6, linewidth=5, ls='dashed', color='b')
	if conv_ep[n]!=0 and conv_ep[n]!=-1:		
		plt.axvline(x=conv_ep[n], linewidth=5, color='g')
	tit = '12AX: Training Convergence'
	plt.title(tit,fontweight="bold",fontsize=fontTitle)		
	plt.xlabel('Training Iterations',fontsize=fontLabel)
	plt.ylabel('Average Number of errors',fontsize=fontLabel)		
	plt.xticks(np.linspace(0,N,5,endpoint=True),fontsize=fontTicks)	
	plt.yticks(fontsize=fontTicks)
	plt.show()

	savestr = image_folder+'/AuG_'+deep_mode+'_'+prop+'_'+task+'error.png'	
	figE.savefig(savestr)
