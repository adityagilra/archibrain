## HIERARCHICAL AuGMEnT (RECURRENT)
## Hierarchical AuGMEnT is a variant of AuGMEnT model where the memory branch of the architecture is made hierarchical by stacking multiple memory levels.
## The model can accept different values for leak, learning rate or decay for each memory level to differentiate their learning and temporal behaviors.
## It is a recurrent version of hierarchical AuGMEnT because the memory layer is connected ALSO with the regular layer in the controller branch making the network recurrent. This is done to try to aggregate the information of all the memory levels before going into the output.
##
## N.B. The current version of hierarchical model presents a gating mechanism that discriminates in a multiplicative way how to distribute the input information over the different levels. Gating is of type: inp(t)= (1-g(t))*inp(t-1) + g(t)*s(t)V, where g(t) is a vector computed in the controller branch of softmax values where each unit is associated to a memory level.
##
## AUTHOR: Marco Martinolli
## DATE: 10.07.2017
from AuGMEnT_model_final import AuGMEnT
import activations as act				

import numpy as np
import matplotlib 
from matplotlib import pyplot as plt
from matplotlib import gridspec
import pylab
import gc 

from sys import version_info


class hierarchical_AuGMEnT(AuGMEnT):

	## Inputs
	# ------------
	# L: int, number of levels of the hierarchical memory architecture
	# alpha,beta and leak can be scalars (as usual) or lists, to indicate specific dynamics for each level


	def __init__(self,L,S,R,M,A,ALPHA,BETA,discount,eps,gain,rew_rule='RL',dic_stim=None,dic_resp=None,prop='std',g_str=1):

		self.L = L

		self.ALPHA = np.reshape(ALPHA,(self.L,1,1))
		self.alpha = self.ALPHA[0,0,0]

		self.BETA = np.reshape(BETA,(self.L,1,1))
		self.beta = self.BETA[0,0,0]

		self.it_lim = 1000		

		self.g_strength = g_str

		super(hierarchical_AuGMEnT,self).__init__(S,R,M,A,self.alpha,self.beta,discount,eps,gain,1,rew_rule,dic_stim,dic_resp,prop)


	def initialize_weights_and_tags(self):

		range = 1

		if self.prop=='std' or self.prop=='RBP' or  self.prop=='SRBP' or  self.prop=='MRBP':
			self.V_r = range*np.random.random((self.S, self.R)) - range/2
			self.W_r = range*np.random.random((self.R, self.A)) - range/2
			self.W_r_back = range*np.random.random((self.A, self.R)) - range/2

			self.V_m = range*np.random.random((self.L, 2*self.S, self.M)) - range/2
			self.W_m = range*np.random.random((self.L, self.M, self.A)) - range/2
			self.W_m_back = range*np.random.random((self.L, self.A, self.M)) - range/2
		
			self.C = range*np.random.random((self.L, self.M, self.R)) - range/2
			self.C_back = range*np.random.random((self.L, self.R, self.M)) - range/2

		elif  self.prop=='BP':
			self.V_r = range*np.random.random((self.S, self.R)) - range/2
			self.W_r = range*np.random.random((self.R, self.A)) - range/2
			self.W_r_back = np.transpose(self.W_r)

			self.V_m = range*np.random.random((self.L, 2*self.S, self.M)) - range/2
			self.W_m = range*np.random.random((self.L, self.M, self.A)) - range/2
			self.W_m_back = np.transpose(self.W_m,(0,2,1))

			self.C = range*np.random.random((self.L, self.M, self.R)) - range/2
			self.C_back = np.transpose(self.C,(0,2,1))

		self.W_g = np.random.random((self.S, self.L)) - 0.5

		self.reset_memory()
		self.reset_tags()


	def update_weights(self,RPE):

		self.W_r += self.beta*RPE*self.Tag_w_r
		self.W_r_back += self.beta*RPE*np.transpose(self.Tag_w_r)
		self.V_r += self.beta*RPE*self.Tag_v_r

		self.W_m += self.BETA*RPE*self.Tag_w_m
		self.V_m += self.BETA*RPE*self.Tag_v_m
		
		self.W_g += RPE*self.Tag_w_g
		self.C += self.beta*RPE*self.Tag_c

		if self.prop=='std' or self.prop=='BP':
			self.W_r_back += self.beta*RPE*np.transpose(self.Tag_w_r)
			self.W_m_back += self.BETA*RPE*np.transpose(self.Tag_w_m,(0,2,1))
			self.C_back += self.BETA*RPE*np.transpose(self.Tag_c,(0,2,1))


	def reset_memory(self):
		self.memory_content = 1e-6*np.ones((self.L,1,self.M))

	def reset_tags(self):

		self.sTRACE = 1e-6*np.ones((self.L, 2*self.S, self.M))

		self.Tag_v_r = 1e-6*np.ones((self.S, self.R))
		self.Tag_v_m = 1e-6*np.ones((self.L, 2*self.S, self.M))

		self.Tag_w_r = 1e-6*np.ones((self.R, self.A))
		self.Tag_w_m = 1e-6*np.ones((self.L, self.M, self.A))

		self.Tag_w_g = 1e-6*np.ones((self.S, self.L))

		self.Tag_c = 1e-6*np.ones((self.L, self.M, self.R))

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
				g = self.gain*(1+ (8/np.pi)*np.arctan(it/self.it_lim))
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


	def update_tags(self,s_inst,s_trans,y_r,y_m,g,z,resp_ind):

		# memory branch
		g_exp = np.reshape(g,(self.L,1,1))
		self.sTRACE = self.sTRACE*(1-g_exp) + g_exp*np.tile(np.transpose(s_trans), (self.L,1,self.M))
		#self.sTRACE = g_exp*np.tile(np.transpose(s_trans), (self.L,1,self.M))		

		self.Tag_w_m += -self.ALPHA*self.Tag_w_m + np.dot(np.transpose(y_m,(0,2,1)), z)
		self.Tag_v_m += -self.ALPHA*self.Tag_v_m + self.sTRACE*y_m*(1-y_m)*(self.W_m_back[:,resp_ind:(resp_ind+1),:] + np.transpose(np.dot(self.W_r_back[resp_ind:(resp_ind+1),:]*y_r*(1-y_r),self.C_back),(1,0,2)))

		# regular branch
		self.Tag_w_r += -self.alpha*self.Tag_w_r + np.dot(np.transpose(y_r), z)
		self.Tag_v_r += -self.alpha*self.Tag_v_r + np.dot(np.transpose(s_inst), y_r*(1-y_r)*self.W_r_back[resp_ind,:] )

		# gate tag update
		g_exp = np.reshape(g,(1,self.L))

		self.Tag_w_g += -self.alpha*self.Tag_w_g + np.dot(np.transpose(s_inst),self.g_strength*g_exp*(1-g_exp)*(np.sum( (np.dot(s_trans,self.V_m)-np.squeeze(self.memory_content,axis=1))*np.transpose(y_m*(1-y_m),(1,0,2))*self.W_m_back[:,resp_ind,:],axis=2)  +  np.sum( (np.dot(s_trans,self.V_m)-np.squeeze(self.memory_content,axis=1))*np.transpose(y_m*(1-y_m),(1,0,2))*np.dot(self.W_r_back[resp_ind,:]*y_r*(1-y_r),self.C_back),axis=2)))

	def feedforward(self,s_inst,s_trans):
		
		g = act.softmax(s_inst,self.W_g, self.g_strength)
		g = np.transpose(g)

		y_m = 1e-6*np.ones((self.L,1,self.M))
		inp_r = 1e-6*np.ones((self.L,1,self.R))
		Q = 1e-6*np.ones((1,self.A))
		for l in np.arange(self.L):
			y_m[l,:,:], self.memory_content[l,:,:] = act.sigmoid_acc_leaky(s_trans, self.V_m[l,:,:], self.memory_content[l,:,:],1-g[l,0],g[l,0])
			inp_r[l,:,:] = act.linear(y_m[l,:,:],self.C[l,:,:])			
			Q += act.linear(y_m[l,:,:], self.W_m[l,:,:])
			#print('\t MEM STATE ',str(l),':', y_m[l,:,:],'\t alpha=',self.ALPHA[l,0,0],'\t gate=',g[l,:])

		inp_r = np.sum(inp_r,axis=0)
		inp_r += act.linear(s_inst, self.V_r)
		y_r = act.sigmoidal(inp_r)

		Q += act.linear(y_r, self.W_r)

		return y_r, y_m, g, Q

######### TRAINING + TEST FOR CPT TASKS LIKE 12AX

	def training(self,N_ep,p_c,max_loops,conv_criterion='strong',stop=True,verb=False):

		self.initialize_weights_and_tags()

		zero = np.zeros((1,self.S))
		E = np.zeros((N))

		correct = 0
		convergence = False
		conv_ep = np.array([0])

		ep_corr=0

		for n_ep in np.arange(N_ep):
		
			if verb:
				print('TRIAL N',n_ep+1)

			S, O = construct_trial(p_c,1,max_loops)
			N_stimuli = np.shape(S)[0]
			s_old = zero
			
			self.reset_memory()
			self.reset_tags()
			ep_corr_bool=True

			for n in np.arange(N_stimuli):

				s = S[n:(n+1),:]
				s_inst = s
				s_trans = self.define_transient(s_inst,s_old)
				s_old = s
				s_print = self.dic_stim[np.argmax(s)]			
				o = O[n:(n+1),:]
				o_print = self.dic_resp[np.argmax(o)]
 
				y_r,y_m,g,Q = self.feedforward(s_inst, s_trans)
				if (np.isnan(Q)).any()==True:
					conv_ep = np.array([-1])	
					break
				resp_ind,P_vec = self.compute_response(Q,n_ep)
				q = Q[0,resp_ind]
				r_print = self.dic_resp[resp_ind]
	
				z = np.zeros(np.shape(Q))
				z[0,resp_ind] = 1

				if n!=0:
					RPE = (r + self.discount*q) - q_old  # Reward Prediction Error
					self.update_weights(RPE)
				self.update_tags(s_inst,s_trans,y_r,y_m,g,z,resp_ind)
			
				if r_print!=o_print:
					r = self.rew_neg
					E[n_ep] += 1
					correct = 0
					ep_corr_bool=False
				else: 
					r = self.rew_pos
					correct += 1			
				q_old = q
				if verb:
					print('\t\tSTIM: ',s_print,'\tOUT: ',o_print,'\tRESP: ', r_print,'\t\tGATE: ',np.transpose(g),'\t\t Q: ',Q,'\t\t corr:',correct)

			RPE = r - q_old
			self.update_weights(RPE)

			if ep_corr_bool==True:
				ep_corr += 1
			else:
				ep_corr = 0

			if conv_criterion=='strong':
				if correct>=1000 and convergence==False:
					conv_ep = np.array([n_ep]) 
					convergence = True
					self.epsilon = 0
					if stop==True:
						break
			elif conv_criterion=='lenient':
				if ep_corr==50 and convergence==False:
					conv_ep = np.array([n_ep]) 
					convergence = True					
					if stop==True:
						break

		if convergence == True:
			print('SIMULATION MET CRITERION AT EPISODE', conv_ep,'\t (',conv_criterion,' criterion)')

		return E,conv_ep

	def test(self,N,p_c,reset_case,verbose=False):

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

			y_r, y_m, g, f, Q = self.feedforward(s_inst,s_trans)
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


##########################################################################################################################################
##########################################################################################################################################
##########################################################################################################################################
##########################################################################################################################################
##########################################################################################################################################

np.set_printoptions(precision=2)

task_dic ={'1':'task 12-AX',
	   '2':'saccade/anti-saccade task'}

py3 = version_info[0] > 2 # creates boolean value for test that Python major version > 2
#task_selection = input("\nPlease select a task: \n\t 1: task 12-AX\n\t 2: saccade/anti-saccade task\n Enter id number:  ")

task_selection='1' 
print("\nYou have selected: ", task_dic[task_selection],'\n\n')

#########################################################################################################################################
#######################   TASK 1-2 AX
#########################################################################################################################################

if (task_selection!="2"):
	
	from task_12AX import construct_trial				
	task = '12-AX'
	
	cues_vec = ['1','2','A','B','C','X','Y','Z']
	cues_vec_tot = ['1+','2+','A+','B+','C+','X+','Y+','Z+','1-','2-','A-','B-','C-','X-','Y-','Z-']
	pred_vec = ['L','R']

	dic_stim = {0:'1',1:'2',2:'A',3:'B',4:'C',5:'X',6:'Y',7:'Z'}
	
	dic_resp =  {0:'L',1:'R'}

	N = 100000
	perc_tr = 0.8
	p_c = 0.5
	N_tr = np.round(N*perc_tr).astype(int)

	## CONSTRUCTION OF THE AuGMEnT NETWORK
	S = 8        # dimension of the input = number of possible stimuli
	R = 10			     # dimension of the regular units
	M = 10			     # dimension of the memory units
	A = 2			     # dimension of the activity units = number of possible responses

	# value parameters were taken from the 
	discount = 0.9			# discount rate for future rewards
	eps = 0.025			# fraction of softmax modality for activity selection
	g = 1
	g_str = 4

	# reward settings
	rew_system = ['RL','PL','SRL','BRL']
	rew = 'BRL'
	prop_system = ['std','BP','RBP','SRBP','MRBP']
	prop = 'std'

	N_sim = 1
	max_loops = 4

	verb = 1
	
	do_training = True
	do_test = False
	do_plots = False

	model_opt_system = ['base','deep','hier']
	model_opt = 'hier'
	spec_opt = None

	L = 2
	LAMBDA = [0.2,0.8]
	BETA = [0.1,0.08]

	#L = 3
	#LAMBDA = [0.2,0.2,0.8]
	#BETA = [0.05,0.05,0.025]

	#L=1
	#LAMBDA = [0.5]
	#BETA = [0.2]

	ALPHA = [ 1 - discount*l for l in LAMBDA]

	## TRAINING
	data_folder = 'DATA'
	image_folder = 'IMAGES'
	weight_folder = 'WEIGHT_DATA'

	conv_ep = np.zeros(N_sim)

	if do_training:

		average_sample=100

		for n in np.arange(N_sim):
		
			print('SIMULATION N.',n+1)

			model = hierarchical_AuGMEnT(L,S,R,M,A,ALPHA,BETA,discount,eps,g,rew,dic_stim,dic_resp,prop,g_str)
			
			E,conv_ep[n] = model.training(N,p_c,max_loops,'strong',True,verb)

		#np.savetxt(data_folder+'/'+model_opt+'_'+prop+'_error.txt', E)
		#np.savetxt(data_folder+'/'+model_opt+'_'+prop+'_conv.txt', conv_ep)

	np.savetxt(data_folder+'/NEW_'+model_opt+'_'+prop+'_conv.txt', conv_ep)
	

	## TEST
	if do_test:
		print('TEST...\n')
		model.test(N,p_c,0)

	## PLOTS
	fontTitle = 26
	fontTicks = 22
	fontLabel = 22
		
	if do_plots:

		N = len(E)
		bin = np.round(0.05*N).astype(int)
		END = np.floor(N/bin).astype(int)
		E = E[:END*bin]
		N = len(E)

		E_bin = np.reshape(E,(-1,bin))
		E_bin = np.sum(E_bin,axis=1)

		figE = plt.figure(figsize=(20,8))
		N_round = np.around(N/1000).astype(int)*1000
		
		plt.subplot(1,2,1)
		plt.bar(bin*np.arange(len(E_bin)),E_bin,width=bin,color='green',edgecolor='black', alpha=0.6)
		plt.axvline(x=4492/6, linewidth=5, ls='dashed', color='b')
		if conv_ep!=0:		
			plt.axvline(x=conv_ep, linewidth=5, color='g')
		tit = '12AX: Training Convergence'
		plt.title(tit,fontweight="bold",fontsize=fontTitle)		
		plt.xlabel('Training Iterations',fontsize=fontLabel)
		plt.ylabel('Number of Errors per bin',fontsize=fontLabel)		
		plt.xticks(np.linspace(0,N_round,5,endpoint=True),fontsize=fontTicks)	
		plt.yticks(fontsize=fontTicks)	
		text = 'Bin = '+str(bin)
		plt.figtext(x=0.38,y=0.78,s=text,fontsize=fontLabel,bbox={'facecolor':'white', 'alpha':0.5, 'pad':10})

		plt.subplot(1,2,2)
		#LOSS = 0.5*np.mean(np.reshape(E,(-1,average_sample)),axis=1)
		#plt.plot(np.arange(np.shape(LOSS)[0])*average_sample,LOSS, color='green',linewidth=7, alpha=0.6)
		#plt.axvline(x=4492/6, linewidth=5, ls='dashed', color='b')
		#if conv_iter!=0:		
		#	plt.axvline(x=conv_iter/6, linewidth=5, color='g')
		#tit = '12AX: Loss Function'
		#plt.title(tit,fontweight="bold",fontsize=fontTitle)			
		#plt.xticks(np.linspace(0,N_round,5,endpoint=True),fontsize=fontTicks)
		#plt.yticks(fontsize=fontTicks)
		#plt.xlabel('Training Trials',fontsize=fontLabel)
		#plt.ylabel('Average Loss Function',fontsize=fontLabel)
		plt.show()

		savestr = image_folder+'/AuG_'+model_opt+'_'+prop+'_'+task+'error.png'	
