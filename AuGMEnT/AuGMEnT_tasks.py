## MAIN FILE FOR AuGMEnT TESTING
## Here are defined the settings for the AuGMEnT architecture and for the tasks to test.
## AUTHOR: Marco Martinolli
## DATE: 27.03.2017

import numpy as np
import matplotlib 
matplotlib.use('GTK3Cairo') 
from matplotlib import pyplot as plt
from matplotlib import gridspec
import pylab
import gc 

import sys
sys.path.append("..")
sys.path.append("AuGMEnT")

from AuGMEnT_model import AuGMEnT


def run_task(task, params_bool=None, params_task=None):
	if(task == '0'):
		AuGMEnT_task_1_2(params_bool, params_task)

	elif(task == '1'):
		AuGMEnT_task_AX_CPT(params_bool, params_task)

	elif(task == '2'):
		AuGMEnT_task_1_2AX_S(params_bool, params_task)

	elif(task == '3'):
		AuGMEnT_task_1_2AX(params_bool, params_task)

	elif(task == '4'):
		AuGMEnT_task_saccades(params_bool, params_task)

	elif(task == '5'):
		AuGMEnT_task_seq_prediction(params_bool, params_task)

	else:
		print('The task is not valid for AuGMEnT\n')


#########################################################################################################################################
#######################   TASK 1-2
#########################################################################################################################################

def AuGMEnT_task_1_2(params_bool, params_task):

	from TASKS.task_1_2 import data_construction
	task = '1-2'

	cues_vec = ['1','2']
	cues_vec_tot = 	['1+','2+','1-','2-']
	pred_vec = ['L','R']

	print('Dataset construction...')

	if params_task is None:
		#N = 1000
		N = 4000
		p1 = 0.7
		tr_perc = 0.8
	else:
		N = int(params_task[0]) if params_task[0] != '' else 4000
		p1 = float(params_task[1]) if params_task[1] != '' else 0.7
		tr_perc = float(params_task[2]) if params_task[2] != '' else 0.8

	np.random.seed(1234)

	[S_tr, O_tr, S_test, O_test, dic_stim, dic_resp] = data_construction(N=N, p1=p1, p2=1-p1, training_perc=tr_perc, model='0')
	print('Done!')
	reset_cond = ['1','2']	

	## CONSTRUCTION OF THE AuGMEnT NETWORK
	S = np.shape(S_tr)[1]        # dimension of the input = number of possible stimuli
	A = np.shape(O_tr)[1]

	R = 3			     # dimension of the regular units
	M = 4 			     # dimension of the memory units
	
	# value parameters were taken from the 
	lamb = 0.2    			# synaptic tag decay 
	beta = 0.1			# weight update coefficient
	discount = 0.09			# discount rate for future rewards
	alpha = 1-lamb*discount 	# synaptic permanence
	eps = 0.05			# percentage of softmax modality for activity selection
	g = 3

	# reward settings
	rew_system = ['RL','PL','SRL']
	rew = 'RL'

	verb = 0
		
	if params_bool is None:
		do_training = True
		do_test = True
		do_weight_plots = True
		do_error_plots = True

	else:
		do_training = params_bool[0]
		do_test = params_bool[1]
		do_weight_plots = params_bool[2]	
		do_error_plots = params_bool[3]

	fontTitle = 22
	fontTicks = 20
	fontLabel = 20

	reg_vec=[]
	mem_vec=[]
	for i in range(R):
		reg_vec.append('R'+str(i+1))
	for i in range(M):
		mem_vec.append('M'+str(i+1))		
	
	## ARCHITECTURE
	model = AuGMEnT(S,R,M,A,alpha,beta,discount,eps,g,rew,dic_stim,dic_resp)


	## TRAINING
	if do_training:	
		print('TRAINING...\n')
		g = 3
		E = model.training_ON_OFF(S_tr,O_tr,reset_cond,verb)

	## TEST
	if do_test:
		print('TEST...\n')
		g = 10
		model.test_ON_OFF(S_test,O_test,reset_cond,0)

	## PLOTS
	# plot of the memory weights
	folder = 'AuGMEnT/IMAGES'
	if do_weight_plots:

		fig = plt.figure(figsize=(30,8))

		W = model.V_r
		plt.subplot(1,3,1)
		plt.pcolor(np.flipud(W),edgecolors='k', linewidths=1)
		plt.set_cmap('gray_r')			
		plt.colorbar()
		tit = 'ASSOCIATION WEIGHTS'
		plt.title(tit,fontweight="bold",fontsize=fontTitle)			
		plt.yticks(np.linspace(0.5,S-0.5,S,endpoint=True),np.flipud(cues_vec),fontsize=fontTicks)
		plt.xticks(np.linspace(0.5,R-0.5,R,endpoint=True),reg_vec,fontsize=fontTicks)

		plt.subplot(1,3,2)
		X = model.V_m
		plt.pcolor(np.flipud(X),edgecolors='k', linewidths=1)
		plt.set_cmap('gray_r')			
		plt.colorbar()
		tit = 'MEMORY WEIGHTS'
		plt.title(tit,fontweight="bold",fontsize=fontTitle)
		plt.yticks(np.linspace(0.5,2*S-0.5,2*S,endpoint=True),np.flipud(cues_vec_tot),fontsize=fontTicks)
		plt.xticks(np.linspace(0.5,M-0.5,M,endpoint=True),mem_vec,fontsize=fontTicks)
		
		WW = np.concatenate((model.W_r, model.W_m),axis=0)
		plt.subplot(1,3,3)
		plt.pcolor(np.flipud(WW),edgecolors='k', linewidths=1)
		plt.set_cmap('gray_r')			
		plt.colorbar()
		tit = 'OUTPUT WEIGHTS'
		plt.title(tit,fontweight="bold",fontsize=fontTitle)			
		plt.yticks(np.linspace(0.5,(M+R)-0.5,M+R,endpoint=True),np.flipud(np.concatenate((reg_vec,mem_vec))),fontsize=fontTicks)
		plt.xticks(np.linspace(0.5,A-0.5,A,endpoint=True),pred_vec,fontsize=fontTicks)
		plt.show()

		savestr = folder+'/'+task+'_weights_'+rew+'.png'
		fig.savefig(savestr)


#########################################################################################################################################
#######################   TASK AX CPT
#########################################################################################################################################

def AuGMEnT_task_AX_CPT(params_bool, params_task):
	
	from TASKS.task_AX_CPT import data_construction
	task = 'AX_CPT'

	cues_vec = ['A','B','X','Y']
	cues_vec_tot = 	['A+','B+','X+','Y+','A-','B-','X-','Y-']
	pred_vec = ['L','R']

	print('Dataset construction...')
	if params_task is None:
		#N_stimuli = 30000
		N_stimuli = 40000
		target_perc = 0.2
		tr_perc = 0.8
	else:
		N_stimuli = int(params_task[0]) if params_task[0] != '' else 40000
		target_perc = float(params_task[1]) if params_task[1] != '' else 0.2
		tr_perc = float(params_task[2]) if params_task[2] != '' else 0.8

	np.random.seed(1234)

	[S_tr, O_tr, S_test, O_test, dic_stim, dic_resp] = data_construction(N=N_stimuli, perc_target=target_perc, perc_training=tr_perc, model='0')
	print('Done!')
	reset_cond = ['A','B']	

	S = np.shape(S_tr)[1]        # dimension of the input = number of possible stimuli
	A = np.shape(O_tr)[1]			     # dimension of the activity units = number of possible responses

	## CONSTRUCTION OF THE AuGMEnT NETWORK
	R = 3			     # dimension of the regular units
	M = 4 			     # dimension of the memory units
	
	# value parameters were taken from the 
	lamb = 0.2    			# synaptic tag decay 
	beta = 0.1			# weight update coefficient
	discount = 0.09			# discount rate for future rewards
	alpha = 1-lamb*discount 	# synaptic permanence
	eps = 0.025			# percentage of softmax modality for activity selection
	g = 4

	# reward settings
	rew_system = ['RL','PL','SRL']
	rew = 'PL'

	verb = 0

	if params_bool is None:
		do_training = True
		do_test = True
		do_weight_plots = True
		do_error_plots = True
		
	else:
		do_training = params_bool[0]
		do_test = params_bool[1]
		do_weight_plots = params_bool[2]	
		do_error_plots = params_bool[3]

	do_reward_comparison = False

	reg_vec=[]
	mem_vec=[]
	for i in range(R):
		reg_vec.append('R'+str(i+1))
	for i in range(M):
		mem_vec.append('M'+str(i+1))		


	model = AuGMEnT(S,R,M,A,alpha,beta,discount,eps,g,rew,dic_stim,dic_resp)

	## TRAINING
	folder = 'AuGMEnT/DATA'
	if do_training:	
		print('TRAINING...\n')
		g = 3
		E,conv_iter = model.training_ON_OFF(S_tr,O_tr,reset_cond,verb)
		str_err = folder+'/'+task+'_error.txt'
		np.savetxt(str_err, E)

		str_V_r = folder+'/'+task+'_weight_Vr.txt'
		np.savetxt(str_V_r, model.V_r)
		str_V_m = folder+'/'+task+'_weight_Vm.txt'
		np.savetxt(str_V_m, model.V_m)

		str_W_r = folder+'/'+task+'_weight_Wr.txt'
		np.savetxt(str_W_r, model.W_r)
		str_W_m = folder+'/'+task+'_weight_Wm.txt'
		np.savetxt(str_W_m, model.W_m)
		
		str_W_r_back = folder+'/'+task+'_weight_Wr_back.txt'
		np.savetxt(str_W_r_back, model.W_r_back)
		str_W_m_back = folder+'/'+task+'_weight_Wm_back.txt'
		np.savetxt(str_W_m_back, model.W_m_back)

		print('Saved model.')
	else:
		str_W_m_back = folder+'/'+task+'_error.txt'
		E = np.loadtxt(str_err)

		str_V_r = folder+'/'+task+'_weight_Vr.txt'
		model.V_r = np.loadtxt(str_V_r)
		str_V_m = folder+'/'+task+'_weight_Vm.txt'
		model.V_m = np.loadtxt(str_V_m)

		str_W_r = folder+'/'+task+'_weight_Wr.txt'
		model.W_r = np.loadtxt(str_W_r)
		str_W_m = folder+'/'+task+'_weight_Wm.txt'
		model.W_m =np.loadtxt(str_W_m)

		
		str_W_r_back = folder+'/'+task+'_weight_Wr_back.txt'
		model.W_r_back = np.loadtxt(str_W_r_back)
		str_W_m_back = folder+'/'+task+'_weight_Wm_back.txt'
		model.W_m_back = np.loadtxt(str_W_m_back)

		print('Loaded model from disk.')
			
	print('\n------------------------------------------------------------------------------------------					\n-----------------------------------------------------------------------------------------------------\n')

	## TEST
	if do_test:
		print('TEST...\n')
		g = 10
		model.test_ON_OFF(S_test,O_test,reset_cond,0)

	print(conv_iter)


	## PLOTS
	# plot of the memory weights
	folder = 'AuGMEnT/IMAGES'
	fontTitle = 26
	fontTicks = 22
	fontLabel = 22
	if do_weight_plots:

		fig = plt.figure(figsize=(30,8))

		W = model.V_r
		plt.subplot(1,3,1)
		plt.pcolor(np.flipud(W),edgecolors='k', linewidths=1)
		plt.set_cmap('gray_r')			
		plt.colorbar()
		tit = 'ASSOCIATION WEIGHTS'
		plt.title(tit,fontweight="bold",fontsize=fontTitle)			
		plt.yticks(np.linspace(0.5,S-0.5,S,endpoint=True),np.flipud(cues_vec),fontsize=fontTicks)
		plt.xticks(np.linspace(0.5,R-0.5,R,endpoint=True),reg_vec,fontsize=fontTicks)

		plt.subplot(1,3,2)
		X = model.V_m
		plt.pcolor(np.flipud(X),edgecolors='k', linewidths=1)
		plt.set_cmap('gray_r')			
		plt.colorbar()
		tit = 'MEMORY WEIGHTS'
		plt.title(tit,fontweight="bold",fontsize=fontTitle)
		plt.yticks(np.linspace(0.5,2*S-0.5,2*S,endpoint=True),np.flipud(cues_vec_tot),fontsize=fontTicks)
		plt.xticks(np.linspace(0.5,M-0.5,M,endpoint=True),mem_vec,fontsize=fontTicks)
		
		WW = np.concatenate((model.W_r, model.W_m),axis=0)
		plt.subplot(1,3,3)
		plt.pcolor(np.flipud(WW),edgecolors='k', linewidths=1)
		plt.set_cmap('gray_r')			
		plt.colorbar()
		tit = 'OUTPUT WEIGHTS'
		plt.title(tit,fontweight="bold",fontsize=fontTitle)			
		plt.yticks(np.linspace(0.5,(M+R)-0.5,M+R,endpoint=True),np.flipud(np.concatenate((reg_vec,mem_vec))),fontsize=fontTicks)
		plt.xticks(np.linspace(0.5,A-0.5,A,endpoint=True),pred_vec,fontsize=fontTicks)
		plt.show()

		savestr = folder+'/'+task+'_weights_'+rew+'.png'
		fig.savefig(savestr)
		
		
	if do_error_plots:

		N = len(E)
		bin = round(N*0.02)
		END = np.floor(N/bin).astype(int)
		E = E[:END*bin]
		N = len(E)

		E_bin = np.reshape(E,(-1,bin))
		E_bin = np.sum(E_bin,axis=1)
		E_cum = np.cumsum(E)
		E_norm = 100*E_cum/(np.arange(N)+1)
		C = np.where(E==0,1,0)
		C_cum = 100*np.cumsum(C)/(np.arange(N)+1)

		figE = plt.figure(figsize=(20,8))
		N_round = np.around(N/1000).astype(int)*1000
		
		plt.subplot(1,2,1)
		plt.bar(bin*np.arange(len(E_bin))/2,E_bin,width=bin/2,color='green',edgecolor='black', alpha=0.6)
		if conv_iter!=0:		
			plt.axvline(x=conv_iter/2, linewidth=5, color='g')
		tit = 'AX: Training Convergence'
		plt.title(tit,fontweight="bold",fontsize=fontTitle)		
		plt.xlabel('Training Trials',fontsize=fontLabel)
		plt.ylabel('Number of Errors per bin',fontsize=fontLabel)		
		plt.xticks(np.linspace(0,N_round/2,5,endpoint=True),fontsize=fontTicks)	
		plt.yticks(fontsize=fontTicks)	
		text = 'Bin = '+str(bin)
		plt.figtext(x=0.38,y=0.78,s=text,fontsize=fontLabel,bbox={'facecolor':'white', 'alpha':0.5, 'pad':10})

		plt.subplot(1,2,2)
		plt.plot(np.arange(N)/2, E_cum, color='green',linewidth=7, alpha=0.6)
		if conv_iter!=0:		
			plt.axvline(x=conv_iter/2, linewidth=5, color='g')
		tit = 'AX: Cumulative Training Error'
		plt.title(tit,fontweight="bold",fontsize=fontTitle)			
		plt.xticks(np.linspace(0,N_round/2,5,endpoint=True),fontsize=fontTicks)
		plt.yticks(fontsize=fontTicks)
		plt.xlabel('Training Trials',fontsize=fontLabel)
		plt.ylabel('Cumulative Error',fontsize=fontLabel)
		plt.show()

		savestr = folder+'/'+task+'_error_'+rew+'.png'		
		if M==0:
			savestr = folder+'/'+task+'_error_'+rew+'_nomemory.png'
		figE.savefig(savestr)


	if do_reward_comparison:
		model = AuGMEnT(S,R,M,A,alpha,beta,discount,eps,g,rew,dic_stim,dic_resp)
		model_RL = AuGMEnT(S,R,M,A,alpha,beta,discount,eps,g,'RL',dic_stim,dic_resp)
		model_PL = AuGMEnT(S,R,M,A,alpha,beta,discount,eps,g,'PL',dic_stim,dic_resp)
		model_SRL = AuGMEnT(S,R,M,A,alpha,beta,discount,eps,g,'SRL',dic_stim,dic_resp)
		g = 10
		E_RL = model_RL.training_ON_OFF(S_tr,O_tr,reset_cond,verb)
		E_PL = model_PL.training_ON_OFF(S_tr,O_tr,reset_cond,verb)
		E_SRL = model_SRL.training_ON_OFF(S_tr,O_tr,reset_cond,verb)
		
		N = len(E_RL)
		bin = round(N*0.05)
		
		E_RL_bin = np.reshape(E_RL,(-1,bin))
		E_RL_bin = np.sum(E_RL_bin,axis=1)
		E_PL_bin = np.reshape(E_PL,(-1,bin))
		E_PL_bin = np.sum(E_PL_bin,axis=1)
		E_SRL_bin = np.reshape(E_SRL,(-1,bin))
		E_SRL_bin = np.sum(E_SRL_bin,axis=1)
		
		E_RL_cum = np.cumsum(E_RL)
		E_PL_cum = np.cumsum(E_PL)
		E_SRL_cum = np.cumsum(E_SRL)

		figE = plt.figure(figsize=(25,16))
		fontLabel = 24
		gs = gridspec.GridSpec(3, 2)
		plt.subplot(gs[0,0])
		plt.bar(bin*np.arange(len(E_RL_bin)),E_RL_bin,width=bin,color='orange',edgecolor='black')
		tit = 'Reinforcement Learning'
		plt.title(tit,fontweight="bold",fontsize=fontTitle)		
		#plt.xlabel('Training Iterations',fontsize=fontLabel)
		plt.ylabel('Number of Errors',fontsize=fontLabel)		
		plt.xticks(np.linspace(0,N,5,endpoint=True),fontsize=fontTicks)	
		plt.yticks(fontsize=fontTicks)	
		text = 'Bin = '+str(bin)
		plt.figtext(x=0.39,y=0.85,s=text,fontsize=fontLabel,bbox=dict(facecolor='white', alpha=0.5))

		plt.subplot(gs[1,0])
		plt.bar(bin*np.arange(len(E_PL_bin)),E_PL_bin,width=bin,color='blue',edgecolor='black')
		tit = 'Punishment Learning'
		plt.title(tit,fontweight="bold",fontsize=fontTitle)		
		#plt.xlabel('Training Iterations',fontsize=fontLabel)
		plt.ylabel('Number of Errors',fontsize=fontLabel)		
		plt.xticks(np.linspace(0,N,5,endpoint=True),fontsize=fontTicks)	
		plt.yticks(fontsize=fontTicks)	

		plt.subplot(gs[2,0])
		plt.bar(bin*np.arange(len(E_SRL_bin)),E_SRL_bin,width=bin,color='red',edgecolor='black')
		tit = 'Super Reinforcement Learning'
		plt.title(tit,fontweight="bold",fontsize=fontTitle)		
		plt.xlabel('Training Iterations',fontsize=fontLabel)
		plt.ylabel('Number of Errors',fontsize=fontLabel)		
		plt.xticks(np.linspace(0,N,5,endpoint=True),fontsize=fontTicks)	
		plt.yticks(fontsize=fontTicks)	

		plt.subplot(gs[:,1])
		plt.plot(np.arange(N), E_RL_cum, color='orange',linewidth=3)
		plt.plot(np.arange(N), E_PL_cum, color='blue',linewidth=3)
		plt.plot(np.arange(N), E_SRL_cum, color='red',linewidth=3)
		plt.legend(['RL','PL','SRL'],loc='upper left',fontsize=fontLabel)
		tit = 'Reward Rule Comparison'
		plt.title(tit,fontweight="bold",fontsize=fontTitle)			
		plt.xticks(np.linspace(0,N,5,endpoint=True),fontsize=fontTicks)
		plt.yticks(fontsize=fontTicks)
		plt.xlabel('Training Iterations',fontsize=fontLabel)
		plt.ylabel('Cumulative Error',fontsize=fontLabel)
		plt.show()
		savestr = folder+'/'+task+'_reward_comparison.png'
		figE.savefig(savestr)

		
#########################################################################################################################################
#######################   TASK 12AX_simple
#########################################################################################################################################

def AuGMEnT_task_1_2AX_S(params_bool, params_task):
	
	from TASKS.task_1_2AX_S import data_construction
	task = '12AX_S' 

	np.random.seed(1234)
	
	cues_vec = ['1','2','AX','AY','BX','BY']
	cues_vec_tot = ['1+','2+','AX+','AY+','BX+','BY+','1-','2-','AX-','AY-','BX-','BY-']
	pred_vec = ['L','R']
	
	if params_task is None:
		#N = 28000
		N = 100000
		p_digit = 0.1
		#p_wrong = 0.2
		#p_correct = 0.2
		p_wrong = 0.15
		p_correct = 0.25
		perc_training = 0.8
	else:
		N = int(params_task[0]) if params_task[0] != '' else 100000
		p_digit = float(params_task[1]) if params_task[1] != '' else 0.1
		p_wrong = float(params_task[2]) if params_task[2] != '' else 0.15
		p_correct = float(params_task[3]) if params_task[3] != '' else 0.25
		perc_training = float(params_task[4]) if params_task[4] != '' else 0.8

	[S_tr, O_tr, S_test, O_test, dic_stim, dic_resp] = data_construction(N, p_digit, p_wrong, p_correct, perc_training, model='0')
	reset_cond = ['1','2']	

	## CONSTRUCTION OF THE AuGMEnT NETWORK
	S = np.shape(S_tr)[1]        # dimension of the input = number of possible stimuli
	A = np.shape(O_tr)[1]			     # dimension of the activity units = number of possible responses

	R = 4			     # dimension of the regular units
	M = 5 			     # dimension of the memory units
	
	# value parameters were taken from the 
	lamb = 0.2    			# synaptic tag decay 
	beta = 0.01			# weight update coefficient
	discount = 0.9			# discount rate for future rewards
	alpha = 1-lamb*discount 	# synaptic permanence
	eps = 0.025			# percentage of softmax modality for activity selection
	g = 4

	# reward settings
	rew_system = ['RL','PL','SRL']
	rew = 'PL'

	verb = 0
	
	if params_bool is None:
		do_training = True
		do_test = True
		do_weight_plots = True
		do_error_plots = True
		
	else:
		do_training = params_bool[0]
		do_test = params_bool[1]
		do_weight_plots = params_bool[2]	
		do_error_plots = params_bool[3]

	reg_vec=[]
	mem_vec=[]
	for i in range(R):
		reg_vec.append('R'+str(i+1))
	for i in range(M):
		mem_vec.append('M'+str(i+1))		


	model = AuGMEnT(S,R,M,A,alpha,beta,discount,eps,g,rew,dic_stim,dic_resp)

	## TRAINING
	folder = 'AuGMEnT/DATA'
	if do_training:	
		print('TRAINING...\n')
		g = 3
		E = model.training_ON_OFF(S_tr,O_tr,reset_cond,verb)
		str_err = folder+'/'+task+'_error.txt'
		np.savetxt(str_err, E)

		str_V_r = folder+'/'+task+'_weight_Vr.txt'
		np.savetxt(str_V_r, model.V_r)
		str_V_m = folder+'/'+task+'_weight_Vm.txt'
		np.savetxt(str_V_m, model.V_m)

		str_W_r = folder+'/'+task+'_weight_Wr.txt'
		np.savetxt(str_W_r, model.W_r)
		str_W_m = folder+'/'+task+'_weight_Wm.txt'
		np.savetxt(str_W_m, model.W_m)
		
		str_W_r_back = folder+'/'+task+'_weight_Wr_back.txt'
		np.savetxt(str_W_r_back, model.W_r_back)
		str_W_m_back = folder+'/'+task+'_weight_Wm_back.txt'
		np.savetxt(str_W_m_back, model.W_m_back)

		print('Saved model.')
	else:
		str_W_m_back = folder+'/'+task+'_error.txt'
		E = np.loadtxt(str_err)

		str_V_r = folder+'/'+task+'_weight_Vr.txt'
		model.V_r = np.loadtxt(str_V_r)
		str_V_m = folder+'/'+task+'_weight_Vm.txt'
		model.V_m = np.loadtxt(str_V_m)

		str_W_r = folder+'/'+task+'_weight_Wr.txt'
		model.W_r = np.loadtxt(str_W_r)
		str_W_m = folder+'/'+task+'_weight_Wm.txt'
		model.W_m =np.loadtxt(str_W_m)

		
		str_W_r_back = folder+'/'+task+'_weight_Wr_back.txt'
		model.W_r_back = np.loadtxt(str_W_r_back)
		str_W_m_back = folder+'/'+task+'_weight_Wm_back.txt'
		model.W_m_back = np.loadtxt(str_W_m_back)

		print('Loaded model from disk.')
			
	print('\n------------------------------------------------------------------------------------------					\n-----------------------------------------------------------------------------------------------------\n')

	## TEST
	if do_test:
		print('TEST...\n')
		g = 10
		model.test_ON_OFF(S_test,O_test,reset_cond,0)


	## PLOTS
	# plot of the memory weights
	folder = 'AuGMEnT/IMAGES'
	
	fontTitle = 26
	fontTicks = 22
	fontLabel = 22

	if do_weight_plots:

		fig = plt.figure(figsize=(30,8))

		W = model.V_r
		plt.subplot(1,3,1)
		plt.pcolor(np.flipud(W),edgecolors='k', linewidths=1)
		plt.set_cmap('gray_r')			
		plt.colorbar()
		tit = 'ASSOCIATION WEIGHTS'
		plt.title(tit,fontweight="bold",fontsize=fontTitle)			
		plt.yticks(np.linspace(0.5,S-0.5,S,endpoint=True),np.flipud(cues_vec),fontsize=fontTicks)
		plt.xticks(np.linspace(0.5,R-0.5,R,endpoint=True),reg_vec,fontsize=fontTicks)

		plt.subplot(1,3,2)
		X = model.V_m
		plt.pcolor(np.flipud(X),edgecolors='k', linewidths=1)
		plt.set_cmap('gray_r')			
		plt.colorbar()
		tit = 'MEMORY WEIGHTS'
		plt.title(tit,fontweight="bold",fontsize=fontTitle)
		plt.yticks(np.linspace(0.5,2*S-0.5,2*S,endpoint=True),np.flipud(cues_vec_tot),fontsize=fontTicks)
		plt.xticks(np.linspace(0.5,M-0.5,M,endpoint=True),mem_vec,fontsize=fontTicks)
		
		WW = np.concatenate((model.W_r, model.W_m),axis=0)
		plt.subplot(1,3,3)
		plt.pcolor(np.flipud(WW),edgecolors='k', linewidths=1)
		plt.set_cmap('gray_r')			
		plt.colorbar()
		tit = 'OUTPUT WEIGHTS'
		plt.title(tit,fontweight="bold",fontsize=fontTitle)			
		plt.yticks(np.linspace(0.5,(M+R)-0.5,M+R,endpoint=True),np.flipud(np.concatenate((reg_vec,mem_vec))),fontsize=fontTicks)
		plt.xticks(np.linspace(0.5,A-0.5,A,endpoint=True),pred_vec,fontsize=fontTicks)
		plt.show()

		savestr = folder+'/'+task+'_weights_'+rew+'.png'
		fig.savefig(savestr)
		
		
	if do_error_plots:

		N = len(E)
		bin = round(N*0.02)
		END = np.floor(N/bin).astype(int)
		E = E[:END*bin]
		N = len(E)

		E_bin = np.reshape(E,(-1,bin))
		E_bin = np.sum(E_bin,axis=1)
		E_cum = np.cumsum(E)
		E_norm = 100*E_cum/(np.arange(N)+1)
		C = np.where(E==0,1,0)
		C_cum = 100*np.cumsum(C)/(np.arange(N)+1)

		figE = plt.figure(figsize=(20,8))
		N_round = np.around(N/1000).astype(int)*1000
		
		plt.subplot(1,2,1)
		plt.bar(bin*np.arange(len(E_bin))/2,E_bin,width=bin/6,color='green',edgecolor='black', alpha=0.6)
		if conv_iter!=0:		
			plt.axvline(x=conv_iter/2, linewidth=5, color='g')
		tit = 'AX: Training Convergence'
		plt.title(tit,fontweight="bold",fontsize=fontTitle)		
		plt.xlabel('Training Trials',fontsize=fontLabel)
		plt.ylabel('Number of Errors per bin',fontsize=fontLabel)		
		plt.xticks(np.linspace(0,N_round/2,5,endpoint=True),fontsize=fontTicks)	
		plt.yticks(fontsize=fontTicks)	
		text = 'Bin = '+str(bin)
		plt.figtext(x=0.38,y=0.78,s=text,fontsize=fontLabel,bbox={'facecolor':'white', 'alpha':0.5, 'pad':10})

		plt.subplot(1,2,2)
		plt.plot(np.arange(N)/6, E_cum, color='green',linewidth=7, alpha=0.6)
		plt.axvline(x=4492/6, linewidth=5, ls='dashed', color='b')
		if conv_iter!=0:		
			plt.axvline(x=conv_iter/6, linewidth=5, color='g')
		tit = 'AX: Cumulative Training Error'
		plt.title(tit,fontweight="bold",fontsize=fontTitle)			
		plt.xticks(np.linspace(0,N_round/2,5,endpoint=True),fontsize=fontTicks)
		plt.yticks(fontsize=fontTicks)
		plt.xlabel('Training Trials',fontsize=fontLabel)
		plt.ylabel('Cumulative Error',fontsize=fontLabel)
		plt.show()

		savestr = folder+'/'+task+'_error_'+rew+'.png'		
		if M==0:
			savestr = folder+'/'+task+'_error_'+rew+'_nomemory.png'
		figE.savefig(savestr)


#########################################################################################################################################
#######################   TASK 1-2 AX
#########################################################################################################################################

def AuGMEnT_task_1_2AX(params_bool, params_task):
	
	from TASKS.task_1_2AX import data_construction				# BE CAREFULLLLL
	task = '12-AX'
	
	cues_vec = ['1','2','A','B','C','X','Y','Z']
	cues_vec_tot = ['1+','2+','A+','B+','C+','X+','Y+','Z+','1-','2-','A-','B-','C-','X-','Y-','Z-']
	pred_vec = ['L','R']

	if params_task is None:
		N = 20000
		p_c = 0.5
		perc_tr = 0.8
	else:
		N = int(params_task[0]) if params_task[0] != '' else 20000
		p_c = float(params_task[1]) if params_task[1] != '' else 0.5
		perc_tr = float(params_task[2]) if params_task[2] != '' else 0.8

	print('Dataset construction...')
	S_tr,O_tr,S_test,O_test,dic_stim,dic_resp = data_construction(N,p_c,perc_tr,model='0')

	reset_cond = ['1','2']	

	S = np.shape(S_tr)[1]        # dimension of the input = number of possible stimuli
	A = np.shape(O_tr)[1]			     # dimension of the activity units = number of possible responses

	## CONSTRUCTION OF THE AuGMEnT NETWORK
	R = 3			     # dimension of the regular units
	M = 4			     # dimension of the memory units
	
	# value parameters were taken from the 
	lamb = 0.2    			# synaptic tag decay 
	beta = 0.05			# weight update coefficient
	discount = 0.9			# discount rate for future rewards
	alpha = 1-lamb*discount 	# synaptic permanence
	eps = 0.025			# percentage of softmax modality for activity selection
	g = 4

	# reward settings
	
	rew_system = ['RL','PL','SRL']
	rew = 'SRL'

	verb = 0

	if params_bool is None:
		do_training = True
		do_test = True
		do_weight_plots = True
		do_error_plots = True
		
	else:
		do_training = params_bool[0]
		do_test = params_bool[1]
		do_weight_plots = params_bool[2]	
		do_error_plots = params_bool[3]

	reg_vec=[]
	mem_vec=[]
	for i in range(R):
		reg_vec.append('R'+str(i+1))
	for i in range(M):
		mem_vec.append('M'+str(i+1))		


	model = AuGMEnT(S,R,M,A,alpha,beta,discount,eps,g,rew,dic_stim,dic_resp)

	## TRAINING
	folder = 'AuGMEnT/DATA'
	if do_training:	
		print('TRAINING...\n')
	
		E,conv_iter = model.training_ON_OFF(S_tr,O_tr,reset_cond,verb)
		str_err = folder+'/'+task+'_error_units'+str(M)+'.txt'
		np.savetxt(str_err, E)
		str_conv = folder+'/'+task+'_conv_units'+str(M)+'.txt'
		np.savetxt(str_conv, conv_iter)

		str_V_r = folder+'/'+task+'_weight_Vr_units'+str(M)+'.txt'
		np.savetxt(str_V_r, model.V_r)
		str_V_m = folder+'/'+task+'_weight_Vm_units'+str(M)+'.txt'
		np.savetxt(str_V_m, model.V_m)

		str_W_r = folder+'/'+task+'_weight_Wr_units'+str(M)+'.txt'
		np.savetxt(str_W_r, model.W_r)
		str_W_m = folder+'/'+task+'_weight_Wm_units'+str(M)+'.txt'
		np.savetxt(str_W_m, model.W_m)
		
		str_W_r_back = folder+'/'+task+'_weight_Wr_back_units'+str(M)+'.txt'
		np.savetxt(str_W_r_back, model.W_r_back)
		str_W_m_back = folder+'/'+task+'_weight_Wm_back_units'+str(M)+'.txt'
		np.savetxt(str_W_m_back, model.W_m_back)

		print('Saved model.')
	else:
		str_err = folder+'/'+task+'_error_units'+str(M)+'.txt'
		E = np.loadtxt(str_err)
		str_conv = folder+'/'+task+'_conv_units'+str(M)+'.txt'
		conv_iter = np.loadtxt(str_conv)
		print(conv_iter)

		str_V_r = folder+'/'+task+'_weight_Vr_units'+str(M)+'.txt'
		model.V_r = np.loadtxt(str_V_r)
		str_V_m = folder+'/'+task+'_weight_Vm_units'+str(M)+'.txt'
		model.V_m = np.loadtxt(str_V_m)

		str_W_r = folder+'/'+task+'_weight_Wr_units'+str(M)+'.txt'
		model.W_r = np.loadtxt(str_W_r)
		str_W_m = folder+'/'+task+'_weight_Wm_units'+str(M)+'.txt'
		model.W_m =np.loadtxt(str_W_m)

		
		str_W_r_back = folder+'/'+task+'_weight_Wr_back_units'+str(M)+'.txt'
		model.W_r_back = np.loadtxt(str_W_r_back)
		str_W_m_back = folder+'/'+task+'_weight_Wm_back_units'+str(M)+'.txt'
		model.W_m_back = np.loadtxt(str_W_m_back)

		print('Loaded model from disk.')
			
	print('\n------------------------------------------------------------------------------------------					\n-----------------------------------------------------------------------------------------------------\n')

	## TEST
	if do_test:
		print('TEST...\n')
		model.test_ON_OFF(S_test,O_test,reset_cond,0)
	

	## PLOTS
	# plot of the memory weights
	folder = 'AuGMEnT/IMAGES'
	
	fontTitle = 26
	fontTicks = 22
	fontLabel = 22
	
	if do_weight_plots:

		figW = plt.figure(figsize=(30,8))

		W = model.V_r
		plt.subplot(1,3,1)
		plt.pcolor(np.flipud(W),edgecolors='k', linewidths=1)
		plt.set_cmap('Greens')			
		plt.colorbar()
		tit = 'ASSOCIATION WEIGHTS'
		plt.title(tit,fontweight="bold",fontsize=fontTitle)			
		plt.yticks(np.linspace(0.5,S-0.5,S,endpoint=True),np.flipud(cues_vec),fontsize=fontTicks)
		plt.xticks(np.linspace(0.5,R-0.5,R,endpoint=True),reg_vec,fontsize=fontTicks)

		plt.subplot(1,3,2)
		X = model.V_m
		plt.pcolor(np.flipud(X),edgecolors='k', linewidths=1)
		plt.set_cmap('Greens')			
		plt.colorbar()
		tit = 'MEMORY WEIGHTS'
		plt.title(tit,fontweight="bold",fontsize=fontTitle)
		plt.yticks(np.linspace(0.5,2*S-0.5,2*S,endpoint=True),np.flipud(cues_vec_tot),fontsize=fontTicks)
		plt.xticks(np.linspace(0.5,M-0.5,M,endpoint=True),mem_vec,fontsize=fontTicks)
		
		WW = np.concatenate((model.W_r, model.W_m),axis=0)
		plt.subplot(1,3,3)
		plt.pcolor(np.flipud(WW),edgecolors='k', linewidths=1)
		plt.set_cmap('Greens')			
		plt.colorbar()
		tit = 'OUTPUT WEIGHTS'
		plt.title(tit,fontweight="bold",fontsize=fontTitle)			
		plt.yticks(np.linspace(0.5,(M+R)-0.5,M+R,endpoint=True),np.flipud(np.concatenate((reg_vec,mem_vec))),fontsize=fontTicks)
		plt.xticks(np.linspace(0.5,A-0.5,A,endpoint=True),pred_vec,fontsize=fontTicks)
		plt.show()

		savestr = folder+'/'+task+'_weights_'+rew+'.png'
		if M==0:
			savestr = folder+'/'+task+'_weights_'+rew+'_nomemory.png'		
		figW.savefig(savestr)
		
		
	if do_error_plots:

		N = len(E)
		bin = round(N*0.02)
		END = np.floor(N/bin).astype(int)
		E = E[:END*bin]
		N = len(E)

		E_bin = np.reshape(E,(-1,bin))
		E_bin = np.sum(E_bin,axis=1)
		E_cum = np.cumsum(E)
		E_norm = 100*E_cum/(np.arange(N)+1)
		C = np.where(E==0,1,0)
		C_cum = 100*np.cumsum(C)/(np.arange(N)+1)

		figE = plt.figure(figsize=(20,8))
		N_round = np.around(N/1000).astype(int)*1000
		
		plt.subplot(1,2,1)
		plt.bar(bin*np.arange(len(E_bin))/6,E_bin,width=bin/6,color='green',edgecolor='black', alpha=0.6)
		plt.axvline(x=4492/6, linewidth=5, ls='dashed', color='b')
		if conv_iter!=0:		
			plt.axvline(x=conv_iter/6, linewidth=5, color='g')
		tit = '12AX: Training Convergence'
		plt.title(tit,fontweight="bold",fontsize=fontTitle)		
		plt.xlabel('Training Trials',fontsize=fontLabel)
		plt.ylabel('Number of Errors per bin',fontsize=fontLabel)		
		plt.xticks(np.linspace(0,N_round/6,5,endpoint=True),fontsize=fontTicks)	
		plt.yticks(fontsize=fontTicks)	
		text = 'Bin = '+str(bin)
		plt.figtext(x=0.38,y=0.78,s=text,fontsize=fontLabel,bbox={'facecolor':'white', 'alpha':0.5, 'pad':10})

		plt.subplot(1,2,2)
		plt.plot(np.arange(N)/6, E_cum, color='green',linewidth=7, alpha=0.6)
		plt.axvline(x=4492/6, linewidth=5, ls='dashed', color='b')
		if conv_iter!=0:		
			plt.axvline(x=conv_iter/6, linewidth=5, color='g')
		tit = '12AX: Cumulative Training Error'
		plt.title(tit,fontweight="bold",fontsize=fontTitle)			
		plt.xticks(np.linspace(0,N_round/6,5,endpoint=True),fontsize=fontTicks)
		plt.yticks(fontsize=fontTicks)
		plt.xlabel('Training Trials',fontsize=fontLabel)
		plt.ylabel('Cumulative Error',fontsize=fontLabel)
		plt.show()

		savestr = folder+'/'+task+'_error_'+rew+'.png'		
		if M==0:
			savestr = folder+'/'+task+'_error_'+rew+'_nomemory.png'
		figE.savefig(savestr)




#########################################################################################################################################
#######################   TASK SACCADES/ANTI-SACCADES
#########################################################################################################################################

def AuGMEnT_task_saccades(params_bool, params_task):
	
	from TASKS.task_saccades import data_construction
	task = 'saccade'

	cues_vec = ['P','A','L','R']
	cues_vec_tot = ['P+','A+','L+','R+','P-','A-','L-','R-']
	pred_vec = ['L','F','R']

	if params_task is None:
		#N_trial = 10000
		N_trial = 20000
		perc_tr = 0.8
	else:
		N_trial = int(params_task[0]) if params_task[0] != '' else 20000
		perc_tr = float(params_task[1]) if params_task[1] != '' else 0.8

	S_tr,O_tr,S_test,O_test,dic_stim,dic_resp = data_construction(N=N_trial,perc_training=perc_tr,model='0')

	reset_cond = ['empty']	

	S = np.shape(S_tr)[1]        # dimension of the input = number of possible stimuli
	A = np.shape(O_tr)[1] 			     # dimension of the activity units = number of possible responses

	## CONSTRUCTION OF THE AuGMEnT NETWORK
	R = 3			     # dimension of the regular units
	M = 4 			     # dimension of the memory units
	
	# value parameters were taken from the 
	lamb = 0.2    			# synaptic tag decay 
	beta = 0.1			# weight update coefficient
	discount = 0.9			# discount rate for future rewards
	alpha = 1-lamb*discount 	# synaptic permanence
	eps = 0.025			# percentage of softmax modality for activity selection

	g = 10

	# reward settings
	rew_system = ['RL','PL','SRL']
	rew = 'RL'

	verb = 0
		
	if params_bool is None:
		do_training = True
		do_test = True
		do_weight_plots = True
		do_error_plots = True
		
	else:
		do_training = params_bool[0]
		do_test = params_bool[1]
		do_weight_plots = params_bool[2]	
		do_error_plots = params_bool[3]

	shape_fac = 0.2

	reg_vec = []
	mem_vec = []
	for i in range(R):
		reg_vec.append('R'+str(i+1))
	for i in range(M):
		mem_vec.append('M'+str(i+1))		


	model = AuGMEnT(S,R,M,A,alpha,beta,discount,eps,g,rew,dic_stim,dic_resp)

	## TRAINING
	folder = 'AuGMEnT/DATA'
	if do_training:	
		print('TRAINING...\n')
		training_trial = np.round(N_trial*perc_tr).astype(int)
		E_fix,E_go,conv_iter = model.training_saccade(training_trial,S_tr,O_tr,reset_cond,verb,shape_fac)	
		
		str_err = folder+'/'+task+'_error_fix.txt'
		np.savetxt(str_err, E_fix)
		str_err = folder+'/'+task+'_error_go.txt'
		np.savetxt(str_err, E_go)
		str_conv = folder+'/'+task+'_conv.txt'
		np.savetxt(str_conv, conv_iter)

		str_V_r = folder+'/'+task+'_weight_Vr.txt'
		np.savetxt(str_V_r, model.V_r)
		str_V_m = folder+'/'+task+'_weight_Vm.txt'
		np.savetxt(str_V_m, model.V_m)

		str_W_r = folder+'/'+task+'_weight_Wr.txt'
		np.savetxt(str_W_r, model.W_r)
		str_W_m = folder+'/'+task+'_weight_Wm.txt'
		np.savetxt(str_W_m, model.W_m)
		
		str_W_r_back = folder+'/'+task+'_weight_Wr_back.txt'
		np.savetxt(str_W_r_back, model.W_r_back)
		str_W_m_back = folder+'/'+task+'_weight_Wm_back.txt'
		np.savetxt(str_W_m_back, model.W_m_back)

		print('Saved model.')
	else:
		str_err = folder+'/'+task+'_error_fix.txt'
		E_fix = np.loadtxt(str_err)
		str_err = folder+'/'+task+'_error_go.txt'
		E_go = np.loadtxt(str_err)
		str_conv = folder+'/'+task+'_conv.txt'
		conv_iter = np.loadtxt(str_conv)
		print(conv_iter)

		str_V_r = folder+'/'+task+'_weight_Vr.txt'
		model.V_r = np.loadtxt(str_V_r)
		str_V_m = folder+'/'+task+'_weight_Vm.txt'
		model.V_m = np.loadtxt(str_V_m)

		str_W_r = folder+'/'+task+'_weight_Wr.txt'
		model.W_r = np.loadtxt(str_W_r)
		str_W_m = folder+'/'+task+'_weight_Wm.txt'
		model.W_m =np.loadtxt(str_W_m)
		
		str_W_r_back = folder+'/'+task+'_weight_Wr_back.txt'
		model.W_r_back = np.loadtxt(str_W_r_back)
		str_W_m_back = folder+'/'+task+'_weight_Wm_back.txt'
		model.W_m_back = np.loadtxt(str_W_m_back)

		print('Loaded model from disk.')
			
	print('\n------------------------------------------------------------------------------------------					\n-----------------------------------------------------------------------------------------------------\n')

	## TEST
	if do_test:
		print('TEST...\n')
		model.test_saccade(S_test,O_test,reset_cond,0)

	## PLOTS
	# plot of the memory weights
	folder = 'AuGMEnT/IMAGES'

	fontTitle = 26
	fontTicks = 22
	fontLabel = 22

	if do_weight_plots:

		figW = plt.figure(figsize=(30,8))

		W = model.V_r
		plt.subplot(1,3,1)
		plt.pcolor(np.flipud(W),edgecolors='k', linewidths=1)
		plt.set_cmap('Greens')			
		plt.colorbar()
		tit = 'ASSOCIATION WEIGHTS'
		plt.title(tit,fontweight="bold",fontsize=fontTitle)			
		plt.yticks(np.linspace(0.5,S-0.5,S,endpoint=True),np.flipud(cues_vec),fontsize=fontTicks)
		plt.xticks(np.linspace(0.5,R-0.5,R,endpoint=True),reg_vec,fontsize=fontTicks)

		plt.subplot(1,3,2)
		X = model.V_m
		plt.pcolor(np.flipud(X),edgecolors='k', linewidths=1)
		plt.set_cmap('Greens')			
		plt.colorbar()
		tit = 'MEMORY WEIGHTS'
		plt.title(tit,fontweight="bold",fontsize=fontTitle)
		plt.yticks(np.linspace(0.5,2*S-0.5,2*S,endpoint=True),np.flipud(cues_vec_tot),fontsize=fontTicks)
		plt.xticks(np.linspace(0.5,M-0.5,M,endpoint=True),mem_vec,fontsize=fontTicks)
		
		WW = np.concatenate((model.W_r, model.W_m),axis=0)
		plt.subplot(1,3,3)
		plt.pcolor(np.flipud(WW),edgecolors='k', linewidths=1)
		plt.set_cmap('Greens')			
		plt.colorbar()
		tit = 'OUTPUT WEIGHTS'
		plt.title(tit,fontweight="bold",fontsize=fontTitle)			
		plt.yticks(np.linspace(0.5,(M+R)-0.5,M+R,endpoint=True),np.flipud(np.concatenate((reg_vec,mem_vec))),fontsize=fontTicks)
		plt.xticks(np.linspace(0.5,A-0.5,A,endpoint=True),pred_vec,fontsize=fontTicks)
		plt.show()	
		
		savestr = folder+'/'+task+'_weights_'+rew+'.png'
		if M==0:
			savestr = folder+'/'+task+'_weights_'+rew+'_nomemory.png'	
		figW.savefig(savestr)
	
	if do_error_plots:

		N = len(E_fix)
		bin = round(N*0.02)
		print(bin)
		E_fix_bin = np.reshape(E_fix,(-1,bin))
		E_fix_bin = np.sum(E_fix_bin,axis=1)
		E_fix_cum = np.cumsum(E_fix)
		E_fix_norm = 100*E_fix_cum/(np.arange(N)+1)
		C_fix = np.where(E_fix==0,1,0)
		C_fix_cum = 100*np.cumsum(C_fix)/(np.arange(N)+1)

		E_go_bin = np.reshape(E_go,(-1,bin))
		E_go_bin = np.sum(E_go_bin,axis=1)
		E_go_cum = np.cumsum(E_go)
		E_go_norm = 100*E_go_cum/(np.arange(N)+1)
		C_go = np.where(E_go==0,1,0)
		C_go_cum = 100*np.cumsum(C_go)/(np.arange(N)+1)

		figE = plt.figure(figsize=(22,8))
		plt.subplot(1,2,1)
		plt.bar(bin*np.arange(len(E_go_bin)),E_go_bin,width=bin,color='green',edgecolor='black',label='go',alpha=0.6)
		plt.bar(bin*np.arange(len(E_fix_bin)),E_fix_bin,width=bin,color='orange',edgecolor='black',label='fix',alpha=0.6)
		plt.axvline(x=4100, linewidth=5, ls='dashed', color='green')
		plt.axvline(x=225, linewidth=5, ls='dashed', color='orange')
		if conv_iter!=0:
			plt.axvline(x=conv_iter, linewidth=5, color='green')
		tit = 'SAS: Training Convergence'
		plt.title(tit,fontweight="bold",fontsize=fontTitle)		
		plt.xlabel('Training Trials',fontsize=fontLabel)
		plt.ylabel('Number of Errors per bin',fontsize=fontLabel)		
		plt.xticks(np.linspace(0,N,5,endpoint=True),fontsize=fontTicks)	
		plt.yticks(fontsize=fontTicks)	
		text = 'Bin = '+str(bin)
		plt.figtext(x=0.37,y=0.78,s=text,fontsize=fontLabel,bbox={'facecolor':'white', 'alpha':0.5, 'pad':10})

		plt.subplot(1,2,2)
		plt.axvline(x=225, linewidth=5, ls='dashed', color='orange')
		plt.axvline(x=4100, linewidth=5, ls='dashed', color='green')
		if conv_iter!=0:
			plt.axvline(x=conv_iter, linewidth=5, color='green')
		plt.plot(np.arange(N), E_go_cum, color='green',linewidth=7,label='go',alpha=0.6)
		plt.plot(np.arange(N), E_fix_cum, color='orange',linewidth=7,label='fix',alpha=0.6)
		tit = 'SAS: Cumulative Training Error'
		plt.title(tit,fontweight="bold",fontsize=fontTitle)			
		plt.xticks(np.linspace(0,N,5,endpoint=True),fontsize=fontTicks)
		plt.yticks(fontsize=fontTicks)
		plt.xlabel('Training Trials',fontsize=fontLabel)
		plt.ylabel('Cumulative Error',fontsize=fontLabel)
		plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,fontsize=fontTitle)
		plt.show()

		savestr = folder+'/'+task+'_error_'+rew+'.png'
		if M==0:
			savestr = folder+'/'+task+'_error_'+rew+'_nomemory.png'		
		figE.savefig(savestr)



#########################################################################################################################################
#######################   TASK SEQUENCE PREDICTION
#########################################################################################################################################

def AuGMEnT_task_seq_prediction(params_bool, params_task):
	
	from TASKS.task_seq_prediction import data_construction				# BE CAREFULLLLL
	
	task = 'sequence_prediction'

	cues_vec = ['A','B','C','X']
	cues_vec_tot = ['A+','B+','C+','X+','A-','B-','C-','X-']
	pred_vec = ['D','B','C','Y']

	if params_task is None:
		N = 20000
		p_c = 0.5
		perc_tr = 0.8
	else:
		N = int(params_task[0]) if params_task[0] != '' else 20000
		p_c = float(params_task[1]) if params_task[1] != '' else 0.5
		perc_tr = float(params_task[2]) if params_task[2] != '' else 0.8

	print('Dataset construction...')
	S_tr,O_tr,S_test,O_test,dic_stim,dic_resp = data_construction(N,p_c,perc_tr,model='0')

	reset_cond = ['A','X']	

	S = np.shape(S_tr)[1]        # dimension of the input = number of possible stimuli
	A = np.shape(O_tr)[1]			     # dimension of the activity units = number of possible responses

	## CONSTRUCTION OF THE AuGMEnT NETWORK
	R = 3			     # dimension of the regular units
	M = 4			     # dimension of the memory units
	
	# value parameters were taken from the 
	lamb = 0.2    			# synaptic tag decay 
	beta = 0.02			# weight update coefficient
	discount = 0.9			# discount rate for future rewards
	alpha = 1-lamb*discount 	# synaptic permanence
	eps = 0.025			# percentage of softmax modality for activity selection
	g = 4

	# reward settings
	
	rew_system = ['RL','PL','SRL','BRL']
	rew = 'SRL'

	verb = 0

	if params_bool is None:
		do_training = True
		do_test = True
		do_weight_plots = True
		do_error_plots = True
		
	else:
		do_training = params_bool[0]
		do_test = params_bool[1]
		do_weight_plots = params_bool[2]	
		do_error_plots = params_bool[3]

	reg_vec=[]
	mem_vec=[]
	for i in range(R):
		reg_vec.append('R'+str(i+1))
	for i in range(M):
		mem_vec.append('M'+str(i+1))		


	model = AuGMEnT(S,R,M,A,alpha,beta,discount,eps,g,rew,dic_stim,dic_resp)

	## TRAINING
	folder = 'AuGMEnT/DATA'
	if do_training:	
		print('TRAINING...\n')
	
		E,conv_iter = model.training_ON_OFF(S_tr,O_tr,reset_cond,verb)
		str_err = folder+'/'+task+'_error_units'+str(M)+'.txt'
		np.savetxt(str_err, E)
		str_conv = folder+'/'+task+'_conv_units'+str(M)+'.txt'
		np.savetxt(str_conv, conv_iter)

		str_V_r = folder+'/'+task+'_weight_Vr_units'+str(M)+'.txt'
		np.savetxt(str_V_r, model.V_r)
		str_V_m = folder+'/'+task+'_weight_Vm_units'+str(M)+'.txt'
		np.savetxt(str_V_m, model.V_m)

		str_W_r = folder+'/'+task+'_weight_Wr_units'+str(M)+'.txt'
		np.savetxt(str_W_r, model.W_r)
		str_W_m = folder+'/'+task+'_weight_Wm_units'+str(M)+'.txt'
		np.savetxt(str_W_m, model.W_m)
		
		str_W_r_back = folder+'/'+task+'_weight_Wr_back_units'+str(M)+'.txt'
		np.savetxt(str_W_r_back, model.W_r_back)
		str_W_m_back = folder+'/'+task+'_weight_Wm_back_units'+str(M)+'.txt'
		np.savetxt(str_W_m_back, model.W_m_back)

		print('Saved model.')
	else:
		str_err = folder+'/'+task+'_error_units'+str(M)+'.txt'
		E = np.loadtxt(str_err)
		str_conv = folder+'/'+task+'_conv_units'+str(M)+'.txt'
		conv_iter = np.loadtxt(str_conv)
		print(conv_iter)

		str_V_r = folder+'/'+task+'_weight_Vr_units'+str(M)+'.txt'
		model.V_r = np.loadtxt(str_V_r)
		str_V_m = folder+'/'+task+'_weight_Vm_units'+str(M)+'.txt'
		model.V_m = np.loadtxt(str_V_m)

		str_W_r = folder+'/'+task+'_weight_Wr_units'+str(M)+'.txt'
		model.W_r = np.loadtxt(str_W_r)
		str_W_m = folder+'/'+task+'_weight_Wm_units'+str(M)+'.txt'
		model.W_m =np.loadtxt(str_W_m)

		
		str_W_r_back = folder+'/'+task+'_weight_Wr_back_units'+str(M)+'.txt'
		model.W_r_back = np.loadtxt(str_W_r_back)
		str_W_m_back = folder+'/'+task+'_weight_Wm_back_units'+str(M)+'.txt'
		model.W_m_back = np.loadtxt(str_W_m_back)

		print('Loaded model from disk.')
			
	print('\n------------------------------------------------------------------------------------------					\n-----------------------------------------------------------------------------------------------------\n')

	## TEST
	if do_test:
		print('TEST...\n')
		model.test_ON_OFF(S_test,O_test,reset_cond,0)
	

	## PLOTS
	# plot of the memory weights
	folder = 'AuGMEnT/IMAGES'
	
	fontTitle = 26
	fontTicks = 22
	fontLabel = 22
	
	if do_weight_plots:

		figW = plt.figure(figsize=(30,8))

		W = model.V_r
		plt.subplot(1,3,1)
		plt.pcolor(np.flipud(W),edgecolors='k', linewidths=1)
		plt.set_cmap('Greens')			
		plt.colorbar()
		tit = 'ASSOCIATION WEIGHTS'
		plt.title(tit,fontweight="bold",fontsize=fontTitle)			
		plt.yticks(np.linspace(0.5,S-0.5,S,endpoint=True),np.flipud(cues_vec),fontsize=fontTicks)
		plt.xticks(np.linspace(0.5,R-0.5,R,endpoint=True),reg_vec,fontsize=fontTicks)

		plt.subplot(1,3,2)
		X = model.V_m
		plt.pcolor(np.flipud(X),edgecolors='k', linewidths=1)
		plt.set_cmap('Greens')			
		plt.colorbar()
		tit = 'MEMORY WEIGHTS'
		plt.title(tit,fontweight="bold",fontsize=fontTitle)
		plt.yticks(np.linspace(0.5,2*S-0.5,2*S,endpoint=True),np.flipud(cues_vec_tot),fontsize=fontTicks)
		plt.xticks(np.linspace(0.5,M-0.5,M,endpoint=True),mem_vec,fontsize=fontTicks)
		
		WW = np.concatenate((model.W_r, model.W_m),axis=0)
		plt.subplot(1,3,3)
		plt.pcolor(np.flipud(WW),edgecolors='k', linewidths=1)
		plt.set_cmap('Greens')			
		plt.colorbar()
		tit = 'OUTPUT WEIGHTS'
		plt.title(tit,fontweight="bold",fontsize=fontTitle)			
		plt.yticks(np.linspace(0.5,(M+R)-0.5,M+R,endpoint=True),np.flipud(np.concatenate((reg_vec,mem_vec))),fontsize=fontTicks)
		plt.xticks(np.linspace(0.5,A-0.5,A,endpoint=True),pred_vec,fontsize=fontTicks)
		plt.show()

		savestr = folder+'/'+task+'_weights_'+rew+'.png'
		if M==0:
			savestr = folder+'/'+task+'_weights_'+rew+'_nomemory.png'		
		figW.savefig(savestr)
		
		
	if do_error_plots:

		N = len(E)
		bin = round(N*0.02)
		END = np.floor(N/bin).astype(int)
		E = E[:END*bin]
		N = len(E)

		E_bin = np.reshape(E,(-1,bin))
		E_bin = np.sum(E_bin,axis=1)
		E_cum = np.cumsum(E)
		E_norm = 100*E_cum/(np.arange(N)+1)
		C = np.where(E==0,1,0)
		C_cum = 100*np.cumsum(C)/(np.arange(N)+1)

		figE = plt.figure(figsize=(20,8))
		N_round = np.around(N/1000).astype(int)*1000
		
		plt.subplot(1,2,1)
		plt.bar(bin*np.arange(len(E_bin))/3,E_bin,width=bin/3,color='green',edgecolor='black', alpha=0.6)
		if conv_iter!=0:		
			plt.axvline(x=conv_iter/3, linewidth=5, color='g')
		tit = 'Seq Predict: Training Convergence'
		plt.title(tit,fontweight="bold",fontsize=fontTitle)		
		plt.xlabel('Training Trials',fontsize=fontLabel)
		plt.ylabel('Number of Errors per bin',fontsize=fontLabel)		
		plt.xticks(np.linspace(0,N_round/3,5,endpoint=True),fontsize=fontTicks)	
		plt.yticks(fontsize=fontTicks)	
		text = 'Bin = '+str(bin)
		plt.figtext(x=0.38,y=0.78,s=text,fontsize=fontLabel,bbox={'facecolor':'white', 'alpha':0.5, 'pad':10})

		plt.subplot(1,2,2)
		plt.plot(np.arange(N)/3, E_cum, color='green',linewidth=7, alpha=0.6)
		if conv_iter!=0:		
			plt.axvline(x=conv_iter/3, linewidth=5, color='g')
		tit = 'Seq Predict: Cumulative Training Error'
		plt.title(tit,fontweight="bold",fontsize=fontTitle)			
		plt.xticks(np.linspace(0,N_round/3,5,endpoint=True),fontsize=fontTicks)
		plt.yticks(fontsize=fontTicks)
		plt.xlabel('Training Trials',fontsize=fontLabel)
		plt.ylabel('Cumulative Error',fontsize=fontLabel)
		plt.show()

		savestr = folder+'/'+task+'_error_'+rew+'.png'		
		if M==0:
			savestr = folder+'/'+task+'_error_'+rew+'_nomemory.png'
		figE.savefig(savestr)
