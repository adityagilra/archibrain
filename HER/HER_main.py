## MAIN FILE FOR HER TESTING
## Here are defined the settings for the HER architecture and for the tasks to test.
## AUTHOR: Marco Martinolli
## DATE: 07.03.2017

from HER_level import HER_level
from HER_base import HER_base
from HER_model import HER_arch

import numpy as np
import matplotlib 
import os
import activations as act

matplotlib.use('GTK3Cairo') 
from matplotlib import pyplot as plt
import pylab
import gc
from sys import version_info

task_dic ={'0':'task 1_2', 
           '1':'task AX_CPT',
	   '2':'task 12 AX-S', 
	   '3':'task 12 AX', 
	   '4':'saccade/antisaccade task'}

py3 = version_info[0] > 2 # creates boolean value for test that Python major version > 2
if py3:
  task_selection = input("\nPlease select a task: \n\t 0: task 1_2 \n\t 1: task AX_CPT \n\t 2: task 12 AX-S\n\t 3: task 12 AX\n\t 4: saccade/antisaccade task\nEnter id number:  ")
else:
  task_selection = raw_input("\nPlease select a task: \n\t 0: task 1_2 \n\t 1: task AX_CPT \n\t 2: task 12 AX-S\n\t 3: task 12 AX\n\t 4: saccade/antisaccade task\nEnter id number:  ")

print("\nYou have selected: ", task_dic[task_selection],'\n\n')


#########################################################################################################################################
#######################   TASK 1-2 
#########################################################################################################################################

if (task_selection=="0"):

	from TASKS.task_1_2 import data_construction

	[S_tr,O_tr,S_test,O_test,dic_stim,dic_resp] = data_construction(N=4000, p1=0.7, p2=0.3, perc_training=0.8)

	## CONSTRUCTION OF BASE LEVEL OF HER ARCHITECTURE
	S = np.shape(S_tr)[1]
	P = np.shape(O_tr)[1]
	alpha = 0.1
	beta = 12
	gamma = 12
	elig_decay_const = 0.3

	verb = 0
 
	L = HER_base(0,S, P, alpha, beta, gamma)

	print('TRAINING...')
	L.base_training(S_tr, O_tr)
	print(' DONE!\n')

	print('TEST....\n')
	L.base_test(S_test, O_test, dic_stim, dic_resp, elig_decay_const, verb)
	print('DONE!\n')

#########################################################################################################################################
#######################   TASK AX CPT
#########################################################################################################################################

elif (task_selection=="1"):
	
	from TASKS.task_AX_CPT import data_construction
	task = 'AX-CPT'
	np.random.seed(1234)

	cues_vec = ['A','B','X','Y']
	pred_vec = ['LC','LW','RC','RW']
	[S_tr,O_tr,S_test,O_test,dic_stim,dic_resp] = data_construction(N=40000, perc_target=0.2, perc_training=0.8)

	## CONSTRUCTION OF THE HER MULTI-LEVEL NETWORK
	NL = 2                      # number of levels (<= 3)
	S = np.shape(S_tr)[1]        # dimension of the input 
	P = np.shape(O_tr)[1] 	     # dimension of the prediction vector
	

	### Parameter values come from Table 1 of the Supplementary Material of the paper "Frontal cortex function derives from hierarchical predictive coding", W. Alexander, J. Brown
	learn_rate_vec = [0.075, 0.075]	# learning rates 
	beta_vec = [15, 15]             # gain parameter for memory dynamics
	elig_decay_vec = [0.1, 0.99]     # decay factors for eligibility trace
	bias_vec = [1,0.1]

	gamma = 5                       # taken from suggestion from "Extended Example of HER Model" 

	fontTitle = 22
	fontTicks = 20

	verb = 0

	learn_rule_WM = 'backprop'  	# options are: backprop or RL
	elig_update = 'pre'		# options are: pre or post (eligibility trace update respectively at the beginning or at the end of the training iteration)

	do_training = True 		# if false, it loads the weights from previous training
	do_test = True
	error_test = False		# see behavior of the network after training on specified scenarios

	do_weight_plots = True

	HER = HER_arch(NL,S,P,learn_rate_vec,beta_vec,gamma,elig_decay_vec)

	## TRAINING
	data_folder = 'DATA'
	if do_training:
		
		HER.training(S_tr,O_tr,bias_vec,learn_rule_WM,elig_update,dic_stim,dic_resp,verb)
		
		# save trained model
		for l in np.arange(NL):
			str_err = data_folder+'/'+task+'_error.txt'
			#np.savetxt(str_err, E)
			str_mem = data_folder+'/'+task+'_weights_memory_'+str(l)+'.txt'
			np.savetxt(str_mem, HER.H[l].X)
			str_pred = data_folder+'/'+task+'_weights_prediction_'+str(l)+'.txt'
			np.savetxt(str_pred, HER.H[l].W)		
		print("\nSaved model to disk.\n")
	
	else:
		for l in np.arange(NL):
			str_mem = data_folder+'/'+task+'_weights_memory_'+str(l)+'.txt'
			HER.H[l].X = np.loadtxt(str_mem)
			str_pred = data_folder+'/'+task+'_weights_prediction_'+str(l)+'.txt'
			HER.H[l].W = np.loadtxt(str_pred)	
		print("\nLoaded model from disk.\n")


	print('\n----------------------------------------------------\nTEST\n------------------------------------------------------\n')

	## TEST
	if do_test:
	
		HER.test(S_test,O_test,dic_stim,dic_resp,verb)


	## test of the error trend after training if X is presented to the system
	if error_test:	

		s = np.array([[0,0,1,0]])
		s_prime_vec = [ np.array([[1,0,0,0]]), np.array([[0,1,0,0]]), np.array([[0,0,1,0]]), np.array([[0,0,0,1]]) ] 		
		o_vec = [ np.array([[0,1,1,0]]), np.array([[1,0,0,1]]), np.array([[1,0,0,1]]), np.array([[1,0,0,1]]) ] 
		
		HER.H[0].r = s  		# X at base level
		p = act.linear(s, HER.H[0].W)
		print('PREDICTION VECTOR: \n',p,'\n')

		for s_prime,o in zip(s_prime_vec,o_vec): 
				
			HER.H[1].r = s_prime 	# stored at upper level
						
			p_prime = act.linear(s_prime, HER.H[1].W)
			HER.H[0].top_down(p_prime)
			m = p + act.linear(s, HER.H[0].P_prime)		

			resp_ind,p_resp = HER.H[0].compute_response(m)
			e, a = HER.H[0].compute_error(p, o, resp_ind)
			o_prime = HER.H[0].bottom_up(e)
			e_prime = HER.H[1].compute_error(p_prime, o_prime)		
			
			print('r0: ', dic_stim[repr(s.astype(int))],'\t r1:',dic_stim[repr(s_prime.astype(int))],'\t outcome: ',dic_resp[repr(o.astype(int))],'\t response: ',dic_resp[repr(resp_ind)],'\n')
			print('Prediction (level 1): \n', p_prime,'\n')
			print('Error (level 1): \n', np.reshape(e_prime,(S,-1)) )
			print('Modulated Prediction: ', m)			

			print('----------------------------------------------------------------')
	
	## PLOTS
	# plot of the memory weights
	image_folder = 'IMAGES'
	if do_weight_plots:
		fig1 = plt.figure(figsize=(10*NL,8))
		for l in np.arange(NL):
			X = HER.H[l].X
			plt.subplot(1,NL,l+1)
			plt.pcolor(np.flipud(X),edgecolors='k', linewidths=1)
			plt.set_cmap('gray_r')			
			plt.colorbar()
			tit = 'MEMORY WEIGHTS: Level '+str(l)
			plt.title(tit,fontweight="bold",fontsize=fontTitle)
			plt.xticks(np.linspace(0.5,S-0.5,4,endpoint=True),cues_vec,fontsize=fontTicks)
			plt.yticks(np.linspace(0.5,S-0.5,4,endpoint=True),np.flipud(cues_vec),fontsize=fontTicks)
		plt.show()
		savestr = image_folder+'/'+task+'_weights_memory.png'
		fig1.savefig(savestr)
	
		fig2 = plt.figure(figsize=(10*NL,8))
		for l in np.arange(NL):	
			W = HER.H[l].W
			plt.subplot(1,NL,l+1)
			plt.pcolor(np.flipud(W),edgecolors='k', linewidths=1)
			plt.set_cmap('gray_r')			
			plt.colorbar()
			tit = 'PREDICTION WEIGHTS: Level '+str(l)
			plt.title(tit,fontweight="bold",fontsize=fontTitle)
			if l==0:			
				plt.xticks(np.linspace(0.5,np.shape(W)[1]-0.5,4,endpoint=True),pred_vec,fontsize=fontTicks)
			else:
				dx = np.shape(W)[1]/(2*S)
				plt.xticks(np.linspace(dx,np.shape(W)[1]-dx,4,endpoint=True),cues_vec,fontsize=fontTicks)
				plt.yticks(np.linspace(0.5,S-0.5,4,endpoint=True),np.flipud(cues_vec),fontsize=fontTicks)
		plt.show()
		savestr = image_folder+'/'+task+'_weights_prediction.png'
		fig2.savefig(savestr)		

#########################################################################################################################################
#######################   TASK 1-2 AX - S
#########################################################################################################################################

elif (task_selection=="2"):
	
	from TASKS.task_1_2AX_S import data_construction
	
	task = '12_AX_S'
	np.random.seed(1234)

	cues_vec = ['1','2','AX','AY','BX','BY']
	pred_vec = ['LC','LW','RC','RW']
	[S_tr,O_tr,S_test,O_test,dic_stim,dic_resp] = data_construction(N=100000, p_digit=0.1, p_wrong=0.15, p_correct=0.25, perc_training=0.9)

	## CONSTRUCTION OF THE HER MULTI-LEVEL NETWORK
	NL = 2                       # number of levels (<= 3)
	S = np.shape(S_tr)[1]        # dimension of the input 
	P = np.shape(O_tr)[1] 	     # dimension of the prediction vector
	

	### Parameter values come from Table 1 of the Supplementary Material of the paper "Frontal cortex function derives from hierarchical predictive coding", W. Alexander, J. Brown
	learn_rate_vec = [0.1, 0.02, 0.02]	# learning rates 
	beta_vec = [12, 12, 12]                 # gain parameter for memory dynamics
	gamma = 12                              # gain parameter for response making
	elig_decay_vec = [0.3,0.5,0.9]          # decay factors for eligibility trace

	learn_rule_WM = 'backprop'
	elig_update = 'inter'

	do_training = True 		# if false, it loads the weights from previous training
	do_test = True

	do_weight_plots = True

	fontTitle = 22
	fontTicks = 20

	verb = 1

	HER = HER_arch(NL,S,P,learn_rate_vec,beta_vec,gamma,elig_decay_vec)

	## TRAINING
	data_folder='DATA'
	if do_training:
		HER.training(S_tr,O_tr,learn_rule_WM,elig_update,dic_stim,dic_resp)
			
		# save trained model
		for l in np.arange(NL):
			str_err = data_folder+'/'+task+'_error.txt'
			#np.savetxt(str_err, E)
			str_mem = data_folder+'/'+task+'_weights_memory_'+str(l)+'.txt'
			np.savetxt(str_mem, HER.H[l].X)
			str_pred = data_folder+'/'+task+'_weights_prediction_'+str(l)+'.txt'
			np.savetxt(str_pred, HER.H[l].W)		
		print("\nSaved model to disk.\n")
	
	else:
		for l in np.arange(NL):
			str_mem = data_folder+'/'+task+'_weights_memory_'+str(l)+'.txt'
			HER.H[l].X = np.loadtxt(str_mem)
			str_pred = data_folder+'/'+task+'_weights_prediction_'+str(l)+'.txt'
			HER.H[l].W = np.loadtxt(str_pred)	
		print("\nLoaded model from disk.\n")


	print('\n----------------------------------------------------\nTEST\n------------------------------------------------------\n')

	## TEST
	if do_test:
		HER.test(S_test,O_test,dic_stim,dic_resp,verb)

	## PLOTS
	# plot of the memory weights
	image_folder = 'IMAGES'
	if do_weight_plots:
		fig1 = plt.figure(figsize=(10*NL,8))
		for l in np.arange(NL):
			X = HER.H[l].X
			plt.subplot(1,NL,l+1)
			plt.pcolor(np.flipud(X),edgecolors='k', linewidths=1)
			plt.set_cmap('gray_r')			
			plt.colorbar()
			tit = 'MEMORY WEIGHTS: Level '+str(l)
			plt.title(tit,fontweight="bold",fontsize=fontTitle)
			plt.xticks(np.linspace(0.5,S-0.5,S,endpoint=True),cues_vec,fontsize=fontTicks)
			plt.yticks(np.linspace(0.5,S-0.5,S,endpoint=True),np.flipud(cues_vec),fontsize=fontTicks)
		plt.show()
		savestr = image_folder+'/'+task+'_weights_memory.png'
		fig1.savefig(savestr)		

	
		fig2 = plt.figure(figsize=(10*NL,8))
		for l in np.arange(NL):	
			W = HER.H[l].W
			plt.subplot(1,NL,l+1)
			plt.pcolor(np.flipud(W),edgecolors='k', linewidths=1)
			plt.set_cmap('gray_r')			
			plt.colorbar()
			tit = 'PREDICTION WEIGHTS: Level '+str(l)
			plt.title(tit,fontweight="bold",fontsize=fontTitle)
			if l==0:			
				plt.xticks(np.linspace(0.5,np.shape(W)[1]-0.5,P,endpoint=True),pred_vec,fontsize=fontTicks)
			else:
				dx = np.shape(W)[1]/(2*S)
				plt.xticks(np.linspace(dx,np.shape(W)[1]-dx,S,endpoint=True),cues_vec,fontsize=fontTicks)
			plt.yticks(np.linspace(0.5,S-0.5,S,endpoint=True),np.flipud(cues_vec),fontsize=fontTicks)
		plt.show()
		savestr = image_folder+'/'+task+'_weights_prediction.png'
		fig2.savefig(savestr)		


#########################################################################################################################################
#######################   TASK 1-2 AX
#########################################################################################################################################

elif (task_selection=="3"):
	
	from TASKS.task_1_2AX import data_construction
	
	task = '12_AX'

	cues_vec = ['1','2','A','B','X','Y','C','Z']
	pred_vec = ['LC','LW','RC','RW']
	np.random.seed(1234)

	N = 8000
	p_c = 0.5
	p_tr = 0.8
	[S_tr,O_tr,S_test,O_test,dic_stim,dic_resp] = data_construction(N,p_c,p_tr)

	## CONSTRUCTION OF THE HER MULTI-LEVEL NETWORK
	NL = 3                       # number of levels (<= 3)
	S = np.shape(S_tr)[1]        # dimension of the input 
	P = np.shape(O_tr)[1] 	     # dimension of the prediction vector
	

	### Parameter values come from Table 1 of the Supplementary Material of the paper "Frontal cortex function derives from hierarchical predictive coding", W. Alexander, J. Brown
	learn_rate_vec = [0.075, 0.075, 0.075]	# learning rates 
	learn_rate_memory = [1,1,1]	
	beta_vec = [15, 15, 15]                 # gain parameter for memory dynamics
	gamma = 15                             # gain parameter for response making
	elig_decay_vec = [0.1,0.5,0.99]         # decay factors for eligibility trace
	bias_vec = [10,0.01,0.01]
	
	learn_rule_WM = 'backprop'
	elig_update = 'pre'
	init = 'zero'
	
	gate = 'softmax'

	do_training = True 		# if false, it loads the weights from previous training
	do_test = True

	do_weight_plots = True
	do_error_plots = True

	verb = 0

	HER = HER_arch(NL,S,P,learn_rate_vec,learn_rate_memory,beta_vec,gamma,elig_decay_vec,dic_stim,dic_resp,init)
	HER.print_HER(False)

	## TRAINING
	data_folder='DATA'
	if do_training:
		E,conv_iter = HER.training(S_tr,O_tr,bias_vec,learn_rule_WM,verb,gate)
			
		# save trained model
		str_err = data_folder+'/'+task+'_error_2.txt'
		np.savetxt(str_err, E)
		str_conv = data_folder+'/'+task+'_conv_2.txt'
		np.savetxt(str_conv, conv_iter)
		print(conv_iter)
		for l in np.arange(NL):
			str_mem = data_folder+'/'+task+'_weights_memory_'+str(l)+'_2.txt'
			np.savetxt(str_mem, HER.H[l].X)
			str_pred = data_folder+'/'+task+'_weights_prediction_'+str(l)+'_2.txt'
			np.savetxt(str_pred, HER.H[l].W)		
		print("\nSaved model to disk.\n")
	
	else:
		str_err = data_folder+'/'+task+'_error_2.txt'
		E = np.loadtxt(str_err)
		str_conv = data_folder+'/'+task+'_conv_2.txt'
		conv_iter = np.loadtxt(str_conv)
		print(conv_iter)
		for l in np.arange(NL):
			str_mem = data_folder+'/'+task+'_weights_memory_'+str(l)+'_2.txt'
			HER.H[l].X = np.loadtxt(str_mem)
			str_pred = data_folder+'/'+task+'_weights_prediction_'+str(l)+'_2.txt'
			HER.H[l].W = np.loadtxt(str_pred)	
		print("\nLoaded model from disk.\n")


	print('\n----------------------------------------------------\nTEST\n------------------------------------------------------\n')

	## TEST
	if do_test:
		HER.test(S_test,O_test,bias_vec,verb,gate)

	## PLOTS
	# plot of the memory weights
	image_folder = 'IMAGES'
	fontTitle = 26
	fontTicks = 22
	fontLabel = 22
	if do_weight_plots:
		fig1 = plt.figure(figsize=(10*NL,8))
		for l in np.arange(NL):
			X = HER.H[l].X
			plt.subplot(1,NL,l+1)
			plt.pcolor(np.flipud(X),edgecolors='k', linewidths=1)
			plt.set_cmap('Blues')			
			plt.colorbar()
			tit = 'MEMORY WEIGHTS: Level '+str(l)
			plt.title(tit,fontweight="bold",fontsize=fontTitle)
			plt.xticks(np.linspace(0.5,S-0.5,S,endpoint=True),cues_vec,fontsize=fontTicks)
			plt.yticks(np.linspace(0.5,S-0.5,S,endpoint=True),np.flipud(cues_vec),fontsize=fontTicks)
		plt.show()
		savestr = image_folder+'/'+task+'_weights_memory.png'
		if gate=='free':
			savestr = image_folder+'/'+task+'_weights_memory_nomemory.png'
		fig1.savefig(savestr)		

	
		fig2 = plt.figure(figsize=(10*NL,8))
		for l in np.arange(NL):	
			W = HER.H[l].W
			plt.subplot(1,NL,l+1)
			plt.pcolor(np.flipud(W),edgecolors='k', linewidths=1)
			plt.set_cmap('Blues')			
			plt.colorbar()
			tit = 'PREDICTION WEIGHTS: Level '+str(l)
			plt.title(tit,fontweight="bold",fontsize=fontTitle)
			if l==0:			
				plt.xticks(np.linspace(0.5,np.shape(W)[1]-0.5,P,endpoint=True),pred_vec,fontsize=fontTicks)
			else:
				dx = np.shape(W)[1]/(2*S)
				plt.xticks(np.linspace(dx,np.shape(W)[1]-dx,S,endpoint=True),cues_vec,fontsize=fontTicks)
			plt.yticks(np.linspace(0.5,S-0.5,S,endpoint=True),np.flipud(cues_vec),fontsize=fontTicks)
		plt.show()
		savestr = image_folder+'/'+task+'_weights_prediction.png'
		if gate=='free':
			savestr = image_folder+'/'+task+'_weights_prediction_nomemory.png'
		fig2.savefig(savestr)		


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
		plt.bar(bin*np.arange(len(E_bin))/6,E_bin,width=bin/6,color='blue',edgecolor='black', alpha=0.6)
		plt.axvline(x=4492/6, linewidth=5, ls='dashed', color='b')
		if conv_iter!=0:
			plt.axvline(x=conv_iter/6, linewidth=5, color='b')
		tit = '12AX: Training Convergence'
		plt.title(tit,fontweight="bold",fontsize=fontTitle)		
		plt.xlabel('Training Trials',fontsize=fontLabel)
		plt.ylabel('Number of Errors per bin',fontsize=fontLabel)		
		plt.xticks(np.linspace(0,N_round/6,5,endpoint=True),fontsize=fontTicks)	
		plt.yticks(fontsize=fontTicks)	
		text = 'Bin = '+str(np.around(bin).astype(int))
		plt.figtext(x=0.38,y=0.78,s=text,fontsize=fontLabel,bbox={'facecolor':'white', 'alpha':0.5, 'pad':10})

		plt.subplot(1,2,2)
		plt.plot(np.arange(N)/6, E_cum, color='blue',linewidth=7, alpha=0.6)
		plt.axvline(x=4492/6, linewidth=5, ls='dashed', color='b')
		if conv_iter!=0:
			plt.axvline(x=conv_iter/6, linewidth=5, color='b')
		tit = '12AX: Cumulative Training Error'
		plt.title(tit,fontweight="bold",fontsize=fontTitle)			
		plt.xticks(np.linspace(0,N_round/6,5,endpoint=True),fontsize=fontTicks)
		plt.yticks(fontsize=fontTicks)
		plt.xlabel('Training Trials',fontsize=fontLabel)
		plt.ylabel('Cumulative Error',fontsize=fontLabel)
		plt.show()

		savestr = image_folder+'/'+task+'_error.png'		
		if gate=='free':
			savestr = image_folder+'/'+task+'_error_nomemory.png'
		figE.savefig(savestr)


#########################################################################################################################################
#######################   TASK SACCADES/ANTI-SACCADES
#########################################################################################################################################

elif (task_selection=="4"):
	
	from TASKS.task_saccades import data_construction
	task = 'saccade'

	np.random.seed(1234)
	cues_vec = ['empty','P','A','L','R']
	pred_vec = ['LC','LW','FC','FW','RC','RW']

	N_trial = 15000 
	perc_tr = 0.8
	S_tr,O_tr,S_test,O_test,dic_stim,dic_resp = data_construction(N=N_trial,perc_training=perc_tr)

	## CONSTRUCTION OF THE HER MULTI-LEVEL NETWORK
	NL = 3                       # number of levels (<= 3)
	S = np.shape(S_tr)[1]        # dimension of the input 
	P = np.shape(O_tr)[1] 	     # dimension of the prediction vector
	

	### Parameter values come from Table 1 of the Supplementary Material of the paper "Frontal cortex function derives from hierarchical predictive coding", W. Alexander, J. Brown
	learn_rate_vec = [0.1, 0.02,0.02]	# learning rates 
	learn_rate_memory = [0.1,0.1,0.1]
	beta_vec = [12, 12,12]                 # gain parameter for memory dynamics
	gamma = 12                              # gain parameter for response making
	elig_decay_vec = [0.3,0.5,0.9]          # decay factors for eligibility trace	
	bias = [0,0,0]

	gate = 'softmax'

	do_training = False 		# if false, it loads the weights from previous training
	do_test = False

	do_weight_plots = False
	do_error_plots = True

	verb = 0

	HER = HER_arch(NL,S,P,learn_rate_vec,learn_rate_memory,beta_vec,gamma,elig_decay_vec,dic_stim,dic_resp)
	HER.print_HER(False)
	#print(S_tr[:20,:])

	## TRAINING
	data_folder='DATA'
	N_training = np.around(N_trial*perc_tr).astype(int)
	if do_training:

		E_fix,E_go,conv_iter = HER.training_saccade(N_training,S_tr,O_tr,bias,gate)
			
		# save trained model
		str_err = data_folder+'/'+task+'_error_fix.txt'
		np.savetxt(str_err, E_fix)
		str_err = data_folder+'/'+task+'_error_go.txt'
		np.savetxt(str_err, E_go)
		str_conv = data_folder+'/'+task+'_conv.txt'
		np.savetxt(str_conv, conv_iter)
		for l in np.arange(NL):
			str_mem = data_folder+'/'+task+'_weights_memory_'+str(l)+'.txt'
			np.savetxt(str_mem, HER.H[l].X)
			str_pred = data_folder+'/'+task+'_weights_prediction_'+str(l)+'.txt'
			np.savetxt(str_pred, HER.H[l].W)		
		print("\nSaved model to disk.\n")
	
	else:

		str_err = data_folder+'/'+task+'_error_fix.txt'
		E_fix = np.loadtxt(str_err)
		str_err = data_folder+'/'+task+'_error_go.txt'
		E_go = np.loadtxt(str_err)

		str_conv = data_folder+'/'+task+'_conv.txt'
		conv_iter = np.loadtxt(str_conv)
		for l in np.arange(NL):
			str_mem = data_folder+'/'+task+'_weights_memory_'+str(l)+'.txt'
			HER.H[l].X = np.loadtxt(str_mem)
			str_pred = data_folder+'/'+task+'_weights_prediction_'+str(l)+'.txt'
			HER.H[l].W = np.loadtxt(str_pred)	
		print("\nLoaded model from disk.\n")


	print('\n----------------------------------------------------\nTEST\n------------------------------------------------------\n')

	## TEST
	if do_test:
		N_test = N_trial - N_training
		HER.test_saccade(N_test,S_test,O_test,bias,verb,gate)
		print(conv_iter)

	## PLOTS
	# plot of the memory weights
	image_folder = 'IMAGES'
	fontTitle = 26
	fontTicks = 22
	fontLabel = 22

	if do_weight_plots:
		fig1 = plt.figure(figsize=(10*NL,8))
		for l in np.arange(NL):
			X = HER.H[l].X
			plt.subplot(1,NL,l+1)
			plt.pcolor(np.flipud(X),edgecolors='k', linewidths=1)
			plt.set_cmap('Blues')			
			plt.colorbar()
			tit = 'MEMORY WEIGHTS: Level '+str(l)
			plt.title(tit,fontweight="bold",fontsize=fontTitle)
			plt.xticks(np.linspace(0.5,S-0.5,S,endpoint=True),cues_vec,fontsize=fontTicks)
			plt.yticks(np.linspace(0.5,S-0.5,S,endpoint=True),np.flipud(cues_vec),fontsize=fontTicks)
		plt.show()
		savestr = image_folder+'/'+task+'_weights_memory.png'
		if gate=='free':
			savestr = image_folder+'/'+task+'_weights_memory_nomemory.png'
		fig1.savefig(savestr)		

	
		fig2 = plt.figure(figsize=(10*NL,8))
		for l in np.arange(NL):	
			W = HER.H[l].W
			plt.subplot(1,NL,l+1)
			plt.pcolor(np.flipud(W),edgecolors='k', linewidths=1)
			plt.set_cmap('Blues')			
			plt.colorbar()
			tit = 'PREDICTION WEIGHTS: Level '+str(l)
			plt.title(tit,fontweight="bold",fontsize=fontTitle)
			if l==0:			
				plt.xticks(np.linspace(0.5,np.shape(W)[1]-0.5,P,endpoint=True),pred_vec,fontsize=fontTicks)
			else:
				dx = np.shape(W)[1]/(2*S)
				plt.xticks(np.linspace(dx,np.shape(W)[1]-dx,S,endpoint=True),cues_vec,fontsize=fontTicks)
			plt.yticks(np.linspace(0.5,S-0.5,S,endpoint=True),np.flipud(cues_vec),fontsize=fontTicks)
		plt.show()
		savestr = image_folder+'/'+task+'_weights_prediction.png'
		if gate=='free':
			savestr = image_folder+'/'+task+'_weights_prediction_nomemory.png'
		fig2.savefig(savestr)	

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

		figE_fix = plt.figure(figsize=(22,8))
		plt.subplot(1,2,1)
		plt.bar(bin*np.arange(len(E_fix_bin)),E_fix_bin,width=bin,color='blue',edgecolor='black',label='fix',alpha=0.6)
		plt.axvline(x=225, linewidth=5, ls='dashed', color='orange')
		plt.axvline(x=0, linewidth=5, color='b')
		tit = 'SAS: Training Convergence for FIX'
		plt.title(tit,fontweight="bold",fontsize=fontTitle)		
		plt.xlabel('Training Trials',fontsize=fontLabel)
		plt.ylabel('Number of Errors per bin',fontsize=fontLabel)		
		plt.xticks(np.linspace(0,N,5,endpoint=True),fontsize=fontTicks)	
		plt.yticks(fontsize=fontTicks)	
		text = 'Bin = '+str(bin)
		plt.ylim((0,130))
		plt.figtext(x=0.37,y=0.78,s=text,fontsize=fontLabel,bbox={'facecolor':'white', 'alpha':0.5, 'pad':10})

		plt.subplot(1,2,2)
		plt.axvline(x=225, linewidth=5, ls='dashed', color='orange')
		plt.plot(np.arange(N), E_fix_cum, color='blue',linewidth=7,label='fix',alpha=0.6)
		plt.axvline(x=0, linewidth=5, color='b')
		tit = 'SAS: Cumulative Training Error for FIX'
		plt.title(tit,fontweight="bold",fontsize=fontTitle)			
		plt.xticks(np.linspace(0,N,5,endpoint=True),fontsize=fontTicks)
		plt.yticks(fontsize=fontTicks)
		plt.xlabel('Training Trials',fontsize=fontLabel)
		plt.ylabel('Cumulative Error',fontsize=fontLabel)
		plt.ylim((0,550))
		plt.show()

		savestr = image_folder+'/'+task+'_error_fix.png'
		if gate=='free':
			savestr = image_folder+'/'+task+'_error_nomemory_fix.png'		
		figE_fix.savefig(savestr)



		figE_go = plt.figure(figsize=(22,8))
		plt.subplot(1,2,1)
		plt.bar(bin*np.arange(len(E_go_bin)),E_go_bin,width=bin,color='blue',edgecolor='black',alpha=0.6)
		plt.axvline(x=4100, linewidth=5, ls='dashed', color='green')
		if conv_iter!=0:
			plt.axvline(x=conv_iter, linewidth=5, color='b')
		tit = 'SAS: Training Convergence for GO'
		plt.title(tit,fontweight="bold",fontsize=fontTitle)		
		plt.xlabel('Training Trials',fontsize=fontLabel)
		plt.ylabel('Number of Errors per bin',fontsize=fontLabel)		
		plt.xticks(np.linspace(0,N,5,endpoint=True),fontsize=fontTicks)	
		plt.yticks(fontsize=fontTicks)	
		text = 'Bin = '+str(bin)
		plt.figtext(x=0.37,y=0.78,s=text,fontsize=fontLabel,bbox={'facecolor':'white', 'alpha':0.5, 'pad':10})

		plt.subplot(1,2,2)
		plt.axvline(x=4100, linewidth=5, ls='dashed', color='green')
		plt.plot(np.arange(N), E_go_cum, color='blue',linewidth=7,alpha=0.6)
		if conv_iter!=0:
			plt.axvline(x=conv_iter, linewidth=5, color='b')
		tit = 'SAS: Cumulative Training Error for GO'
		plt.title(tit,fontweight="bold",fontsize=fontTitle)			
		plt.xticks(np.linspace(0,N,5,endpoint=True),fontsize=fontTicks)
		plt.yticks(fontsize=fontTicks)
		plt.xlabel('Training Trials',fontsize=fontLabel)
		plt.ylabel('Cumulative Error',fontsize=fontLabel)
		plt.show()

		savestr = image_folder+'/'+task+'_error_go.png'
		if gate=='free':
			savestr = image_folder+'/'+task+'_error_nomemory_go.png'		
		figE_go.savefig(savestr)



#########################################################################################################################################
#########################################################################################################################################	

else:
	print("No task identified. Please, retry.")

gc.collect()
