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

	cues_vec = ['A','B','X','Y']
	pred_vec = ['LC','LW','RC','RW']
	[S_tr,O_tr,S_test,O_test,dic_stim,dic_resp] = data_construction(N=100000, perc_target=0.2, perc_training=0.8)

	## CONSTRUCTION OF THE HER MULTI-LEVEL NETWORK
	NL = 2                      # number of levels (<= 3)
	S = np.shape(S_tr)[1]        # dimension of the input 
	P = np.shape(O_tr)[1] 	     # dimension of the prediction vector
	

	### Parameter values come from Table 1 of the Supplementary Material of the paper "Frontal cortex function derives from hierarchical predictive coding", W. Alexander, J. Brown
	learn_rate_vec = [0.1, 0.02]	# learning rates 
	beta_vec = [12, 12]             # gain parameter for memory dynamics
	elig_decay_vec = [0.3, 0.7]     # decay factors for eligibility trace

	gamma = 5                       # taken from suggestion from "Extended Example of HER Model" 

	fontTitle = 22
	fontTicks = 20

	verb = 0

	learn_rule_WM = 'backprop'  	# options are: backprop or RL
	elig_update = 'inter'		# options are: pre or post (eligibility trace update respectively at the beginning or at the end of the training iteration)

	do_training = True 		# if false, it loads the weights from previous training
	do_test = True
	error_test = False		# see behavior of the network after training on specified scenarios

	do_weight_plots = True

	HER = HER_arch(NL,S,P,learn_rate_vec,beta_vec,gamma,elig_decay_vec)

	## TRAINING
	data_folder = 'DATA'
	if do_training:
		
		HER.training(S_tr,O_tr,learn_rule_WM,elig_update,dic_stim,dic_resp)
		
		# save trained model
		for l in np.arange(NL):
			str_err = data_folder+'/'+task+'_error.txt'
			np.savetxt(str_err, E)
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

	cues_vec = ['1','2','A','B','X','Y']
	pred_vec = ['LC','LW','RC','RW']
	[S_tr,O_tr,S_test,O_test,dic_stim,dic_resp] = data_construction(N=10000,p_correct=0.25,perc_training=0.8)

	## CONSTRUCTION OF THE HER MULTI-LEVEL NETWORK
	NL = 3                       # number of levels (<= 3)
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
#######################   TASK SACCADES/ANTI-SACCADES
#########################################################################################################################################

if (task_selection=="4"):
	
	from TASKS.task_saccades import data_construction
	task = 'saccade'

	N_trial = 10000 
	perc_tr = 0.8
	S_tr,O_tr,S_test,O_test,dic_stim,dic_resp = data_construction(N=N_trial,perc_training=perc_tr)

	## CONSTRUCTION OF THE HER MULTI-LEVEL NETWORK
	NL = 3                       # number of levels (<= 3)
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
#########################################################################################################################################	

else:
	print("No task identified. Please, retry.")

gc.collect()
