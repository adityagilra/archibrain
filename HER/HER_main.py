## MAIN FILE FOR HER TESTING
## Here are defined the settings for the HER architecture and for the tasks to test.
## AUTHOR: Marco Martinolli
## DATE: 07.03.2017

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.regularizers import l2
from keras.utils import np_utils

from HER_level import HER_level
from HER_base import HER_base
from HER_model import HER_arch

import numpy as np
import matplotlib 
matplotlib.use('GTK3Cairo') 
from matplotlib import pyplot as plt
import pylab
import gc; gc.collect()

from sys import version_info

task_dic ={'0':'task_1_2', 
           '1':'task_AX_CPT',
	   '2':'task_1_2AX', 
	   '3':'titanic'}

py3 = version_info[0] > 2 # creates boolean value for test that Python major version > 2
if py3:
  task_selection = input("\nPlease select a task: \n\t 0: task_1_2 \n\t 1: task_AX_CPT \n\t 2: task_1_2AX\n\t 3: titanic\nEnter id number:  ")
else:
  task_selection = raw_input("\nPlease select a task: \n\t 0: task_1_2 \n\t 1: task_AX_CPT \n\t 2: task_1_2AX\n\t 3: titanic\nEnter id number:  ")

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

	cues_vec = ['A','B','X','Y']
	pred_vec = ['LC','LW','RC','RW']
	[S_tr,O_tr,S_test,O_test,dic_stim,dic_resp] = data_construction(N=10000, perc_target=0.2, perc_training=0.8)

	## CONSTRUCTION OF THE HER MULTI-LEVEL NETWORK
	NL = 2                      # number of levels (<= 3)
	S = np.shape(S_tr)[1]        # dimension of the input 
	P = np.shape(O_tr)[1] 	     # dimension of the prediction vector
	

	### Parameter values come from Table 1 of the Supplementary Material of the paper "Frontal cortex function derives from hierarchical predictive coding", W. Alexander, J. Brown
	learn_rate_vec = [0.25, 0.2]	# learning rates 
	beta_vec = [5, 5]                 # gain parameter for memory dynamics
	gamma = 12                        # gain parameter for response making
	elig_decay_vec = [0.3,0.9]          # decay factors for eligibility trace
	
	mem_plot = 1
	pred_plot = 1
	fontTitle = 22
	fontTicks = 20

	verb = 1

	HER = HER_arch(NL,S,P,learn_rate_vec,beta_vec,gamma,elig_decay_vec,reg_value=0,mem_activ_fct='linear',pred_activ_fct='linear')

	## TRAINING
	HER.training(S_tr,O_tr,dic_stim,dic_resp)
	#HER.false_training(S_tr,O_tr,dic_stim,dic_resp)
	
	if mem_plot:
		plt.figure(figsize=(10*NL,8))
		for l in np.arange(NL):
			W_m = HER.H[l].memory_branch.get_weights()
			plt.subplot(1,NL,l+1)
			plt.pcolor(np.flipud(W_m[0]),edgecolors='k', linewidths=1)
			plt.set_cmap('gray_r')			
			plt.colorbar()
			tit = 'MEMORY WEIGHTS: Level '+str(l)
			plt.title(tit,fontweight="bold",fontsize=fontTitle)
			plt.xticks(np.linspace(0.5,S-0.5,4,endpoint=True),cues_vec,fontsize=fontTicks)
			plt.yticks(np.linspace(0.5,S-0.5,4,endpoint=True),np.flipud(cues_vec),fontsize=fontTicks)
		plt.show()

	if pred_plot:
		plt.figure(figsize=(10*NL,8))
		for l in np.arange(NL):	
			W_p = HER.H[l].prediction_branch.get_weights()
			plt.subplot(1,NL,l+1)
			plt.pcolor(np.flipud(W_p[0]),edgecolors='k', linewidths=1)
			plt.set_cmap('gray_r')			
			plt.colorbar()
			tit = 'PREDICTION WEIGHTS: Level '+str(l)
			plt.title(tit,fontweight="bold",fontsize=fontTitle)
			if l==0:			
				plt.xticks(np.linspace(0.5,np.shape(W_p[0])[1]-0.5,4,endpoint=True),pred_vec,fontsize=fontTicks)
			else:
				dx = np.shape(W_p[0])[1]/(2*S)
				plt.xticks(np.linspace(dx,np.shape(W_p[0])[1]-dx,4,endpoint=True),cues_vec,fontsize=fontTicks)
			plt.yticks(np.linspace(0.5,S-0.5,4,endpoint=True),np.flipud(cues_vec),fontsize=fontTicks)
		plt.show()


	print('\n----------------------------------------------------\n---------------------------------------------------------------\n')
	## TEST
	#HER.test(S_test,O_test,dic_stim,dic_resp,verb)


#########################################################################################################################################
#######################   TASK 1-2 AX
#########################################################################################################################################

elif (task_selection=="2"):
	
	from TASKS.task_1_2AX import data_construction

	[S_tr,O_tr,S_test,O_test,dic_stim,dic_resp] = data_construction(N=10000, p_digit=0.1, p_wrong=0.15, p_correct=0.25, perc_training=0.9)

	## CONSTRUCTION OF THE HER MULTI-LEVEL NETWORK
	NL = 2                       # number of levels (<= 3)
	S = np.shape(S_tr)[1]        # dimension of the input 
	P = np.shape(O_tr)[1] 	     # dimension of the prediction vector
	

	### Parameter values come from Table 1 of the Supplementary Material of the paper "Frontal cortex function derives from hierarchical predictive coding", W. Alexander, J. Brown
	learn_rate_vec = [0.1, 0.02, 0.02]	# learning rates 
	beta_vec = [12, 12, 12]                 # gain parameter for memory dynamics
	gamma = 12                              # gain parameter for response making
	elig_decay_vec = [0.3,0.5,0.9]          # decay factors for eligibility trace

	verb = 1

	HER = HER_arch(NL,S,P,learn_rate_vec,beta_vec,gamma,elig_decay_vec,reg_value=1,mem_activ_fct='sigmoid',pred_activ_fct='sigmoid')

	## TRAINING
	HER.training(S_tr,O_tr,dic_stim,dic_resp,verb)


	## TEST
	HER.test(S_test,O_test,dic_stim,dic_resp,verb)


#########################################################################################################################################
#######################   TITANIC TASK
#########################################################################################################################################

elif (task_selection=="3"):

	from TASKS.titanic import data_construction

	[S_tr,O_tr,S_test,O_test,dic_resp] = data_construction(perc_training=0.9)

	## CONSTRUCTION OF BASE LEVEL OF HER ARCHITECTURE
	ss = np.shape(S_tr)[1]
	P = np.shape(O_tr)[1]
	alpha = 0.1
	beta = 12
	gamma = 12
	elig_decay_const = 0.3

	verb = 0
 
	L = HER_base(0,ss, P, alpha, beta, gamma, elig_decay_const, reg_value=0,loss_fct='categorical_crossentropy',pred_activ_fct='sigmoid')

	print('TRAINING...')
	L.base_training(S_tr, O_tr)
	print(' DONE!\n')

	print('TEST....\n')
	L.base_test(S_test, O_test, None, dic_resp, verb)
	print('DONE!\n')	
	
#########################################################################################################################################
#########################################################################################################################################

else:
	print("No task identified. Please, retry.")
