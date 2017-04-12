## MAIN FILE FOR LSTM TESTING
## Here are defined the settings for the AuGMEnT architecture and for the tasks to test.
## AUTHOR: Marco Martinolli
## DATE: 11.04.2017

from LSTM_model import LSTM_arch

import numpy as np
import matplotlib 
matplotlib.use('GTK3Cairo') 
from matplotlib import pyplot as plt
from matplotlib import gridspec
import pylab
import gc 

from sys import version_info

task_dic ={'0':'task 12-AX',
	   '1':'saccade/anti-saccade task'}

py3 = version_info[0] > 2 	# creates boolean value for test that Python major version > 2
if py3:
  task_selection = input("\nPlease select a task: \n\t 0: task 12-AX\n\t 1: saccade/anti-saccade task\n Enter id number:  ")
else:
  task_selection = raw_input("\nPlease select a task: \n\t 0: task 12-AX\n\t 1: saccade/anti-saccade task\n Enter id number:  ")

print("\nYou have selected: ", task_dic[task_selection],'\n\n')



#########################################################################################################################################
#######################   TASK 1-2 AX
#########################################################################################################################################

if (task_selection=="0"):
	
	from TASKS.task_1_2AX import data_construction
	task = '12-AX'
	
	cues_vec = ['1','2','A','B','X','Y']
	cues_vec_tot = ['1+','2+','A+','B+','X+','Y+','1-','2-','A-','B-','X-','Y-']
	pred_vec = ['L','R']

	N = 20000
	perc_tr = 0.8
	p_c = 0.25
	S_tr,O_tr,S_test,O_test,dic_stim,dic_resp = data_construction(N,p_c,perc_tr)

	reset_cond = ['1','2']	

	## CONSTRUCTION OF THE LSTM NETWORK
	S = np.shape(S_tr)[1]        # dimension of the input = number of possible stimuli
	H = 100 		     # number of the hidden units
	O = 2			     # dimension of the activity units = number of possible responses
	
	# value parameters were taken from the 
	alpha = 0.1		# learning rate
	hysteresis = 0.5	# hysteresis coefficient ???  s.t. context layer update: c(t) = 0.5 h(t-1) + 0.5 c(t-1)
	toll_err = 0.1		# tolerance error ???

	verb = 1
	
	do_training = True
	do_test = True

	do_error_plots = False		
	
	fontTitle = 22
	fontTicks = 20
	fontLabel = 20

	model = LSTM_arch(S,H,O,alpha,dic_stim,dic_resp)

	## TRAINING
	folder = 'DATA'
	if do_training:	
		print('TRAINING...\n')
		S_tr = np.reshape(S_tr,(np.shape(S_tr)[0],1,S))
		model.training(S_tr,O_tr,reset_cond)
			
	print('\n------------------------------------------------------------------------------------------					\n-----------------------------------------------------------------------------------------------------\n')

	## TEST
	if do_test:
		print('TEST...\n')
		S_test = np.reshape(S_test,(np.shape(S_test)[0],1,S))
		model.test(S_test,O_test,reset_cond)


	## PLOTS
	# plot of the memory weights
	folder = 'IMAGES'
	if do_error_plots:

		N = len(E)
		bin = round(N*0.02)
		E_bin = np.reshape(E,(-1,bin))
		E_bin = np.sum(E_bin,axis=1)
		E_cum = np.cumsum(E)
		E_norm = 100*E_cum/(np.arange(N)+1)
		C = np.where(E==0,1,0)
		C_cum = 100*np.cumsum(C)/(np.arange(N)+1)
		X_stim = np.zeros((N))
		for i in np.arange(N):
			X_stim[i] = dic_stim[repr(S_tr[i:i+1,:].astype(int))]=='X'
		E_X = E[X_stim==1]
		num_X = len(E_X)

		figE = plt.figure(figsize=(20,8))
		plt.subplot(1,2,1)
		plt.bar(bin*np.arange(len(E_bin)),E_bin,width=bin,color='#7e88ee',edgecolor='black')
		tit = 'Training Error: Histogram'
		plt.title(tit,fontweight="bold",fontsize=fontTitle)		
		plt.xlabel('Training Iterations',fontsize=fontLabel)
		plt.ylabel('Number of Errors per bin',fontsize=fontLabel)		
		plt.xticks(np.linspace(0,N,5,endpoint=True),fontsize=fontTicks)	
		plt.yticks(fontsize=fontTicks)	
		text = 'Bin = '+str(bin)
		plt.figtext(x=0.8*N,y=0.9*np.max(E_bin),s=text,fontsize=fontLabel)

		plt.subplot(1,2,2)
		plt.plot(np.arange(N), E_cum, color='red',linewidth=2.5)
		tit = 'Training Error: Cumulative Error'
		plt.title(tit,fontweight="bold",fontsize=fontTitle)			
		plt.xticks(np.linspace(0,N,5,endpoint=True),fontsize=fontTicks)
		plt.yticks(fontsize=fontTicks)
		plt.xlabel('Training Iterations',fontsize=fontLabel)
		plt.ylabel('Cumulative Error',fontsize=fontLabel)
		plt.show()
		savestr = folder+'/'+task+'_error_'+rew+'.png'
		fig.savefig(savestr)





#########################################################################################################################################
#######################   TASK SACCADES/ANTI-SACCADES
#########################################################################################################################################

if (task_selection=="1"):
	
	from TASKS.task_saccades import data_construction
	task = 'saccade'

	N_trial = 1000 
	perc_tr = 0.8
	S_tr,O_tr,S_test,O_test,dic_stim,dic_resp = data_construction(N=N_trial,perc_training=perc_tr)

	reset_cond = ['empty']	

	## CONSTRUCTION OF THE LSTM NETWORK
	S = np.shape(S_tr)[1]        # dimension of the input = number of possible stimuli
	H = 100 		     # number of the hidden units
	O = 2			     # dimension of the activity units = number of possible responses
	
	# value parameters were taken from the 
	alpha = 0.1		# learning rate
	hysteresis = 0.5	# hysteresis coefficient ???  s.t. context layer update: c(t) = 0.5 h(t-1) + 0.5 c(t-1)
	toll_err = 0.1		# tolerance error ???

	verb = 1
	
	do_training = True
	do_test = True

	do_error_plots = False		
	
	fontTitle = 22
	fontTicks = 20
	fontLabel = 20

	model = LSTM_arch(S,H,O,alpha,dic_stim,dic_resp)

	## TRAINING
	folder = 'DATA'
	if do_training:	
		print('TRAINING...\n')
		S_tr = np.reshape(S_tr,(np.shape(S_tr)[0],1,S))
		model.training(S_tr,O_tr,reset_cond)
			
	print('\n------------------------------------------------------------------------------------------					\n-----------------------------------------------------------------------------------------------------\n')

	## TEST
	if do_test:
		print('TEST...\n')
		S_test = np.reshape(S_test,(np.shape(S_test)[0],1,S))
		model.test(S_test,O_test,reset_cond)


	## PLOTS
	# plot of the memory weights
	folder = 'IMAGES'
	if do_error_plots:

		N = len(E)
		bin = round(N*0.02)
		E_bin = np.reshape(E,(-1,bin))
		E_bin = np.sum(E_bin,axis=1)
		E_cum = np.cumsum(E)
		E_norm = 100*E_cum/(np.arange(N)+1)
		C = np.where(E==0,1,0)
		C_cum = 100*np.cumsum(C)/(np.arange(N)+1)
		X_stim = np.zeros((N))
		for i in np.arange(N):
			X_stim[i] = dic_stim[repr(S_tr[i:i+1,:].astype(int))]=='X'
		E_X = E[X_stim==1]
		num_X = len(E_X)

		figE = plt.figure(figsize=(20,8))
		plt.subplot(1,2,1)
		plt.bar(bin*np.arange(len(E_bin)),E_bin,width=bin,color='#7e88ee',edgecolor='black')
		tit = 'Training Error: Histogram'
		plt.title(tit,fontweight="bold",fontsize=fontTitle)		
		plt.xlabel('Training Iterations',fontsize=fontLabel)
		plt.ylabel('Number of Errors per bin',fontsize=fontLabel)		
		plt.xticks(np.linspace(0,N,5,endpoint=True),fontsize=fontTicks)	
		plt.yticks(fontsize=fontTicks)	
		text = 'Bin = '+str(bin)
		plt.figtext(x=0.8*N,y=0.9*np.max(E_bin),s=text,fontsize=fontLabel)

		plt.subplot(1,2,2)
		plt.plot(np.arange(N), E_cum, color='red',linewidth=2.5)
		tit = 'Training Error: Cumulative Error'
		plt.title(tit,fontweight="bold",fontsize=fontTitle)			
		plt.xticks(np.linspace(0,N,5,endpoint=True),fontsize=fontTicks)
		plt.yticks(fontsize=fontTicks)
		plt.xlabel('Training Iterations',fontsize=fontLabel)
		plt.ylabel('Cumulative Error',fontsize=fontLabel)
		plt.show()
		savestr = folder+'/'+task+'_error_'+rew+'.png'
		fig.savefig(savestr)

