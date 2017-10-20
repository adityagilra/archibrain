from HER_level import HER_level
from HER_base import HER_base
from HER_model import HER_arch

import numpy as np
import matplotlib 
import os
import activations as act

from matplotlib import pyplot as plt
import pylab
import gc
from sys import version_info


from task_saccades import data_construction
task = 'saccade'

cues_vec = ['P','A','L','R']
pred_vec = ['LC','LW','FC','FW','RC','RW']

dic_stim = {'array([[0, 0, 0, 0]])':'e',
		    'array([[1, 0, 0, 0]])':'P',
		    'array([[0, 1, 0, 0]])':'A',
		    'array([[0, 0, 1, 0]])':'L',
		    'array([[0, 0, 0, 1]])':'R'}		
dic_resp =  {'array([[0, 0, 0, 0, 0, 0]])':'None','array([[1, 0, 0, 1, 0, 1]])':'L', 'array([[0, 1, 1, 0, 0, 1]])':'F','array([[0, 1, 0, 1, 1, 0]])':'R', '0':'L','1':'F','2':'R'}		

N_trial = 25000 
perc_tr = 1

## CONSTRUCTION OF THE HER MULTI-LEVEL NETWORK
NL = 2                       # number of levels (<= 3)
S = 4			      # dimension of the input 
P = 6		 	     # dimension of the prediction vector
	
learn_rate_vec = [0.15, 0.1]	# learning rates 
learn_rate_memory = [1,1]
beta_vec = [8, 8]                 # gain parameter for memory dynamics
gamma = 5                              # gain parameter for response making
elig_decay_vec = [0.1,0.9]          # decay factors for eligibility trace	
bias = [0,0]

save_weights=True

verb = 0

HER = HER_arch(NL,S,P,learn_rate_vec,learn_rate_memory,beta_vec,gamma,elig_decay_vec,dic_stim,dic_resp)
HER.print_HER(False)

## TRAINING
data_folder='DATA'

N_sim = 10
E_fix = np.zeros((N_sim,N_trial))
E_go = np.zeros((N_sim,N_trial))
conv_tr = np.zeros((N_sim))
perc_fix = np.zeros((N_sim))
perc_go = np.zeros((N_sim))
stop = True

for n in np.arange(N_sim):

	print('SIMULATION ', n+1)
	S_tr,O_tr,_,_,_,_ = data_construction(N=N_trial,perc_training=1)

	HER = HER_arch(NL,S,P,learn_rate_vec,learn_rate_memory,beta_vec,gamma,elig_decay_vec,dic_stim,dic_resp)
	E_fix[n,:],E_go[n,:],conv_tr[n] = HER.training_saccade(S_tr,O_tr,bias,'softmax',stop)

	S_test,O_test,_,_,_,_ = data_construction(N=100,perc_training=1)
	perc_fix[n], perc_go[n] = HER.test_saccade(S_test,O_test,bias,'softmax')
	
	print('\t Percentage of correct fix responses: ', perc_fix[n],'%')
	print('\t Percentage of correct go responses: ', perc_go[n],'%')

E_fix_mean = np.mean(np.reshape(E_fix,(-1,50)),axis=1)
str_err_fix = data_folder+'/HER_long_'+task+'error_fix_2.txt'
np.savetxt(str_err_fix, E_fix_mean)	
E_go_mean = np.mean(np.reshape(E_go,(-1,50)),axis=1)
str_err_go = data_folder+'/HER_long_'+task+'error_go_2.txt'
np.savetxt(str_err_go, E_go_mean)
str_conv = data_folder+'/HER_long_'+task+'_conv_2.txt'
np.savetxt(str_conv, conv_tr)

str_err_fix = data_folder+'/HER_long_'+task+'_perc_fix_2.txt'
np.savetxt(str_err_fix, perc_fix)
str_err_go = data_folder+'/HER_long_'+task+'_perc_go_2.txt'
np.savetxt(str_err_go, perc_go)


fontTitle = 28
fontTicks = 22
fontLabel = 22
image_folder='IMAGES'

if save_weights==True and conv_tr[n]!=0:
	for l in np.arange(NL):
		str_mem = data_folder+'/long_'+task+'_weights_memory_'+str(l)+'.txt'
		np.savetxt(str_mem, HER.H[l].X)
		str_pred = data_folder+'/long_'+task+'_weights_prediction_'+str(l)+'.txt'
		np.savetxt(str_pred, HER.H[l].W)

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

	fig2.savefig(savestr)		

