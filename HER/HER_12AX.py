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

from task_12AX import data_construction
task = '12AX'

cues_vec = ['1','2','A','B','C','X','Y','Z']
pred_vec = ['LC','LW','RC','RW']

dic_stim = {'array([[1, 0, 0, 0, 0, 0, 0, 0]])':'1',
		    'array([[0, 1, 0, 0, 0, 0, 0, 0]])':'2',
		    'array([[0, 0, 1, 0, 0, 0, 0, 0]])':'A',
		    'array([[0, 0, 0, 1, 0, 0, 0, 0]])':'B',
		    'array([[0, 0, 0, 0, 1, 0, 0, 0]])':'C',
		    'array([[0, 0, 0, 0, 0, 1, 0, 0]])':'X',
		    'array([[0, 0, 0, 0, 0, 0, 1, 0]])':'Y',
		    'array([[0, 0, 0, 0, 0, 0, 0, 1]])':'Z'}
dic_resp =  {'array([[1, 0, 0, 1]])':'L', 'array([[0, 1, 1, 0]])':'R',
			'0':'L','1':'R'}			

N_trial = 10000
perc_target = 0.5

## CONSTRUCTION OF THE HER MULTI-LEVEL NETWORK
### Parameter values come from Table 1 of the Supplementary Material of the paper "Frontal cortex function derives from hierarchical predictive coding", W. Alexander, J. Brown

NL = 3                       # number of levels (<= 3)
S = 8			      # dimension of the input 
P = 4		 	     # dimension of the prediction vector
	
learn_rate_vec = [0.075, 0.075, 0.075]	# learning rates 
learn_rate_memory = [1,1,1]
beta_vec = [15, 15, 15]                 # gain parameter for memory dynamics
gamma = 15                             # gain parameter for response making
elig_decay_vec = [0.1,0.5,0.99]         # decay factors for eligibility trace
bias_vec = [10,0.1,0.01]

save_weights=False
do_test = False

HER = HER_arch(NL,S,P,learn_rate_vec,learn_rate_memory,beta_vec,gamma,elig_decay_vec,dic_stim,dic_resp)
HER.print_HER(False)

## TRAINING
data_folder='DATA'

N_sim = 100
E = np.zeros((N_sim,N_trial))
conv_tr = np.zeros((N_sim))
perc = np.zeros((N_sim))

stop = True
N_test = 1000
criterion = 'human'

for n in np.arange(N_sim):

	print('SIMULATION ', n+1)
	HER = HER_arch(NL,S,P,learn_rate_vec,learn_rate_memory,beta_vec,gamma,elig_decay_vec,dic_stim,dic_resp)
	E[n,:],conv_tr[n] = HER.training(N_trial,perc_target,bias_vec,'softmax',stop,criterion)

	if do_test:
		perc[n] = HER.test(N_test,perc_target,bias_vec,'softmax')
		print('\t Percentage of correct TEST trials: ', perc[n],'%')

E_mean = np.mean(np.reshape(E,(-1,50)),axis=1)
str_err = data_folder+'/HER_'+task+'error_human.txt'
np.savetxt(str_err, E_mean)
str_conv = data_folder+'/HER_long_'+task+'_conv_human.txt'
np.savetxt(str_conv, conv_tr)

if do_test:
	str_test = data_folder+'/HER_long_'+task+'_perc_human.txt'
	np.savetxt(str_test, perc)


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
	savestr = image_folder+'/'+task+'_weights_memory_3.png'
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
	savestr = image_folder+'/'+task+'_weights_prediction_3.png'

	fig2.savefig(savestr)		

