from AuGMEnT_model_random import AuGMEnT			

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec
import pylab
import gc 

from sys import version_info

np.set_printoptions(precision=3)

from task_saccades import data_construction
task = 'saccade'

cues_vec = ['P','A','L','R']
cues_vec_tot = ['P+','A+','L+','R+']
pred_vec = ['L','F','R']

N_trial = 10000 
reset_cond = ['empty']	

## CONSTRUCTION OF THE AuGMEnT NETWORK
S = 4        			# dimension of the input = number of possible stimuli
R = 3			     # dimension of the regular units
M = 4 			     # dimension of the memory units
A = 3			     # dimension of the activity units = number of possible responses
	
# value parameters were taken from the 
lamb = 0.2    			# synaptic tag decay 
beta = 0.15			# weight update coefficient
discount = 0.9			# discount rate for future rewards
alpha = 1-lamb*discount 	# synaptic permanence	
eps = 0.025			# percentage of softmax modality for activity selection
leak = [0.7, 1.0] 			# additional parameter: leaking decay of the integrative memory
g = 1

if isinstance(leak, list):
	AuG_type = 'hybrid_AuG' 
	print('Hybrid-AuGMEnT')	 
elif leak!=1:
	AuG_type = 'leaky_AuG' 
	print('Leaky-AuGMEnT')
else:
	AuG_type = 'AuGMEnT' 
	print('Standard AuGMEnT')

do_test=False


# reward settings
rew = 'RL'
shape_fac = 0.2
prop = 'std'

verb = 1

N_sim = 1
E_fix = np.zeros((N_sim,N_trial))
E_go = np.zeros((N_sim,N_trial))
conv_ep = np.zeros((N_sim))
perc_go = np.zeros((N_sim))
perc_fix = np.zeros((N_sim))

stop = True	

_,_,_,_,dic_stim,dic_resp = data_construction(N=1,perc_training=1)	

for n in np.arange(N_sim):
	print('SIMULATION ', n+1)

	S_tr,O_tr,_,_,_,_ = data_construction(N=N_trial,perc_training=1)

	model = AuGMEnT(S,R,M,A,alpha,beta,discount,eps,g,leak,rew,dic_stim,dic_resp,prop)

	_,E_go[n,:],conv_ep[n] = model.training_saccade(N_trial,S_tr,O_tr,reset_cond,verb,shape_fac,stop)

	print('\t CONVERGED AT TRIAL ', conv_ep[n])
	
	if do_test==True:
		S_test,O_test,_,_,_,_ = data_construction(N=100,perc_training=1)
		perc_fix[n], perc_go[n] = model.test_saccade(S_test,O_test,reset_cond)	

		print('Percentage of correct FIX responses during test: ',perc_fix,'%')
		print('Percentage of correct GO responses during test: ',perc_go,'%')

folder = 'DATA'
str_conv = folder+'/'+AuG_type+'_'+task+'_conv.txt'
np.savetxt(str_conv,conv_ep)
print(np.shape(E_go))	
E_go_mean = np.mean(np.reshape(E_go,(N_sim,-1,50)),axis=2)
print(np.shape(E_go_mean))	
str_go = folder+'/'+AuG_type+'_'+task+'_error_go.txt'
np.savetxt(str_go,E_go_mean)	
E_fix_mean = np.mean(np.reshape(E_fix,(N_sim,-1,50)),axis=2)
str_fix = folder+'/'+AuG_type+'_'+task+'_error_fix.txt'
np.savetxt(str_fix,E_fix_mean)
if do_test==True:
	str_perc_go = folder+'/'+AuG_type+'_'+task+'_perc_go.txt'
	np.savetxt(str_perc_go,perc_go)	
	str_perc_fix = folder+'/'+AuG_type+'_'+task+'_perc_fix.txt'
	np.savetxt(str_perc_fix,perc_fix)	

