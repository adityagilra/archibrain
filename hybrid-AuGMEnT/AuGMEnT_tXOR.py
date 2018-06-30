from AuGMEnT_model import AuGMEnT			

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec
import pylab
import gc 

from sys import version_info

np.set_printoptions(precision=3)

task = 'tXOR'
from task_tXOR import get_dictionary

cues_vec = ['1','2','A','B']
cues_vec_tot = ['1+','2+','A+','B+','1-','2-','A-','B-']
pred_vec = ['L','R']

N_trial = 10000 

dic_stim, dic_resp = get_dictionary() 

## CONSTRUCTION OF THE AuGMEnT NETWORK
S = 4        			# dimension of the input = number of possible stimuli
R = 3			     	# dimension of the regular units
M = 4 			     	# dimension of the memory units
A = 2			     	# dimension of the activity units = number of possible responses
	
# value parameters were taken from the 
lamb = 0.15    			# synaptic tag decay 
beta = 0.15			# weight update coefficient
discount = 0.9			# discount rate for future rewards
alpha = 1-lamb*discount 	# synaptic permanence	
eps = 0.025			# percentage of softmax modality for activity selection
g = 1

leak = [0.7, 1.0] 			# additional parameter: leaking decay of the integrative memory

if isinstance(leak, list):
	AuG_type = 'hybrid_AuG' 
	print('Hybrid-AuGMEnT')	 
elif leak!=1:
	AuG_type = 'leaky_AuG' 
	print('Leaky-AuGMEnT')
else:
	AuG_type = 'AuGMEnT' 
	print('Standard AuGMEnT')

# reward settings
rew = 'RL'
prop = 'std'

N_sim = 1
E = np.zeros((N_sim,N_trial))
conv_ep = np.zeros((N_sim))

stop = True		
verb = 1

for n in np.arange(N_sim):

	print('SIMULATION ', n+1)

	model = AuGMEnT(S,R,M,A,alpha,beta,discount,eps,g,leak,rew,dic_stim,dic_resp,prop)

	E[n,:],conv_ep[n] = model.training_tXOR(N_trial,stop,verb)

	print('\t CONVERGED AT TRIAL ', conv_ep[n])

folder = 'DATA'
str_conv = folder+'/'+AuG_type+'_'+task+'_conv.txt'
np.savetxt(str_conv,conv_ep)	
E_mean = np.mean(np.reshape(E,(-1,50)),axis=1)
str_err = folder+'/'+AuG_type+'_'+task+'_error.txt'
np.savetxt(str_err,E_mean)
