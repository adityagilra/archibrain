import os 

os.chdir('..')

from AuGMEnT_model import AuGMEnT			

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec
import pylab
import gc 


from sys import version_info

np.set_printoptions(precision=3)

from TASKS.task_seq_prediction import subset_construction, get_dictionary

task = 'seq_prediction'

N_train = 5000
d = 5

## build training dataset
#S_train, O_train = subset_construction(N_train, d) 

## or load it
S_train = np.loadtxt('DATA/seq_pred_dataset_d5_t5000_inp.txt')
O_train = np.loadtxt('DATA/seq_pred_dataset_d5_t5000_out.txt')
S_train = np.reshape(S_train, (N_train,-1,d+2))


dic_stim, dic_resp = get_dictionary(d)

## CONSTRUCTION OF THE AuGMEnT NETWORK
S = d+2        		     # dimension of the input = number of possible stimuli
R = 3			     # dimension of the regular units
M = 8 			     # dimension of the memory units
A = 2			     # dimension of the activity units = number of possible responses
	
# value parameters were taken from the 
lamb = 0.2    			# synaptic tag decay 
beta = 0.15			# weight update coefficient
discount = 0.9			# discount rate for future rewards
alpha = 1-lamb*discount 	# synaptic permanence	
eps = 0.025			# percentage of softmax modality for activity selection
g = 1

leak = [0.7, 1.0] 			# additional parameter: leaking decay of the integrative memory

# reward settings
rew = 'SRL'

verb = 0

N_sim = 10
conv_ep = np.zeros((N_sim))	

if isinstance(leak, list):
	AuG_type = 'hybrid_AuG' 
	print('Hybrid-AuGMEnT')	 
elif leak!=1:
	AuG_type = 'leaky_AuG' 
	print('Leaky-AuGMEnT')
else:
	AuG_type = 'AuGMEnT' 
	print('Standard AuGMEnT')

folder = 'DATA'

Q_m = np.zeros((N_sim,N_train*(d+1),A)) 
RPE_v = np.zeros((N_sim,N_train))
resp_v = np.zeros((N_sim,N_train*(d+1)))

for n in np.arange(N_sim):

	print('SIMULATION ', n+1)
	
	model = AuGMEnT(S,R,M,A,alpha,beta,discount,eps,g,leak,rew,dic_stim,dic_resp)

	conv_ep[n], Q_m[n,:,:], RPE_v[n,:], _ = model.training_seq_pred_2(S_train,O_train,dic_stim,dic_resp,verb)


print(np.mean(conv_ep))	

str_conv = folder+'/'+AuG_type+'_'+task+'_ALL_CONV_2.txt'
np.savetxt(str_conv,conv_ep)
str_Q = folder+'/'+AuG_type+'_'+task+'_ALL_Q_2.txt'
np.savetxt(str_Q,np.reshape(Q_m,(N_train*(d+1)*N_sim,A)))
str_RPE = folder+'/'+AuG_type+'_'+task+'_ALL_RPE_2.txt'
np.savetxt(str_RPE,RPE_v)
