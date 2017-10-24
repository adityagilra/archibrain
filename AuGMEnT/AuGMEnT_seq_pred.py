from AuGMEnT_model_new import AuGMEnT			

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec
import pylab
import gc 

from sys import version_info

np.set_printoptions(precision=3)

from task_seq_prediction import get_dictionary
task = 'seq_prediction'

d = 15
dic_stim, dic_resp = get_dictionary(d)

cues_vec = []
for i in np.arange(len(dic_stim)):
	cues_vec.append(dic_stim[i])
pred_vec = [dic_resp[0], dic_resp[1]]

N_train = 10000 
N_test = 0	

## CONSTRUCTION OF THE AuGMEnT NETWORK
S = d+2        		     # dimension of the input = number of possible stimuli
R = 3			     # dimension of the regular units
M = 4 			     # dimension of the memory units
A = 2			     # dimension of the activity units = number of possible responses
	
# value parameters were taken from the 
lamb = 0.2    			# synaptic tag decay 
beta = 0.15			# weight update coefficient
discount = 0.9			# discount rate for future rewards
alpha = 1-lamb*discount 	# synaptic permanence	
eps = 0.025			# percentage of softmax modality for activity selection
g = 1

leak = 1  			# additional parameter: leaking decay of the integrative memory

# reward settings
rew = 'SRL'

verb = 1

N_sim = 1
E = np.zeros((N_sim,N_train))
conv_ep = np.zeros((N_sim))
perc = np.zeros((N_sim))

stop = True		

if leak==1:
	print('Standard AuGMEnT')
else:
	print('Leaky-AuGMEnT')

for n in np.arange(N_sim):

	print('SIMULATION ', n+1)

	model = AuGMEnT(S,R,M,A,alpha,beta,discount,eps,g,leak,rew,dic_stim,dic_resp)

	E[n,:],conv_ep[n] = model.training_seq_pred(N_train,d,stop,verb)

	print('\t CONVERGED AT TRIAL ', conv_ep[n])
	
	if N_test!=0:

		perc[n] = model.test_seq_pred(N_test)	

		print('Percentage of correct responses during test: ',perc,'%')


folder = 'DATA'
str_conv = folder+'/AuGMEnT_'+task+'_conv.txt'
np.savetxt(str_conv,conv_ep)	
E_mean = np.mean(np.reshape(E,(-1,50)),axis=1)
str = folder+'/AuGMEnT_'+task+'_error.txt'
np.savetxt(str,E_mean)	
str_perc = folder+'/AuGMEnT_'+task+'_perc.txt'
np.savetxt(str_perc,perc)
