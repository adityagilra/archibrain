from AuGMEnT_model_2 import AuGMEnT			

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec
import pylab
import gc 

from sys import version_info

np.set_printoptions(precision=3)

task = 'AX'

cues_vec = ['A','B','X','Y']
cues_vec_tot = ['A+','B+','X+','Y+','A-','B-','X-','Y-']
pred_vec = ['L','R']

N_trial = 10000 
p_target = 0.5

dic_stim = {'array([[1, 0, 0, 0]])':'A',
		    'array([[0, 1, 0, 0]])':'B',
		    'array([[0, 0, 1, 0]])':'X',
		    'array([[0, 0, 0, 1]])':'Y'}
dic_resp =  {'array([[1, 0]])':'L', 'array([[0, 1]])':'R',
			'0':'L','1':'R'}

## CONSTRUCTION OF THE AuGMEnT NETWORK
S = 4        			# dimension of the input = number of possible stimuli
R = 10			     # dimension of the regular units
M = 10 			     # dimension of the memory units
A = 2			     # dimension of the activity units = number of possible responses
	
# value parameters were taken from the 
lamb = 0.5    			# synaptic tag decay 
beta = 0.15			# weight update coefficient
discount = 0.9			# discount rate for future rewards
alpha = 1-lamb*discount 	# synaptic permanence	
eps = 0.025			# percentage of softmax modality for activity selection
leak = 1  			# additional parameter: leaking decay of the integrative memory
g = 1

# reward settings
rew = 'BRL'
prop = 'std'

N_sim = 10
E = np.zeros((N_sim,N_trial))
conv_ep = np.zeros((N_sim))
perc = np.zeros((N_sim))

stop = True
criterion='strong'		

do_test = False

for n in np.arange(N_sim):

	print('SIMULATION ', n+1)

	model = AuGMEnT(S,R,M,A,alpha,beta,discount,eps,g,leak,rew,dic_stim,dic_resp,prop)

	E[n,:],conv_ep[n] = model.training_12AX(N_trial,p_target,criterion,stop)

	print('\t CONVERGED AT TRIAL ', conv_ep[n])
	
	if do_test:
		N_test = 1000
		perc[n] = model.test(N_test,p_target)	
		print('Percentage of correct trials during test: ',perc,'%')

folder = 'DATA'
str_conv = folder+'/AuGMEnT_'+task+'_conv_3.txt'
np.savetxt(str_conv,conv_ep)	
E_mean = np.mean(np.reshape(E,(-1,50)),axis=1)
str_err = folder+'/AuGMEnT'+task+'_error_3.txt'
np.savetxt(str_err,E_mean)
if do_test:	
	str_perc = folder+'/AuGMEnT'+task+'_perc_3.txt'
	np.savetxt(str_perc,perc)
