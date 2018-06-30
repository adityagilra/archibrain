from AuGMEnT_model import AuGMEnT			

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec
import pylab
import gc 

from sys import version_info

np.set_printoptions(precision=3)

from TASKS.task_seq_prediction import get_dictionary
task = 'seq_prediction_CPT'

N_train = 2
N_test = 0	
	
# value parameters were taken from the 
lamb = 0.2   			# synaptic tag decay 
beta = 0.15			# weight update coefficient
discount = 0.9			# discount rate for future rewards
alpha = 1-lamb*discount 	# synaptic permanence	
eps = 0.025			# percentage of softmax modality for activity selection
g = 1

leak = [1.0, 0.7] 			# additional parameter: leaking decay of the integrative memory

# reward settings
rew = 'BRL'

verb = 0

N_sim = 100
E = np.zeros((N_sim,N_train))
conv_ep = np.zeros((N_sim))
perc = np.zeros((N_sim))

do_weight_plots = False
stop = True
do_save = True		


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

d_vec = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
d_dim = np.shape(d_vec)
N_vec = np.repeat(N_sim,d_dim)

AVG_CONV = np.zeros(d_dim)
SD_CONV = np.zeros(d_dim)
d_cont = 0

for d in d_vec:
	
	print('Length = ',d+2)

	dic_stim, dic_resp = get_dictionary(d)

	## CONSTRUCTION OF THE AuGMEnT NETWORK
	S = d+2        		     # dimension of the input = number of possible stimuli
	R = 3			     # dimension of the regular units
	M = 8 			     # dimension of the memory units
	A = d+2			     # dimension of the activity units = number of possible responses

	for n in np.arange(N_sim):

		print('SIMULATION ', n+1)

		model = AuGMEnT(S,R,M,A,alpha,beta,discount,eps,g,leak,rew,dic_stim,dic_resp)

		E[n,:],conv_ep[n] = model.training_seq_pred_CPT(N_train,d,stop,verb)

		#print('\t CONVERGED AT TRIAL ', conv_ep[n])
	
		if N_test!=0:

			perc[n] = model.test_seq_pred(N_test)	

			print('Percentage of correct responses during test: ',perc,'%')

	conv_ep_reduced = np.delete(conv_ep, np.where(conv_ep==0))
	AVG_CONV[d_cont] = np.mean(conv_ep_reduced)
	SD_CONV[d_cont] = np.std(conv_ep_reduced)
	print('Average convergence time for trials with length ',d+2,':\t', AVG_CONV[d_cont],' (',SD_CONV[d_cont],')')
	d_cont += 1

	if do_save:
		str_conv = folder+'/'+AuG_type+'_'+task+'_'+'distr'+str(d)+'_conv.txt'
		np.savetxt(str_conv,conv_ep)	
		E_mean = np.mean(np.reshape(E,(-1,50)),axis=1)
		str_e = folder+'/'+AuG_type+'_'+task+'_'+'distr'+str(d)+'_error.txt'
		np.savetxt(str_e,E_mean)	
		str_perc = folder+'/'+AuG_type+'_'+task+'_'+'distr'+str(d)+'_perc.txt'
		np.savetxt(str_perc,perc)

str_conv_all_d = folder+'/'+AuG_type+'_'+task+'_CONV.txt'
np.savetxt(str_conv_all_d, AVG_CONV)	
str_sd_all_d = folder+'/'+AuG_type+'_'+task+'_SD.txt'
np.savetxt(str_sd_all_d, AVG_CONV)
