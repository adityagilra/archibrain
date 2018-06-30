from AuGMEnT_model import AuGMEnT			

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec
import pylab
import gc 

from sys import version_info

np.set_printoptions(precision=3)

from TASKS.task_seq_prediction import get_dictionary
task = 'seq_prediction'

N_train = 10000
N_test = 1000
	
# value parameters were taken from the 
lamb = 0.15    			# synaptic tag decay 
beta = 0.15			# weight update coefficient
discount = 0.9			# discount rate for future rewards
alpha = 1-lamb*discount 	# synaptic permanence	
eps = 0.025			# percentage of softmax modality for activity selection
g = 1

leak = 1.0 			# additional parameter: leaking decay of the integrative memory

# reward settings
rew = 'SRL'

verb = 0

N_sim = 100
E = np.zeros((N_sim,N_train))
conv_ep = np.zeros((N_sim))
perc = np.zeros((N_sim))

do_weight_plots = False
do_save = True
stop = False		

if isinstance(leak, list):
	AuG_type = 'hybrid_AuG'
	tit_aug = 'Hybrid AuGMEnT'
	print('Hybrid-AuGMEnT')	 
elif leak!=1:
	tit_aug = 'Leaky AuGMEnT'
	AuG_type = 'leaky_AuG' 
	print('Leaky-AuGMEnT')
else:
	tit_aug = 'AuGMEnT'
	AuG_type = 'AuGMEnT' 
	print('Standard AuGMEnT')

folder = 'DATA'

#d_vec = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
d_vec = np.array([4])
d_dim = np.shape(d_vec)
N_vec = np.repeat(N_sim,d_dim)

AVG_CONV = np.zeros(d_dim)
SD_CONV = np.zeros(d_dim)
d_cont = 0

for d in d_vec:
	
	print('Length = ',d+2)

	dic_stim, dic_resp = get_dictionary(d)

	## CONSTRUCTION OF THE AuGMEnT NETWORK
	S = d+2        		 # dimension of the input = number of possible stimuli
	R = 3			     # dimension of the regular units
	M = 4 			     # dimension of the memory units
	A = 2			     # dimension of the activity units = number of possible responses

	for n in np.arange(N_vec[d_cont]):

		print('SIMULATION ', n+1)
	
		model = AuGMEnT(S,R,M,A,alpha,beta,discount,eps,g,leak,rew,dic_stim,dic_resp)

		E[n,:],conv_ep[n] = model.training_seq_pred(N_train,d,stop,verb)

		#print('\t CONVERGED AT TRIAL ', conv_ep[n])
	
		if N_test!=0:
			perc[n] = model.test_seq_pred(N_test,d,verb)	
			print('Percentage of correct responses during test: ',perc,'%')

	conv_ep_reduced = np.delete(conv_ep, np.where(conv_ep==0))
	AVG_CONV[d_cont] = np.mean(conv_ep_reduced)
	SD_CONV[d_cont] = np.std(conv_ep_reduced)
	print('Average convergence time for trials with length ',d+2,':\t', AVG_CONV[d_cont],' (',SD_CONV[d_cont],')')
	d_cont += 1

	if do_save:
		str_conv = folder+'/'+task+'/'+'distr'+str(d)+'_conv_6.txt'
		np.savetxt(str_conv,conv_ep)	
		E_mean = np.mean(np.reshape(E,(-1,50)),axis=1)
		str_e = folder+'/'+task+'/'+'distr'+str(d)+'_error_6.txt'
		np.savetxt(str_e,E_mean)	
		str_perc = folder+'/'+task+'/'+'distr'+str(d)+'_perc_6.txt'
		np.savetxt(str_perc,perc)

#str_conv_all_d = folder+'/'+AuG_type+'_'+task+'_CONV.txt'
#np.savetxt(str_conv_all_d, AVG_CONV)	
#str_sd_all_d = folder+'/'+AuG_type+'_'+task+'_SD.txt'
#np.savetxt(str_sd_all_d, SD_CONV)	

###############################################################################

cues_vec = []
values_vec = list(dic_stim.values())
for l in values_vec:
	cues_vec.append(l+'+')
for l in values_vec:
	cues_vec.append(l+'-')
mem_vec=[]
for i in range(M):
	mem_vec.append('M'+str(i+1))
act_vec=list(dic_resp.values())[-2:]

if do_weight_plots:

	fontTitle = 32
	fontTicks = 24
	fontLabel = 28

	fig1 = plt.figure(figsize=(12,10))


	X = model.V_m
	savestr = AuG_type+'_'+task+'_'+'distr'+str(d)+'_weights_Vm.txt'
	np.savetxt(savestr,X)

	plt.pcolor(np.flipud(X),edgecolors='k', linewidths=1)
	plt.set_cmap('bwr')		
	cb = plt.colorbar()
	cb.ax.tick_params(labelsize=fontTicks)
	tit = tit_aug+' (L='+str(d+2)+') - $V^M$'
	plt.title(tit,fontweight="bold",fontsize=fontTitle)
	plt.xticks(np.linspace(0.5,M-0.5,M,endpoint=True),mem_vec,fontsize=fontTicks)
	plt.yticks(np.linspace(0.5,2*S-0.5,2*S,endpoint=True),np.flipud(cues_vec),fontsize=fontTicks)

	savestr = AuG_type+'_'+task+'_'+'distr'+str(d)+'_weights_Vm.png'
	fig1.savefig(savestr)

	fig2 = plt.figure(figsize=(10,10))
	X = model.W_m
	plt.pcolor(np.flipud(X),edgecolors='k', linewidths=1)
	plt.set_cmap('bwr')		
	cb = plt.colorbar()
	cb.ax.tick_params(labelsize=fontTicks)
	tit = tit_aug+' (L='+str(d+2)+') - $W^M$'
	plt.title(tit,fontweight="bold",fontsize=fontTitle)
	plt.xticks(np.linspace(0.5,A-0.5,A,endpoint=True),act_vec,fontsize=fontTicks)
	plt.yticks(np.linspace(0.5,M-0.5,M,endpoint=True),np.flipud(mem_vec),fontsize=fontTicks)

	savestr = AuG_type+'_'+task+'_'+'distr'+str(d)+'_weights_Wm.png'
	fig2.savefig(savestr)
	plt.show()	
