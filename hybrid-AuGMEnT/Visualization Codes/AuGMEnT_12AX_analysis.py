import os 
os.chdir('..')

from AuGMEnT_model import AuGMEnT			
from TASKS.task_12AX import data_construction

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec
import pylab
import gc 

from sys import version_info

np.set_printoptions(precision=3)

task = '12AX'

cues_vec = ['1','2','A','B','C','X','Y','Z']
cues_vec_tot = ['1+','2+','A+','B+','C+','X+','Y+','Z+','1-','2-','A-','B-','C-','X-','Y-','Z-']
pred_vec = ['L','R']

N_trial = 200000
p_target = 0.5

# build dataset
S_train, O_train = data_construction(N_trial,p_target)

## or load long dataset
#S_train = np.loadtxt('../DATA/12AX_Sdata_1000000.txt')
#O_train = np.loadtxt('../DATA/12AX_Odata_1000000.txt')
#S_train = S_train[:N_trial]

dic_stim = {'array([[1, 0, 0, 0, 0, 0, 0, 0]])':'1',
       	    'array([[0, 1, 0, 0, 0, 0, 0, 0]])':'2',
	    'array([[0, 0, 1, 0, 0, 0, 0, 0]])':'A',
	    'array([[0, 0, 0, 1, 0, 0, 0, 0]])':'B',
	    'array([[0, 0, 0, 0, 1, 0, 0, 0]])':'C',
	    'array([[0, 0, 0, 0, 0, 1, 0, 0]])':'X',
	    'array([[0, 0, 0, 0, 0, 0, 1, 0]])':'Y',
	    'array([[0, 0, 0, 0, 0, 0, 0, 1]])':'Z'}
dic_resp =  {'array([[1, 0]])':'L', 'array([[0, 1]])':'R',
	     '0':'L','1':'R'}

## CONSTRUCTION OF THE AuGMEnT NETWORK
S = 8        			# dimension of the input = number of possible stimuli
R = 10			     	# dimension of the regular units
M = 20 			     	# dimension of the memory units
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
	tit_aug = 'Hybrid AuGMEnT'
	print('Hybrid-AuGMEnT')	 
elif leak!=1:
	AuG_type = 'leaky_AuG'
	tit_aug = 'Leaky AuGMEnT'
	print('Leaky-AuGMEnT')
else:
	AuG_type = 'AuGMEnT' 
	tit_aug = 'AuGMEnT'
	print('Standard AuGMEnT')

# reward settings
rew = 'BRL'
prop = 'std'

N_sim = 1
E = np.zeros((N_sim,N_trial))
conv_ep = np.zeros((N_sim))
perc = np.zeros((N_sim))

do_weight_plots = True

S_tot = np.shape(S_train)[0]
Q_m = np.zeros((N_sim,S_tot,A)) 
RPE_v = np.zeros((N_sim,S_tot-1))

for n in np.arange(N_sim):

	print('SIMULATION ', n+1)

	model = AuGMEnT(S,R,M,A,alpha,beta,discount,eps,g,leak,rew,dic_stim,dic_resp,prop)

	conv_ep[n], Q_m[n,:], RPE_v[n,:] = model.training_12AX_2(S_train, O_train)

	print('\t CONVERGED AT TRIAL ', conv_ep[n])

cues_vec = []
values_vec = list(dic_stim.values())
for l in values_vec:
	cues_vec.append(l+'+')
for l in values_vec:
	cues_vec.append(l+'-')
mem_vec=[]
for i in range(M):
	mem_vec.append('M'+str(i+1))
act_vec=list(dic_resp.values())

if do_weight_plots:

	fontTitle = 32
	fontTicks = 24
	fontLabel = 28

	fig1 = plt.figure(figsize=(24,20))

	X = model.V_m
	np.savetxt('DATA/12AX_'+AuG_type+'_memory_V_m.txt',X)

	plt.pcolor(np.flipud(X),edgecolors='k', linewidths=1)
	plt.set_cmap('bwr')		
	plt.colorbar()
	tit = tit_aug+' - $V^M$'
	plt.title(tit,fontweight="bold",fontsize=fontTitle)
	plt.xticks(np.linspace(0.5,M-0.5,M,endpoint=True),mem_vec,fontsize=fontTicks)
	plt.yticks(np.linspace(0.5,2*S-0.5,2*S,endpoint=True),np.flipud(cues_vec),fontsize=fontTicks)

	savestr = AuG_type+'_'+task+'_weights_Vm.png'
	fig1.savefig(savestr)

	fig2 = plt.figure(figsize=(12,10))
	X = model.W_m
	np.savetxt('DATA/12AX_'+AuG_type+'_memory_W_m.txt',X)

	plt.pcolor(np.flipud(X),edgecolors='k', linewidths=1)
	plt.set_cmap('bwr')		
	plt.colorbar()
	tit = tit_aug+' - $W^M$'
	plt.title(tit,fontweight="bold",fontsize=fontTitle)
	plt.xticks(np.linspace(0.5,A-0.5,A,endpoint=True),act_vec,fontsize=fontTicks)
	plt.yticks(np.linspace(0.5,M-0.5,M,endpoint=True),np.flipud(mem_vec),fontsize=fontTicks)

	savestr = 'IMAGES/'+AuG_type+'_'+task+'_weights_Wm.png'
	fig2.savefig(savestr)
	plt.show()
