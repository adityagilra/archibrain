from AuGMEnT_model_beta import AuGMEnT

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

N_trial = 100000
p_target = 0.5

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
eps = 0.025		# percentage of softmax modality for activity selection
g = 1

leak = [0.7,1.0]			# additional parameter: leaking decay of the integrative memory

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
rew = 'BRL'
prop = 'std'

policy = 'softmax'
stoc = 'soft'
t_weighted = True
e_weighted = False

if policy=='greedy':
	policy_str = policy
elif policy=='softmax':
	policy_str = policy
	if t_weighted:
		policy_str = policy_str+'_weighted'		
elif policy=='eps_greedy':
	policy_str = policy+'_'+stoc
	if t_weighted:
		policy_str = policy_str+'_weighted_t'
	if e_weighted:
		policy_str = policy_str+'_weighted_e'			

print('Policy: ', policy, '\tStochastic mth: ',stoc, '\tWeight t g? ', t_weighted,'\tWeight e g? ', e_weighted)

N_sim = 10
E = np.zeros((N_sim,N_trial))
conv_ep = np.zeros((N_sim))

stop = True
criterion='strong'	

verb=0

do_test = True
perc_expl = np.zeros((N_sim))
perc_no_expl = np.zeros((N_sim))
perc_soft = np.zeros((N_sim))

BET = [2, 4, 6, 8, 12, 14, 16, 18, 20]

for bet in BET:
    print('\nBETA = ', bet)
    for n in np.arange(N_sim):

	    print('\tSIMULATION ', n+1)

	    model = AuGMEnT(S,R,M,A,alpha,beta,discount,eps,g,leak,rew,dic_stim,dic_resp,prop)

	    E[n,:],conv_ep[n],_ = model.training_12AX(N_trial,p_target,criterion,stop,verb,policy,stoc,t_weighted,e_weighted,bet)
	    	
	    if do_test:
		    N_test = 1000
		    perc_expl[n], perc_no_expl[n], perc_soft[n] = model.test(N_test,p_target)	
		    print('Percentage of correct trials during test (exploration): ',perc_expl[n],'%')
		    print('Percentage of correct trials during test (no exploration): ',perc_no_expl[n],'%')
		    print('Percentage of correct trials during test (softmax): ',perc_soft[n],'%')

	    print('\t\tCONVERGED AT TRIAL ', conv_ep[n])

    folder = 'DATA'
    str_conv = folder+'/beta/'+policy_str+'/conv_'+str(bet)+'_3.txt'
    np.savetxt(str_conv,conv_ep)	
    
    if do_test:
	    str_perc = folder+'/beta/'+policy_str+'/perc_expl_'+str(bet)+'_2.txt'
	    np.savetxt(str_perc,perc_expl)
	    str_perc = folder+'/beta/'+policy_str+'/perc_no_expl_'+str(bet)+'_2.txt'
	    np.savetxt(str_perc,perc_no_expl)
	    str_perc = folder+'/beta/'+policy_str+'/perc_soft_'+str(bet)+'_2.txt'
