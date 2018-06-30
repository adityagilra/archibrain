from AuGMEnT_model import AuGMEnT			
from AuGMEnT_model_mod import AuGMEnT_mod

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

from TASKS.task_12AX import data_construction
N_trial = 1000 
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


S_tr = np.load('DATA/dataset_12AX/dataset_12AX_100000_S.npy')
O_tr = np.load('DATA/dataset_12AX/dataset_12AX_100000_O.npy')

## CONSTRUCTION OF THE AuGMEnT NETWORK
S = 8        			# dimension of the input = number of possible stimuli
R = 10			     	# dimension of the regular units
M = 20 			     	# dimension of the memory units
A = 2			     	# dimension of the activity units = number of possible responses
	
# value parameters were taken from the 
lamb = 0.15    			# synaptic tag decay 
beta = 0.015			# weight update coefficient
discount = 0.9			# discount rate for future rewards
alpha = 1-lamb*discount 	# synaptic permanence	
eps = 0			# percentage of softmax modality for activity selection
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
rew = 'SRL'
prop = 'std'

N_check = 5
E_orig = np.zeros((N_trial))
H_m_orig = np.zeros((N_check,M))
E_mod = np.zeros((N_trial))
H_m_mod = np.zeros((N_check,M))	

model_orig = AuGMEnT(S,R,M,A,alpha,beta,discount,eps,g,leak,rew,dic_stim,dic_resp,prop)
model_mod = AuGMEnT_mod(S,R,M,A,alpha,beta,discount,eps,g,leak,rew,dic_stim,dic_resp,prop)

V_m_1 = np.copy(model_orig.V_m)
W_m_1 = np.copy(model_orig.W_m)
V_r_1 = np.copy(model_orig.V_r)
W_r_1 = np.copy(model_orig.W_r)
W_m_back_1 = np.copy(model_orig.W_m_back)
W_r_back_1 = np.copy(model_orig.W_r_back)

print(np.linalg.norm(model_orig.V_m))
E_orig, conv_ep_orig, H_m_orig = model_orig.training_12AX_3(S_tr,O_tr,'greedy','unif',False)
print(np.linalg.norm(model_orig.V_m))
print('\t CONVERGED AT TRIAL ', conv_ep_orig)

model_mod.V_m = V_m_1
model_mod.W_m = W_m_1
model_mod.V_r = V_r_1
model_mod.W_r = W_r_1
model_mod.W_m_back = W_m_back_1
model_mod.W_r_back = W_r_back_1

print(np.linalg.norm(model_mod.V_m))
E_mod, conv_ep_mod, H_m_mod = model_mod.training_12AX_2(S_tr,O_tr,'greedy','unif',False)
print(np.linalg.norm(model_mod.V_m))
print('\t CONVERGED AT TRIAL ', conv_ep_mod)

folder = 'DATA'
str_conv = folder+'/12AX_orig_eq_conv'
np.save(str_conv,conv_ep_orig)
str_conv = folder+'/12AX_mod_eq_conv'
np.save(str_conv,conv_ep_mod)
str_err = folder+'/12AX_orig_eq_error'
np.save(str_err,E_orig)
str_err = folder+'/12AX_mod_eq_error'
np.save(str_err,E_mod)
str_mem = folder+'/12AX_orig_eq_mem'
np.save(str_mem,H_m_orig)
str_mem = folder+'/12AX_mod_eq_mem'
np.save(str_mem,H_m_mod)

V_m_orig = model_orig.V_m
V_m_mod = model_mod.V_m
W_m_orig = model_orig.W_m
W_m_mod = model_mod.W_m
str_weigth_V = folder+'/12AX_orig_eq_weight_V'
np.save(str_weigth_V, V_m_orig)
str_weigth_V = folder+'/12AX_mod_eq_weight_V'
np.save(str_weigth_V, V_m_mod)
str_weigth_W = folder+'/12AX_orig_eq_weight_W'
np.save(str_weigth_W, W_m_orig)
str_weigth_W = folder+'/12AX_mod_eq_weight_W'
np.save(str_weigth_W, W_m_mod)
