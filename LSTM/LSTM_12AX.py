## MAIN FILE FOR LSTM TESTING
## Here are defined the settings for the AuGMEnT architecture and for the tasks to test.
## AUTHOR: Marco Martinolli
## DATE: 11.04.2017

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec
import pylab
import gc 
from keras.models import load_model
import h5py
from sys import version_info


from task_1_2AX import data_construction
from LSTM_model import LSTM_arch

task = '12AX'

N_trial = 10000
perc_target = 0.5 
dt = 10 		

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

## CONSTRUCTION OF THE LSTM NETWORK
S = 8        	             # dimension of the input = number of possible stimuli
H = 10			     # dimension of the activity units = number of possible responses
O = 2

# value parameters were taken from the 
alpha = 0.01		# learning rate
b_sz  = 1	

N_sim = 10
E_sim = np.zeros((N_sim,N_trial))
conv_ep = np.zeros((N_sim))
perc = np.zeros((N_sim))

stop=True

do_test = True

for n in np.arange(N_sim):

	print('SIMULATION ',n+1)
	model = LSTM_arch(S,H,O,alpha,b_sz,dt,dic_stim,dic_resp)
	E_sim[n,:], conv_ep[n] = model.training_12AX(N_trial,perc_target,'strong',stop)
	print('\t LSTM converged in ', conv_ep[n],' trials')

	if do_test:
		perc[n] = model.test_12AX(1000,perc_target)
		print('\t Percentage of correct responses during test: ', perc[n])

folder = 'DATA'
E_sim_mean = np.mean(np.reshape(E_sim,(-1,50)),axis=1)
str_err = folder+'/LSTM_long_'+task+'_error.txt'
np.savetxt(str_err,E_sim_mean)
str_conv = folder+'/LSTM_long_'+task+'_conv.txt'
np.savetxt(str_conv,conv_ep)
str_perc = folder+'/LSTM_long_'+task+'_perc.txt'
np.savetxt(str_perc,perc)
