## MAIN FILE FOR HER TESTING
## Here are defined the settings for the HER architecture and for the tasks to test.
## AUTHOR: Marco Martinolli
## DATE: 07.03.2017

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.regularizers import l2
from keras.utils import np_utils

from HER_level import HER_level, HER_base 
from HER_model import HER_arch

import numpy as np
import gc; gc.collect()

from sys import version_info

task_dic ={'0':'task_1_2', 
	   '1':'task_1_2AX', 
	   '2':'titanic'}

py3 = version_info[0] > 2 # creates boolean value for test that Python major version > 2
if py3:
  task_selection = input("\nPlease select a task: \n\t 0: task_1_2 \n\t 1: task_1_2AX\n\t 2: titanic\nEnter id number:  ")
else:
  task_selection = raw_input("Please select a task: \n\t 0: task_1_2 \n\t 1: task_1_2AX\n\t 2: titanic\nEnter id number:  ")

print("\nYou have selected: ", task_dic[task_selection],'\n\n')

#########################################################################################################################################
#######################   TASK 1-2 
#########################################################################################################################################

if (task_selection=="0"):

	from TASKS.task_1_2 import data_construction

	[S_tr,O_tr,S_test,O_test,dic_stim,dic_resp] = data_construction(N=1000, p1=0.7, p2=0.3, perc_training=0.8)

	## CONSTRUCTION OF BASE LEVEL OF HER ARCHITECTURE
	ss = np.shape(S_tr)[1]
	M = ss
	P = np.shape(O_tr)[1]
	alpha = 0.1
	beta = 12
	gamma = 12

	verb = 1
 
	L = HER_base(0,ss,M, P, alpha, beta, gamma)

	print('TRAINING...')
	L.base_training(S_tr, O_tr)
	print(' DONE!\n')

	print('TEST....\n')
	L.base_test_binary(S_test, O_test, dic_stim, dic_resp, verb)
	print('DONE!\n')


#########################################################################################################################################
#######################   TASK 1-2 AX
#########################################################################################################################################

elif (task_selection=="1"):
	
	from TASKS.task_1_2AX import data_construction

	[S_tr,O_tr,S_test,O_test,dic_stim,dic_resp] = data_construction(N=1000, p_digit=0.05, p_wrong=0.225, p_correct=0.225, perc_training=0.8)

	## CONSTRUCTION OF THE HER MULTI-LEVEL NETWORK
	NL = 3
	S = np.shape(S_tr)[1]
	M = 20
	P = np.shape(O_tr)[1]
	learn_rate_vec = [0.1, 0.02, 0.02]
	beta_vec = [12, 12, 12]
	gamma = 2

	verb = 1

	HER = HER_arch(NL,S,M,P,learn_rate_vec,beta_vec,gamma,reg_value=0.1)


	## TRAINING
	HER.training(S_tr,O_tr,5)


	## TEST
	HER.test(S_test,O_test,dic_stim,dic_resp,verb)

#########################################################################################################################################
#######################   TITANIC TASK
#########################################################################################################################################

elif (task_selection=="2"):

	from TASKS.titanic import data_construction

	[S_tr,O_tr,S_test,O_test,dic_resp] = data_construction(perc_training=0.9)

	## CONSTRUCTION OF BASE LEVEL OF HER ARCHITECTURE
	ss = np.shape(S_tr)[1]
	M = 10
	P = np.shape(O_tr)[1]
	alpha = 0.1
	beta = 12
	gamma = 12

	verb = 0
 
	L = HER_base(0,ss,M, P, alpha, beta, gamma, reg_value=0,loss_fct='categorical_crossentropy',pred_activ_fct='sigmoid')

	print('TRAINING...')
	L.base_training(S_tr, O_tr)
	print(' DONE!\n')

	print('TEST....\n')
	L.base_test_binary(S_test, O_test, None, dic_resp, verb)
	print('DONE!\n')	
	
#########################################################################################################################################
#########################################################################################################################################

else:
	print("No task identified. Please, retry.")

