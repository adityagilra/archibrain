## TRIVIAL TASK FOR HER EXPERIMENTATION
## The 1-2 task consists in the presentation to the subject of two stimuli '1' and '2', and two possible answers 'L' and 'R'.
## Simply, stimulus '1' is associated to response 'L' and stimulus '2' to response 'R'. 
## AUTHOR: Marco Martinolli
## DATE: 24.02.2017

import numpy as np

from HER_level import HER_level, HER_base 

def preprocess_data(S,R):

	S_new=np.where(S==1,[1,0],[0,1])
	R_new=np.where(R=='L',[1,0,0,1],[0,1,1,0])		
	
	return S_new, R_new

def subset_data(S,O,training_perc=0.8):

	sz = np.shape(O)[0]
	idx = int(np.around(sz*training_perc))	

	# Distintion in training and test sets
	S_train = S[:idx, :]
	O_train = O[:idx, :]
	S_test = S[idx:, :]
	O_test = O[idx:, :]

	return S_train,O_train,S_test,O_test


# construction of the dataset
N1 = 800
N2 = 200
s_1 = np.full((N1,1),1,dtype=int)
s_2 = np.full((N2,1),2,dtype=int)
SS = np.concatenate((s_1,s_2),axis=0)
np.random.shuffle(SS)

RR = np.where(SS==1,'L','R')
[S, O] = preprocess_data(SS,RR)

[S_tr,O_tr,S_test,O_test]=subset_data(S,O,0.8)

L = HER_base(0,np.shape(S_tr)[1], 10, np.shape(O_tr)[1], 12, 12)
print('TRAINING...')
L.base_training(S_tr, O_tr)
print(' DONE\n')
print('TEST....\n')
L.base_test_binary(S_test, O_test)
print('DONE!\n')

import gc; gc.collect()
