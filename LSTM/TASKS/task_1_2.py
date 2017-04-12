## TRIVIAL TASK FOR HER EXPERIMENTATION
## The 1-2 task consists in the presentation to the subject of two stimuli '1' and '2', and two possible answers 'L' and 'R'.
## Simply, stimulus '1' is associated to response 'L' and stimulus '2' to response 'R'. 
## AUTHOR: Marco Martinolli
## DATE: 24.02.2017

import numpy as np

def preprocess_data(S,R):

	S_new=np.where(S==1,[1,0],[0,1])
	R_new=np.where(R=='L',[1,0],[0,1])		
	
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
def data_construction(N=1000, perc_1=0.7, perc_training=0.8):

	SS = np.random.choice(np.arange(3), (N,1), p=[0,perc_1,1-perc_1]) 
	RR = np.where(SS==1,'L','R')

	[S, O] = preprocess_data(SS,RR)

	[S_tr,O_tr,S_test,O_test]=subset_data(S,O,perc_training)

	dic_stim = {'array([[1, 0]])':'1', 'array([[0, 1]])':'2'}
	dic_resp =  {'array([[1, 0]])':'L', 'array([[0, 1]])':'R','0':'L','1':'R'}			

	return S_tr,O_tr,S_test,O_test,dic_stim,dic_resp

