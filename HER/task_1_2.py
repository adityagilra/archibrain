from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.regularizers import l2
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
N2 = 500
s_1 = np.full((N1,1),1,dtype=int)
s_2 = np.full((N2,1),2,dtype=int)
SS = np.concatenate((s_1,s_2),axis=0)
np.random.shuffle(SS)

RR = np.where(SS==1,'L','R')
[S, O] = preprocess_data(SS,RR)

[S_tr,O_tr,S_test,O_test]=subset_data(S,O,0.7)

L = HER_base(np.shape(S_tr)[1], 10, np.shape(O_tr)[1], 10)

L.training(S_tr, O_tr)
L.test(S_test, O_test)


import gc; gc.collect()
