## AX CPT TASK
## The AX CPT task consists in the presentation to the subject of four possible stimuli/cues: two context cues 'A' - 'B' and 2 target cues 'X' - 'Y'.
## The tester has 2 possible responses which depend on the temporal order of previous and current stimuli: 
## he has to answer 'R' when
##	- the current stimulus is 'X' AND the previous stimulus is 'A' , 
## in any other case , reply 'L'.    
## AUTHOR: Marco Martinolli
## DATE: 16.03.2017

from keras.utils import np_utils
import numpy as np

def preprocess_data(S,R):

	S_new=np_utils.to_categorical(S, nb_classes=4)
	
	R_new=np.where(R==0,[1,0,0,1],[0,1,1,0])		
	
	return S_new, R_new


def subset_data(S,O,training_perc=0.8):

	sz = np.shape(O)[0]
	idx = int(np.around(sz*training_perc))	

	# Distintion in training and test sets
	S_train = S[:idx, :]
	O_train = O[:idx, :]

	S_test = S[idx:, :]
	O_test = O[idx:, :]

	# the test subset must starts with the a context cue
	if np.array_equiv(S_test[0,:],[[0,0,1,0]]) or np.array_equiv(S_test[0,:],[[0,0,0,1]]):
		if np.random.random()<0.5: 
			S_test = np.concatenate([ [[1,0,0,0]],S_test ])
		else:
			S_test = np.concatenate([ [[0,1,0,0]],S_test ])
		O_test = np.concatenate([ [[1,0,0,1]],O_test ])

	return S_train,O_train,S_test,O_test


# construction of the dataset
def data_construction(N=500,perc_target=0.2,perc_training=0.8):

	if np.remainder(N,2)!=0:
		N += 1

	cue_type = ['A','B','X','Y']
	
	SS = np.zeros((N,1))
	for i in np.arange(N/2):
		SS[int(2*i)] = np.random.choice(np.arange(2), (1,1), p=[0.5,0.5])
		if SS[int(2*i)]==0:		
			SS[int(2*i+1)] = np.random.choice(np.array([2,3]), (1,1), p=[perc_target,1-perc_target])
		else:
			SS[int(2*i+1)] = np.random.choice(np.array([2,3]), (1,1), p=[1-perc_target,perc_target])

	RR = np.zeros(np.shape(SS))
	for i,s in enumerate(SS):

		if s==2 and SS[i-1]==0:
			RR[i]=1
	
	# preprocess data to have the right format	
	[S, O] = preprocess_data(SS,RR)

	# data division in training and test subsets
	[S_tr, O_tr, S_test, O_test] = subset_data(S,O,0.8)

	dic_stim = {'array([[1, 0, 0, 0]])':'A',
		    'array([[0, 1, 0, 0]])':'B',
		    'array([[0, 0, 1, 0]])':'X',
		    'array([[0, 0, 0, 1]])':'Y'}
	dic_resp =  {'array([[1, 0, 0, 1]])':'L', 
			'array([[0, 1, 1, 0]])':'R',
			'0':'L',
			'1':'R'}			

	return S_tr,O_tr,S_test,O_test,dic_stim,dic_resp
