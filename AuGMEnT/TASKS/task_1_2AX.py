## 1-2AX TASK 
## The 1-2AX task consists in the presentation to the subject of six possible stimuli/cues '1' - '2', 'A' - 'B', 'X' - 'Y'.
## The tester has 2 possible responses which depend on the temporal order of previous and current stimuli: 
## he has to answer 'R' when
##	- the last stored digit is '1' AND the previous stimulus is 'A' AND the current one is 'X', 
##	- the last stored digit is '2' AND the previous stimulus is 'B' AND the current one is 'Y';
## in any other case , reply 'L'.    
## AUTHOR: Marco Martinolli
## DATE: 28.02.2017

import numpy as np

def preprocess_data(S,R,dic):

	leng = np.shape(S)[0]
	num_task = np.max(S)+1
	S_new = dic[S[0,0]]
	for i in np.arange(leng-1):
		S_new = np.concatenate((S_new, dic[S[i+1,0]]))
	
	R_new=np.where(R==0,[1,0],[0,1])		
	
	return S_new, R_new

def subset_data(S,O,training_perc=0.8):

	sz = np.shape(O)[0]
	idx = int(np.around(sz*training_perc))	

	# Distintion in training and test sets
	S_train = S[:idx, :]
	O_train = O[:idx, :]

	S_test = S[idx:, :]
	O_test = O[idx:, :]

	# the test subset must starts with the digit correspondent to the first positive answer
	ind_R = np.where((O_test==[0,1]).all(1))	
	ind_first_R = ind_R[0][0]
	
	if(np.array_equiv(S_test[ind_first_R,:],[[0,0,1,0,0,0]])):
		# digit_to_insert = '1'
		S_test = np.concatenate([ [[1,0,0,0,0,0]],S_test ])
	else:
		# digit_to_insert = '2'
		S_test = np.concatenate([ [[0,1,0,0,0,0]],S_test ])
	O_test = np.concatenate([ [[1,0]],O_test ])

	return S_train,O_train,S_test,O_test


# construction of the dataset
def data_construction(N=500,p_correct=0.25,perc_training=0.8):

	p_wrong = (1-p_correct)/3	
	SS = []
	RR = []

	for n in np.arange(N):	
		s_dig = np.random.choice(np.arange(2), (1,1))
		SS.append(s_dig)
		RR.append(0)	
		random_inner_loops = np.random.choice(np.arange(4),(1,1)) +1
		for i in np.arange(random_inner_loops):

			if s_dig==0:
				rand_inner_pattern = np.random.choice(np.arange(4),(1,1), p=[p_correct,p_wrong,p_wrong,p_wrong]) + 2	
				if rand_inner_pattern==2:
					RR.append(0)			
					RR.append(1)
				else:	
					RR.append(0)			
					RR.append(0)					
			else:
				rand_inner_pattern = np.random.choice(np.arange(4),(1,1), p=[p_wrong,p_wrong,p_wrong,p_correct]) + 2
				if rand_inner_pattern==5:
					RR.append(0)			
					RR.append(1)
				else:	
					RR.append(0)			
					RR.append(0)							
			SS.append(rand_inner_pattern)	

	SS = np.reshape(SS,(-1,1))
	RR = np.reshape(RR,(-1,1))

	dic_building = {0:np.array([[1, 0, 0, 0, 0, 0]]),
		    	1:np.array([[0, 1, 0, 0, 0, 0]]),
		    	2:np.array([[0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 1, 0]]), 		# return 2 vectors: A, X
		    	3:np.array([[0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1]]),		# return 2 vectors: A, Y
			4:np.array([[0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0]]),		# return 2 vectors: B, X
			5:np.array([[0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1]])}		# return 2 vectors: B, Y

	# preprocess data to have the right format	
	[S, O] = preprocess_data(SS,RR,dic_building)

	# data division in training and test subsets
	[S_tr, O_tr, S_test, O_test] = subset_data(S,O,0.8)

	dic_stim = {'array([[1, 0, 0, 0, 0, 0]])':'1',
		    'array([[0, 1, 0, 0, 0, 0]])':'2',
		    'array([[0, 0, 1, 0, 0, 0]])':'A',
		    'array([[0, 0, 0, 1, 0, 0]])':'B',
		    'array([[0, 0, 0, 0, 1, 0]])':'X',
		    'array([[0, 0, 0, 0, 0, 1]])':'Y'}
	dic_resp =  {'array([[1, 0]])':'L', 'array([[0, 1]])':'R',
			'0':'L','1':'R'}			

	return S_tr, O_tr, S_test, O_test, dic_stim, dic_resp
