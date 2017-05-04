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

def preprocess_data(S,R):

	leng = np.shape(S)[0]
	num_task = np.max(S)+1
	S_new = np.zeros((leng,num_task))
	seq = np.arange(leng)

	S_new[seq,S] = 1
	R_new=np.where(R==0,[1,0],[0,1])		
	
	return S_new, R_new


# construction of the dataset
def subset_construction(N=500,p_correct=0.25):

	p_wrong = (1-p_correct)/3	
	SS = []
	RR = []

	for n in np.arange(N):	

		s_dig = np.random.choice(np.arange(2), (1,1))
		inner_loops = 3

		if s_dig==0:
			SS.append(0)
			RR.append(0)	
		else:
			SS.append(1)
			RR.append(0)				

		for i in np.arange(inner_loops):

			if s_dig==0:
	
				rand_inner_pattern = np.random.choice(np.arange(4),(1,1), p=[p_correct,p_wrong,p_wrong,p_wrong]) + 2	

				if rand_inner_pattern==2:
					SS.append(2)
					SS.append(4)
				elif rand_inner_pattern==3:
					SS.append(2)
					SS.append(5)
				elif rand_inner_pattern==4:
					SS.append(3)
					SS.append(4)
				elif rand_inner_pattern==5:
					SS.append(3)
					SS.append(5)

				if rand_inner_pattern==2:
					RR.append(0)			
					RR.append(1)
				else:	
					RR.append(0)			
					RR.append(0)					
			else:

				rand_inner_pattern = np.random.choice(np.arange(4),(1,1), p=[p_correct,p_wrong,p_wrong,p_wrong]) + 2	

				if rand_inner_pattern==2:
					SS.append(2)
					SS.append(4)
				elif rand_inner_pattern==3:
					SS.append(2)
					SS.append(5)
				elif rand_inner_pattern==4:
					SS.append(3)
					SS.append(4)
				elif rand_inner_pattern==5:
					SS.append(3)
					SS.append(5)

				if rand_inner_pattern==5:
					RR.append(0)			
					RR.append(1)
				else:	
					RR.append(0)			
					RR.append(0)	
	RR = np.reshape(RR,(-1,1))
	
	# preprocess data to have the right format	
	[S, O] = preprocess_data(SS,RR)
	
	return S,O

def data_construction(N=500,p_c=0.25,p_tr=0.8):
	
	N_tr = int(np.around(N*p_tr))	
	
	# data division in training and test subsets
	[S_tr, O_tr] = subset_construction(N_tr,p_c)
	[S_test, O_test] = subset_construction(N-N_tr,p_c)

	dic_stim = {'array([[1, 0, 0, 0, 0, 0]])':'1',
		    'array([[0, 1, 0, 0, 0, 0]])':'2',
		    'array([[0, 0, 1, 0, 0, 0]])':'A',
		    'array([[0, 0, 0, 1, 0, 0]])':'B',
		    'array([[0, 0, 0, 0, 1, 0]])':'X',
		    'array([[0, 0, 0, 0, 0, 1]])':'Y'}
	dic_resp =  {'array([[1, 0]])':'L', 'array([[0, 1]])':'R',
			'0':'L','1':'R'}			

	return S_tr, O_tr, S_test, O_test, dic_stim, dic_resp	
