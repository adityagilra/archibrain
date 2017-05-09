# 1-2AX TASK 
## The 1-2AX task consists in the presentation to the subject of six possible stimuli/cues '1' - '2', 'A' - 'B', 'X' - 'Y'.
## The tester has 2 possible responses which depend on the temporal order of previous and current stimuli: 
## he has to answer 'R' when
##	- the last stored digit is '1' AND the previous stimulus is 'A' AND the current one is 'X', 
##	- the last stored digit is '2' AND the previous stimulus is 'B' AND the current one is 'Y';
## in any other case , reply 'L'.    
## AUTHOR: Marco Martinolli
## DATE: 28.02.2017

import numpy as np

def preprocess_data(S, R, model=None):

	leng = np.shape(S)[0]
	num_task = np.max(S)+1
	S_new = np.zeros((leng,num_task))
	seq = np.arange(leng)
	
	if(model == '0' or model == '2'):
		S_new[seq,S] = 1
		R_new = np.where(R == 0, [1,0], [0,1])
	elif(model == '1'):
		S_new[seq,S] = 1
		R_new = np.where(R == 0, [1,0,0,1], [0,1,1,0])
	else:
		raise TypeError	
	
	return S_new, R_new

# construction of the dataset
def subset_construction(N=500, p_target=0.5, model=None):
	np.random.seed(1234)		

	p_wrong = (1-p_target)/7
	p_targ = p_target/2
	
	SS = []
	RR = []

	S_DIG = np.random.choice(np.arange(2), (N,1))
	RANDOM_NUMBER_INNER_LOOPS = np.random.choice(np.arange(4), (N,1)) +1
	tot_number = np.sum(RANDOM_NUMBER_INNER_LOOPS)
	RANDOM_PATTERNS = np.random.choice(np.arange(9), (tot_number,1), p=[p_targ, p_wrong, p_wrong, p_wrong, p_targ, p_wrong, p_wrong, p_wrong, p_wrong]) + 2

	cont = 0

	for n in np.arange(N):	

		s_dig = S_DIG[n]
		random_inner_loops = RANDOM_NUMBER_INNER_LOOPS[n]

		if s_dig==0:
			SS.append(0)
			RR.append(0)			
		else:
			SS.append(1)
			RR.append(0)			

		for i in np.arange(random_inner_loops):

			rand_inner_pattern = RANDOM_PATTERNS[cont]			
			cont += 1

			if s_dig==0:
				if rand_inner_pattern==2:
					SS.append(2)
					SS.append(4)
					RR.append(0)			
					RR.append(1)
				elif rand_inner_pattern==3:
					SS.append(2)
					SS.append(5)
					RR.append(0)			
					RR.append(0)
				elif rand_inner_pattern==4:
					SS.append(2)
					SS.append(7)
					RR.append(0)			
					RR.append(0)
				elif rand_inner_pattern==5:
					SS.append(3)
					SS.append(4)
					RR.append(0)			
					RR.append(0)
				elif rand_inner_pattern==6:
					SS.append(3)
					SS.append(5)
					RR.append(0)			
					RR.append(0)
				elif rand_inner_pattern==7:
					SS.append(3)
					SS.append(7)
					RR.append(0)			
					RR.append(0)
				elif rand_inner_pattern==8:
					SS.append(6)
					SS.append(4)
					RR.append(0)			
					RR.append(0)
				elif rand_inner_pattern==9:
					SS.append(6)
					SS.append(5)
					RR.append(0)			
					RR.append(0)
				elif rand_inner_pattern==10:
					SS.append(6)
					SS.append(7)
					RR.append(0)			
					RR.append(0)
	
			else:		
				if rand_inner_pattern==2:
					SS.append(2)
					SS.append(4)
					RR.append(0)			
					RR.append(0)
				elif rand_inner_pattern==3:
					SS.append(2)
					SS.append(5)
					RR.append(0)			
					RR.append(0)
				elif rand_inner_pattern==4:
					SS.append(2)
					SS.append(7)
					RR.append(0)			
					RR.append(0)
				elif rand_inner_pattern==5:
					SS.append(3)
					SS.append(4)
					RR.append(0)			
					RR.append(0)
				elif rand_inner_pattern==6:
					SS.append(3)
					SS.append(5)
					RR.append(0)			
					RR.append(1)
				elif rand_inner_pattern==7:
					SS.append(3)
					SS.append(7)
					RR.append(0)			
					RR.append(0)
				elif rand_inner_pattern==8:
					SS.append(6)
					SS.append(4)
					RR.append(0)			
					RR.append(0)
				elif rand_inner_pattern==9:
					SS.append(6)
					SS.append(5)
					RR.append(0)			
					RR.append(0)
				elif rand_inner_pattern==10:
					SS.append(6)
					SS.append(7)
					RR.append(0)			
					RR.append(0)
	RR = np.reshape(RR,(-1,1))	
	
	# preprocess data to have the right format	
	[S, O] = preprocess_data(SS, RR, model)
	
	return S,O

def data_construction(N=500, p_c=0.5, p_tr=0.8, model=None):
	
	N_tr = int(np.around(N*p_tr))	
	
	# data division in training and test subsets
	[S_tr, O_tr] = subset_construction(N_tr, p_c, model)
	[S_test, O_test] = subset_construction(N-N_tr, p_c, model)

	dic_stim = {'array([[1, 0, 0, 0, 0, 0, 0, 0]])':'1',
		    'array([[0, 1, 0, 0, 0, 0, 0, 0]])':'2',
		    'array([[0, 0, 1, 0, 0, 0, 0, 0]])':'A',
		    'array([[0, 0, 0, 1, 0, 0, 0, 0]])':'B',
		    'array([[0, 0, 0, 0, 0, 0, 1, 0]])':'C',
		    'array([[0, 0, 0, 0, 1, 0, 0, 0]])':'X',
		    'array([[0, 0, 0, 0, 0, 1, 0, 0]])':'Y',
		    'array([[0, 0, 0, 0, 0, 0, 0, 1]])':'Z'}
	
	if(model == '0' or model == '2'):
		dic_resp =  {'array([[1, 0]])':'L', 'array([[0, 1]])':'R', '0':'L', '1':'R'}
	elif(model == '1'):
		dic_resp =  {'array([[1, 0, 0, 1]])':'L', 'array([[0, 1, 1, 0]])':'R', '0':'L', '1':'R'}
	else:
		raise TypeError	

	return S_tr, O_tr, S_test, O_test, dic_stim, dic_resp


def data_modification_for_LSTM(S,O,dt=10):
	
	S_dim = np.shape(S)[1] 	
	sub_S = S[0:dt,:]
	# print(np.shape(sub_S))

	S_3D = np.reshape(sub_S,(1,dt,S_dim))
	O_2 = O[(dt-1):dt,:]

	for n in ((np.arange(np.shape(S)[0]-dt))+1):	

		seq = S[n:(n+dt),:]
		seq_3D = np.reshape(seq,(1, np.shape(seq)[0], np.shape(seq)[1]))	
		S_3D = np.concatenate((S_3D,seq_3D),axis=0)	
	
		O_2 = np.concatenate((O_2,O[(n+dt-1):(n+dt),:]),axis=0)	

	return S_3D, O_2