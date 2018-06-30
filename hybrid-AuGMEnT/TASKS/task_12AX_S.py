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

def construct_trial(dig, num_loops,tr_types):
	
	leng = 1 + 2*num_loops
	S_tr = np.zeros((leng,8))
	O_tr = np.zeros((leng,2))
	O_tr[:,0] = 1 # default, all Non target

	# digit
	S_tr[0,dig] = 1

	# inner loops
	for i,tr in enumerate(tr_types):
		S_tr[2*i+1, 2+np.floor(tr/3).astype(int)] = 1
		S_tr[2*(i+1), 5+np.remainder(tr,3)] = 1
		if (dig==0 and tr==0) or (dig==1 and tr==4):
			O_tr[2*(i+1),0]=0
			O_tr[2*(i+1),1]=1 # correction from non-target
	return S_tr, O_tr

def data_construction(N_tr=500,p_target=0.5):	
	
	p_wrong = (1-p_target)/7
	p_targ = p_target/2

	S_DIG = np.random.choice(np.arange(2), (N_tr))
	NUMBER_INNER_LOOPS = 1
	tot_number = np.sum(NUMBER_INNER_LOOPS)
	RANDOM_PATTERNS = np.random.choice(np.arange(9),(tot_number), p=[p_targ,p_wrong,p_wrong, p_wrong,p_targ,p_wrong, p_wrong,p_wrong,p_wrong])

	tot_length = N_tr + 2*tot_number
	S_train = np.zeros((tot_length,8))
	O_train = np.zeros((tot_length,2))

	# data division in training and test subsets
	cont = 0
	cont_loop = 0
	for n in np.arange(N_tr):
		num_loops = NUMBER_INNER_LOOPS
		leng = 1 + 2*num_loops
		S_train[cont:(cont+leng),:], O_train[cont:(cont+leng),:] = construct_trial(S_DIG[n],num_loops,RANDOM_PATTERNS[cont_loop:(cont_loop+num_loops)])
		cont += leng
		cont_loop += num_loops

	return S_train, O_train


def main():

	S, O = construct_trial(0,3,[0,4,7])
	print(S)
	print(O)
	
	S, O = data_construction(3)
	print(S)
	print(O)

#main()
