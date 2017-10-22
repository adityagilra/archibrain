import numpy as np

def construct_trial(num_loops,tr_types):
	
	leng = 2*num_loops
	S_tr = np.zeros((leng,4))
	O_tr = np.zeros((leng,2))
	O_tr[:,0] = 1 # default, all Non target

	# inner loops
	for i,tr in enumerate(tr_types):
		S_tr[2*i, np.floor(tr/2).astype(int)] = 1
		S_tr[2*i+1, 2+np.remainder(tr,2)] = 1
		if tr==0:
			O_tr[2*i+1,0]=0
			O_tr[2*i+1,1]=1 # correction from non-target
	return S_tr, O_tr

def data_construction(N_tr=500,p_target=0.5):	
	
	p_wrong = (1-p_target)/3
	p_targ = p_target

	NUMBER_INNER_LOOPS = 1
	tot_number = np.sum(NUMBER_INNER_LOOPS)
	RANDOM_PATTERNS = np.random.choice(np.arange(4),(tot_number), p=[p_targ,p_wrong,p_wrong,p_wrong])

	tot_length =  2*tot_number
	S_train = np.zeros((tot_length,4))
	O_train = np.zeros((tot_length,2))

	# data division in training and test subsets
	cont = 0
	cont_loop = 0
	for n in np.arange(N_tr):
		num_loops = NUMBER_INNER_LOOPS
		leng = 2*num_loops
		S_train[cont:(cont+leng),:], O_train[cont:(cont+leng),:] = construct_trial(num_loops,RANDOM_PATTERNS[cont_loop:(cont_loop+num_loops)])

		cont += leng
		cont_loop += num_loops

	return S_train, O_train


def main():

	S, O = construct_trial(3,[0,2,3])
	print(S)
	print(O)

#main()
