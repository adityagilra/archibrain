import numpy as np

def preprocess_data(S, R, model=None):

	S_new = np.where(S == 1, [1,0],[0,1])

	if(model == '0' or model == '2'):
		R_new = np.where(R == 'L', [1,0],[0,1])	
	elif(model == '1'):
		R_new = np.where(R == 'L', [1,0,0,1],[0,1,1,0])
	else:
		raise TypeError
	
	return S_new, R_new

def subset_data(S, O, training_perc=0.8):

	sz = np.shape(O)[0]
	idx = int(np.around(sz*training_perc))	

	# Distinction in training and test sets
	S_train = S[:idx, :]
	O_train = O[:idx, :]
	S_test = S[idx:, :]
	O_test = O[idx:, :]

	return S_train, O_train, S_test, O_test

# construction of the dataset
def data_construction(N=1000, p1=0.7, p2=0.3, training_perc=0.8, model=None):

	SS = np.random.choice(np.arange(3), (N,1), p=[0,p1,p2]) 
	RR = np.where(SS == 1, 'L', 'R')

	[S, O] = preprocess_data(SS, RR, model)

	[S_tr, O_tr, S_test, O_test] = subset_data(S, O, training_perc)

	dic_stim = {'array([[1, 0]])':'1', 'array([[0, 1]])':'2'}
	
	if(model == '0' or model == '2'):
		dic_resp =  {'array([[1, 0]])':'L', 'array([[0, 1]])':'R','0':'L','1':'R'}	
		
	elif(model == '1'):
		dic_resp =  {'array([[1, 0, 0, 1]])':'L', 'array([[0, 1, 1, 0]])':'R',}	

	else:
		raise TypeError		

	return S_tr, O_tr, S_test, O_test, dic_stim, dic_resp