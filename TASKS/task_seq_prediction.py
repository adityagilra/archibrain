import numpy as np

def preprocess_data(S, R, model=None):

	leng = np.shape(S)[0]
	num_task = np.max(R)+1
	S_new = np.zeros((leng,num_task))
	seq = np.arange(leng)
	
	S_new[seq,S] = 1
	R_new = []

	if(model == '0' or model == '2'):
		for r in R:
		    if r == 0:
		        R_new.append([1,0,0,0])
		    elif r == 1:
		        R_new.append([0,1,0,0])
		    elif r == 2:
		        R_new.append([0,0,1,0])
		    elif r == 3:
		        R_new.append([0,0,0,1])

	elif(model == '1'):
		for r in R:
		    if r == 0:
		        R_new.append([1,0,0,1,0,1,0,1])
		    elif r == 1:
		        R_new.append([0,1,1,0,0,1,0,1])
		    elif r == 2:
		        R_new.append([0,1,0,1,1,0,0,1])
		    elif r == 3:
		        R_new.append([0,1,0,1,0,1,1,0])
	else:
		raise TypeError

	R_new = np.asarray(R_new)

	return S_new, R_new

def subset_construction(N=100, p=0.5, model=None):
	
	np.random.seed()

	p_A = p
	p_X = 1-p

	x = np.random.choice(np.arange(2), size=(N,1), p=[p_A, p_X])

	SS = []
	RR = []

	for i in range(N):
		if x[i] == 0:
			SS.append(0)
			SS.append(1)
			SS.append(2)

			RR.append(1)
			RR.append(2)
			RR.append(0)

		else:
			SS.append(3)
			SS.append(1)
			SS.append(2)

			RR.append(1)
			RR.append(2)
			RR.append(3)

	S, O = preprocess_data(SS, RR, model)
	return S, O


def data_construction(N=100, p=0.5, tr_perc=0.8, model=None):

	N_tr = int(np.around(N*tr_perc))

	S_train, O_train = subset_construction(N_tr, p, model)
	S_test, O_test = subset_construction(N-N_tr, p, model)

	dic_stim = {'array([[1, 0, 0, 0]])':'A',
		    'array([[0, 1, 0, 0]])':'B',
		    'array([[0, 0, 1, 0]])':'C',
		    'array([[0, 0, 0, 1]])':'X'}
	
	if(model == '0' or model == '2'):
		dic_resp = {'array([[1, 0, 0, 0]])':'D',
			    'array([[0, 1, 0, 0]])':'B',
			    'array([[0, 0, 1, 0]])':'C',
			    'array([[0, 0, 0, 1]])':'Y',
			    '0':'D', '1':'B', '2':'C', '3':'Y'}
	elif(model == '1'):
		dic_resp = {'array([[1, 0, 0, 1, 0, 1, 0, 1]])':'D',
			    'array([[0, 1, 1, 0, 0, 1, 0, 1]])':'B',
			    'array([[0, 1, 0, 1, 1, 0, 0, 1]])':'C',
			    'array([[0, 1, 0, 1, 0, 1, 1, 0]])':'Y',
			    '0':'D', '1':'B', '2':'C', '3':'Y'}
	else:
		raise TypeError

	
	return S_train, O_train, S_test, O_test, dic_stim, dic_resp

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