import numpy as np

def construct_trial(dist=2,p=0.5):

	time_length = dist+1
	num_letters = dist+2
	
	S_tr = np.zeros((time_length,num_letters))
	for i in np.arange(time_length-1)+1:
		S_tr[i,i]=1
	O_tr = np.zeros(2)

	p_A = p
	p_X = 1-p

	x = np.random.choice(np.arange(2), size=(1), p=[p_A, p_X])

	if x==0:
		S_tr[0,0]=1
		O_tr[0]=1
	else:
		S_tr[0,num_letters-1]=1
		O_tr[1]=1
	
	return S_tr,O_tr


def get_dictionary(dist):
	
	LETTERS = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','Z']

	dic_stim = {}
	for l_id in np.arange(dist+1): 
		dic_stim[l_id] = LETTERS[l_id]
	dic_stim[dist+1]='X'
	
	dic_resp = {0:LETTERS[dist+1], 1:'Y'}
	
	return dic_stim, dic_resp
	

def subset_construction(N=100, dist=2,p=0.5):

	S = np.zeros((N,3,4),dtype=int)
	O = np.zeros((N,2),dtype=int)

	for tr in np.arange(N):
		S[tr,:,:],O[tr,:] = construct_trial(dist,p)
	
	return S, O	


def data_construction(N=100, dist=2, p=0.5, tr_perc=0.8):

	np.random.seed()

	N_tr = int(np.around(N*tr_perc))

	S_train, O_train = subset_construction(N_tr, dist, p)
	S_test, O_test = subset_construction(N-N_tr, dist, p)

	dic_stim, dic_resp = get_dictionary(dist)
	
	return S_train, O_train, S_test, O_test, dic_stim, dic_resp


def main():

	d=5
	p=0.5
	S, O = construct_trial(d,p)
	d_s, d_o = get_dictionary(d)
	print(S)
	print(O)
	print('INP: ',d_s[np.argmax(S[0,:])],'\t OUT: ',d_o[np.argmax(O)])

main()
