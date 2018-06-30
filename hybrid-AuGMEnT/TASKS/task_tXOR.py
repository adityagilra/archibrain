import numpy as np


def get_dictionary():
	dic_stim = {0:'1',1:'2',2:'A',3:'B'}
	dic_resp = {0:'L',1:'R'}

	return dic_stim, dic_resp


def construct_trial(tr_type=None):

	if tr_type==None:
		tr_type = np.random.choice(np.arange(4),1)
		
	s_tr = np.zeros((3,4))
	o_tr = np.zeros((1,2))

	if tr_type==0 or tr_type==3:
		o_tr[0,1] = 1
	else:
		o_tr[0,0] = 1

	if int(tr_type/2)==0:
		s_tr[0,0] = 1
	else:
		s_tr[0,1] = 1	
	if tr_type%2==0:
		s_tr[1,2] = 1
	else:
		s_tr[1,3] = 1

	return s_tr, o_tr

def data_construction(N_tr=500):	
	
	S_tr = np.zeros((N_tr,3,4))
	O_tr = np.zeros((N_tr,2))

	TRIAL_TYPES = np.random.choice(np.arange(4),(N_tr))

	for n in np.arange(N_tr):
		S_tr[n:(n+1),:,:], O_tr[n:(n+1),:] = construct_trial(TRIAL_TYPES[n])

	return S_tr, O_tr


def main():

	S, O = construct_trial(3)
	print(S)
	print(O)

	print('\n\n')
	S2, O2 = data_construction(2)
	print(S2)
	print(O2)

	d_s, d_r = get_dictionary()
	print(d_s)
	print(d_r)

#main()
