## SACCADE-ANTISACCADE TASK 
## The Saccade-Antisaccade task consists in a sequence of multi-step trials where the final goal is to direct the eye movement according to previous cues projected on a screen.
## The cues are essentialy of two types:
##	- fixation mark : squaree at the center of the screen; if it is black it is a pro-saccade trial (P), if it is white it is an anti-saccade one (A);
##	- location cue: circle at the side of the screen; it can be either at the left side (L) or at the right side (R) of the screen
## The test-taker has 3 possible activities: F=front,L=left, R=right.
## In case of a pro-saccade trial, the eye movement has to be in the same direction as the location cue (PL or PR); otherwise, it has to be in the opposite direction (AL or AR).
## Each trial is composed by different phases:
##	- START: empty screen
##	- FIX: fixation mark appears; the test-taker has to fix it for two consecutive timesteps to have a first reward r_f (F selected twice)
##	- CUE: location cue appears together with fixation mark
##	- DELAY: location cue disappears for two timesteps to test the memory delay
##      - GO: fixation mark disappears as well (empty screen) and the subject has to solve the task (it has up to 8 timesteps to answer L or R)
##
## The implementation follows the rules from the paper "How Attention Can Create Synaptic Tags for the Learning of Working Memories in Sequential Tasks"
## by J. Rombouts, S. Bohte, P. Roeffsema.
##
## AUTHOR: Marco Martinolli
## DATE: 06.04.2017


import numpy as np

def build_trial(tr):
	
	S = np.zeros((6,5))
	O = np.zeros((6,3))
	
	# start
	S[0,0]=1

	# fix
	if tr==0 or tr==1:
		S[1,1] = 1
	else:
		S[1,2] = 1
	O[1,1] = 1

	# cue
	if tr==0 or tr==2:
		S[2,3] = 1
	else:
		S[2,4] = 1
	

	# delay (two timesteps)
	if tr==0 or tr==1:
		S[3:5,1] = 1
	else:
		S[3:5,2] = 1
	
	# go
	S[5,0]=1
	if tr==0 or tr==3:
		O[5,0] = 1
	else:
		O[5,2] = 1

	return S, O



# construction of the dataset
def data_construction(N=500,perc_training=0.8):

	Trials = np.random.choice(np.arange(4), (N,1))
	idx = int(np.around(N*perc_training))
	Trials_training = Trials[:idx,0]

	S_train = np.zeros((idx,6,5))
	O_train = np.zeros((idx,6,3))
	for i,tr in enumerate(Trials_training):
		S_train[i,:,:], O_train[i,:,:] =  build_trial(tr)

	if perc_training!=1:
		Trials_test = Trials[idx:,0] 
		S_test = np.zeros((N-idx,6,5))
		O_test = np.zeros((N-idx,6,3))
		for i, tr in enumerate(Trials_test):
			S_test[i,:,:], O_test[i,:,:] =  build_trial(tr)
	else:
		S_test, O_test = None, None


	dic_stim = {'array([[1, 0, 0, 0, 0]])':'empty',
		    'array([[0, 1, 0, 0, 0]])':'P',
		    'array([[0, 0, 1, 0, 0]])':'A',
		    'array([[0, 0, 0, 1, 0]])':'L',
		    'array([[0, 0, 0, 0, 1]])':'R'}
	dic_resp =  {'array([[1, 0, 0]])':'L', 'array([[0, 1, 0]])':'F','array([[0, 0, 1]])':'R', '0':'L','1':'F','2':'R'}			

	return S_train,O_train,S_test,O_test,dic_stim,dic_resp


def main():
	S_train,O_train,_,_,_,_=data_construction(1,1)
	print(S_train)
	print(O_train)

main()
