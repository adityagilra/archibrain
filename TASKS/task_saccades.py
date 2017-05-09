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

def build_trial(tr, model=None):
	
	if(model == '0'):
		# start
		S = np.zeros((1,4)) # empty screen
		O = np.zeros((1,3)) # no action
		
		# fix
		if tr==0 or tr==1:
			S = np.concatenate((S, np.array([[1,0,0,0]]), np.array([[1,0,0,0]])))  # Pro-saccade trials (black)
		else:
			S = np.concatenate((S, np.array([[0,1,0,0]]), np.array([[0,1,0,0]])))  # Anti-saccade trials (white)
		O = np.concatenate((O, np.array([[0,1,0]]), np.array([[0,1,0]])))

		# cue
		if tr==0:
			S = np.concatenate((S,np.array([[1,0,1,0]])))  # Pro-saccade + Left (PL)
		elif tr==1:
			S = np.concatenate((S,np.array([[1,0,0,1]])))  # Pro-saccade + Right (PR)
		elif tr==2:
			S = np.concatenate((S,np.array([[0,1,1,0]])))  # Anti-saccade + Left (AL)
		elif tr==3:
			S = np.concatenate((S,np.array([[0,1,0,1]])))  # Anti-saccade + Right (AR)
		O = np.concatenate((O,np.zeros((1,3))))	 # no action

		# delay (two timesteps)
		if tr==0 or tr==1:
			S = np.concatenate((S,np.array([[1,0,0,0]]),np.array([[1,0,0,0]])))  
		else:
			S = np.concatenate((S,np.array([[0,1,0,0]]),np.array([[0,1,0,0]])))  
		O = np.concatenate((O,np.array([[0,0,0]]),np.array([[0,0,0]])))  # no action

		# go
		S = np.concatenate((S,np.zeros((1,4)) ))  # empty screen
		if tr==0 or tr==3:
			O = np.concatenate((O, np.array([[1,0,0]])) ) # L = response for PL or AR 
		else:
			O = np.concatenate((O, np.array([[0,0,1]])) )  # R = response for PR or AL

	elif(model == '1'):
		# start
		S = np.array([[1,0,0,0,0]]) # empty
		O = np.zeros((1,6)) # no action
		
		# fix
		if tr==0 or tr==1:
			S = np.concatenate((S, np.array([[0,1,0,0,0]]), np.array([[0,1,0,0,0]])))  # Pro-saccade trials (black)
		else:
			S = np.concatenate((S, np.array([[0,0,1,0,0]]), np.array([[0,0,1,0,0]])))  # Anti-saccade trials (white)
		O = np.concatenate((O, np.array([[0,1,1,0,0,1]]), np.array([[0,1,1,0,0,1]])))

		# cue
		if tr==0 or tr==2:
			S = np.concatenate((S,np.array([[0,0,0,1,0]])))  # Left (L)
		elif tr==1 or tr==3:
			S = np.concatenate((S,np.array([[0,0,0,0,1]])))  # Right (R)
		O = np.concatenate((O,np.zeros((1,6))))	 # no action

		# delay (two timesteps)
		if tr==0 or tr==1:
			S = np.concatenate((S,np.array([[0,1,0,0,0]]),np.array([[0,1,0,0,0]])))  
		else:
			S = np.concatenate((S,np.array([[0,0,1,0,0]]),np.array([[0,0,1,0,0]])))  
		O = np.concatenate((O,np.array([[0,0,0,0,0,0]]),np.array([[0,0,0,0,0,0]])))  # no action

		# go
		S = np.concatenate((S,np.array([[1,0,0,0,0]]) ))  # empty screen
		if tr==0 or tr==3:
			O = np.concatenate((O, np.array([[1,0,0,1,0,1]])) ) # L = response for PL or AR 
		else:
			O = np.concatenate((O, np.array([[0,1,0,1,1,0]])) )  # R = response for PR or AL

	elif(model == '2'):
		# start
		S = np.array([[1,0,0,0,0]]) # empty screen
		O = np.zeros((1,2)) # no action
		
		# fix
		if tr==0 or tr==1:
			S = np.concatenate((S,np.array([[0,1,0,0,0]])))  # Pro-saccade trials (black)
		else:
			S = np.concatenate((S,np.array([[0,0,1,0,0]])))  # Anti-saccade trials (white)
		O = np.concatenate((O,np.array([[0,0]])))

		# cue
		if tr==0 or tr==2:
			S = np.concatenate((S,np.array([[0,0,0,1,0]])))  # Left (L)
		elif tr==1 or tr==3:
			S = np.concatenate((S,np.array([[0,0,0,0,1]])))  # Right (R)
		O = np.concatenate((O,np.zeros((1,2))))	 # no action

		# delay (two timesteps)
		if tr==0 or tr==1:
			S = np.concatenate((S,np.array([[0,1,0,0,0]]),np.array([[0,1,0,0,0]])))  
		else:
			S = np.concatenate((S,np.array([[0,0,1,0,0]]),np.array([[0,0,1,0,0]])))  
		O = np.concatenate((O,np.array([[0,0]]),np.array([[0,0]])))  # no action

		# go
		S = np.concatenate((S,np.array([[1,0,0,0,0]]) ))  # empty screen
		if tr==0 or tr==3:
			O = np.concatenate((O, np.array([[1,0]])) ) # L = response for PL or AR 
		else:
			O = np.concatenate((O, np.array([[0,1]])) )  # R = response for PR or AL
	else:
		raise TypeError	

	return S, O


# construction of the dataset
def data_construction(N=500, perc_training=0.8, model=None):

	np.random.seed(1234)

	Trials = np.random.choice(np.arange(4), (N,1))
	idx = int(np.around(N*perc_training))	
	Trials_training = Trials[:idx,0]
	Trials_test = Trials[idx:,0] 
	S_train, O_train = build_trial(Trials_training[0], model)
	S_test, O_test = build_trial(Trials_test[0], model)
	for tr in Trials_training[1:]:
		S_tr, O_tr = build_trial(tr, model)
		S_train = np.concatenate((S_train,S_tr))
		O_train = np.concatenate((O_train,O_tr))
	for tr in Trials_test[1:]:
		S_tst, O_tst = build_trial(tr, model)		
		S_test = np.concatenate((S_test,S_tst))
		O_test = np.concatenate((O_test,O_tst))

	if(model == '0'):
		dic_stim = {'array([[0, 0, 0, 0]])':'empty',
			    'array([[1, 0, 0, 0]])':'P',
			    'array([[0, 1, 0, 0]])':'A',
			    'array([[1, 0, 1, 0]])':'PL',
			    'array([[1, 0, 0, 1]])':'PR',
			    'array([[0, 1, 1, 0]])':'AL',
			    'array([[0, 1, 0, 1]])':'AR'}
		dic_resp =  {'array([[0, 0, 0]])':'None','array([[1, 0, 0]])':'L', 'array([[0, 1, 0]])':'F','array([[0, 0, 1]])':'R', '0':'L','1':'F','2':'R'}			

	elif(model == '1'):
		dic_stim = {'array([[1, 0, 0, 0, 0]])':'empty',
		    	'array([[0, 1, 0, 0, 0]])':'P',
		    	'array([[0, 0, 1, 0, 0]])':'A',
		    	'array([[0, 0, 0, 1, 0]])':'L',
		    	'array([[0, 0, 0, 0, 1]])':'R'}		
		dic_resp =  {'array([[0, 0, 0, 0, 0, 0]])':'None','array([[1, 0, 0, 1, 0, 1]])':'L', 'array([[0, 1, 1, 0, 0, 1]])':'F','array([[0, 1, 0, 1, 1, 0]])':'R', '0':'L','1':'F','2':'R'}			

	elif(model == '2'):
		dic_stim = {'array([[1, 0, 0, 0, 0]])':'empty',
		    	'array([[0, 1, 0, 0, 0]])':'P',
		    	'array([[0, 0, 1, 0, 0]])':'A',
		    	'array([[0, 0, 0, 1, 0]])':'L',
		    	'array([[0, 0, 0, 0, 1]])':'R'}
		dic_resp =  {'array([[1, 0]])':'L', 'array([[0, 1]])':'R', '0':'L','1':'R'}
	else:
		raise TypeError	

	return S_train, O_train, S_test, O_test, dic_stim, dic_resp


def data_modification_for_LSTM(S,O,dt=10):
	
	S_dim = np.shape(S)[1]

	S_3D = np.reshape(S,(-1,dt,S_dim))

	go_phase = dt + np.arange(np.shape(S_3D)[0])*dt -1
	O_2 = O[go_phase,:]

	return S_3D, O_2