import numpy as np

# Args:
#      perc_target: Probability to have a target sequence.
#      min_loops: Lower limit of inner loops per trial.
#      max_loops: Upper limit of inner loops per trial.


def construct_trial():
	
	# Samples each batch index's sequence length and the number of repeats.
	trial_type = np.random.choice(np.arange(4))		# 0: PL   1: PR   2: AL   3: AR
    
	total_length = 6
	S = 5

	obs_vec = np.zeros((total_length,S), dtype=int)
	targ_vec = np.zeros((total_length,3), dtype=int)

	# start
	obs_vec[0,0] = 1
	
	# fixation mark
	if trial_type==0 or trial_type==1:
		obs_vec[1,1]=1
		obs_vec[3:5,1]=1
	else:
		obs_vec[1,2]=1
		obs_vec[3:5,2]=1

	# location cue
	if trial_type==0 or trial_type==2:
		obs_vec[2,3]=1
	else:
		obs_vec[2,4]=1
	    
	# go
	obs_vec[total_length-1,0] = 1
	
	# target
	if trial_type==0 or trial_type==3:
		targ_vec[total_length-1,0]=1
	else:
		targ_vec[total_length-1,2]=1
	targ_vec[:total_length-1,1]=1
	
	return obs_vec, targ_vec, trial_type


def main():

	dic_stim = {0:'empty', 1:'P', 2:'A', 3:'L', 4:'R'}
	dic_resp =  {0:'L', 1:'F',2:'R'}	

	o, t = construct_trial()

	print(o)
	print(np.vectorize(dic_stim.get)(np.argmax(o,axis=1)))
	print(np.vectorize(dic_resp.get)(np.argmax(t,axis=1)))


#main()
