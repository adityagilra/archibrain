import numpy as np

# Args:
#      perc_target: Probability to have a target sequence.
#      min_loops: Lower limit of inner loops per trial.
#      max_loops: Upper limit of inner loops per trial.


def construct_trial(perc_target, min_loops, max_loops):

    p_w = (1-perc_target)/7
    p_t = perc_target/2
    prob = [p_t,p_w,p_w, p_w,p_t,p_w, p_w,p_w,p_w]

    # Samples each batch index's sequence length and the number of repeats.
    num_loops = np.random.choice(np.arange(max_loops)) + min_loops
    total_length = num_loops*2 + 1

    digit_cue = np.random.choice(np.arange(2))
    pattern_types = np.random.choice(np.arange(9),(num_loops), p=prob)

    obs_vec = np.zeros((total_length), dtype=int)
    targ_vec = np.zeros((total_length), dtype=int)

    obs_vec[0] = digit_cue
    targ_vec[0] = 0

    for l in np.arange(num_loops):
       ind = 2*l+1
       o, t = construct_pattern(digit_cue,pattern_types[l])
       obs_vec[ind:ind+2] = np.reshape(o,(-1))
       targ_vec[ind:ind+2] = np.reshape(t,(-1))
    
    obs = one_hot(obs_vec, size=8)
    targ = one_hot(targ_vec, size=2)

    return obs, targ


def construct_pattern(digit,pattern_id):

    if (digit==0 and pattern_id==0) or (digit==1 and pattern_id==4):
          tar_p = np.array([0, 1])
    else:
          tar_p = np.array([0, 0])

    obs_p = np.array([[np.floor_divide(pattern_id,3) + 2], [np.remainder(pattern_id,3) + 5]])

    return obs_p, tar_p


def one_hot(indexes, size):

    N = np.shape(indexes)[0]
    vec = np.zeros((N,size), dtype=np.float32)
    vec[np.arange(N),indexes] = 1.0
 
    return vec

def main():
    dic_stim = {0:'1', 1:'2', 2:'A', 3:'B', 4:'C', 5:'X', 6:'Y', 7:'Z'}
    dic_resp = {0:'L', 1:'R'}
    o, t = construct_trial(0.5, 1, 4)

    print(o)
    print(np.vectorize(dic_stim.get)(np.argmax(o,axis=1)))
    print(np.vectorize(dic_resp.get)(np.argmax(t,axis=1)))

main()
