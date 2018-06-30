# List of minor analysis

-AuGMEnT_12AX_equation_comparison.py: Variation of the memory dynamics where synaptic trace is updated immediately and fed in the memory in the place of the transient sensory stimulus. See Supplementary Material C

-AuGMEnT_12AX_no_reset.py: Analysis of hybrid performance in case of no memory and tag resets after each outer loop. Notice that the outer loop-based structure of the code is maintained. See Supplementary Material E

-AuGMEnT_12AX_triplets.py: Variant of 12AX task where each trial is composed of 1 to 3 triplets with no memory reset in betweeen (but maintaining the reset at the beginning of each trial). See Supplementary Material E

-AuGMEnT_12AX_beta.py: Analysis of the effect of the beta parameter in the stabilizing function g(t). Here, beta corresponds to the m parameter at the numerator of g(t) discussed in the paper. See Supplementary Material A


N.B. Notice that you cannot run these codes from this position. Links to input (TASKS) and output folders (DATA, IMAGES,..) have to be coherent with the current directory from where you run the simualtion. Please before running any analysis or simulation, please check the relative/absolute paths.
