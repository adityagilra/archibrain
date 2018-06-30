import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec
import pylab

np.set_printoptions(precision=2)

def compute_statistics(conv_str,perc_str, perc_str_e):
	C = np.loadtxt(conv_str)
	P = np.loadtxt(perc_str)
	P_e = np.loadtxt(perc_str_e)

	N_sim = np.shape(C)[0]
	C_s = 100*np.count_nonzero(C)/N_sim
	if np.count_nonzero(C)!=0:
		C_m = np.sum(C)/np.count_nonzero(C)
	else:
		C_m = None

	P_m = np.mean(P)
	Pe_m = np.mean(P_e)

	return C_s, C_m, P_m, Pe_m

def compute_std(conv_str,perc_str, perc_str_e):
	C = np.loadtxt(conv_str)
	P = np.loadtxt(perc_str)
	P_e = np.loadtxt(perc_str_e)

	N_sim = np.shape(C)[0]
	if np.count_nonzero(C)!=0:
		C_sd = np.std(C[C!=0])
	else:
		C_sd = None

	P_sd = np.std(P)
	Pe_sd = np.std(P_e)
	
	return C_sd, P_sd, Pe_sd
    
    
N_mod = 7
C_S = np.zeros((N_mod,1))
C_M = np.zeros((N_mod,1))
P_M = np.zeros((N_mod,1))
P_E = np.zeros((N_mod,1))		
	
C_S[0], C_M[0], P_M[0], P_E[0] = compute_statistics('eps_greedy_soft_weighted/conv_tot.txt', 'eps_greedy_soft_weighted/perc_tot.txt','eps_greedy_soft_weighted/perc_expl_3.txt')
C_S[1], C_M[1], P_M[1], P_E[1] = compute_statistics('eps_greedy_soft/conv_tot.txt', 'eps_greedy_soft/perc_tot.txt','eps_greedy_soft/perc_expl_3.txt')
C_S[2], C_M[2], P_M[2], P_E[2] = compute_statistics('eps_greedy_unif/conv_tot.txt', 'eps_greedy_unif/perc_tot.txt','eps_greedy_unif/perc_expl_3.txt')
C_S[3], C_M[3], P_M[3], P_E[3] = compute_statistics('greedy/conv_tot.txt', 'greedy/perc_tot.txt','greedy/perc_expl_3.txt')
C_S[4], C_M[4], P_M[4], P_E[4] = compute_statistics('softmax_weighted/conv_tot.txt', 'softmax_weighted/perc_tot.txt', 'softmax_weighted/perc_expl_3.txt')
C_S[5], C_M[5], P_M[5], P_E[5] = compute_statistics('softmax/conv_tot.txt', 'softmax/perc_tot.txt','softmax/perc_expl_3.txt')
C_S[6], C_M[6], P_M[6], P_E[6] = compute_statistics('eps_greedy_soft_weighted_e/conv_tot.txt', 'eps_greedy_soft_weighted_e/perc_no_expl_tot.txt','eps_greedy_soft_weighted_e/perc_expl_tot.txt')

LABELS = ['ESW_g','ES','EU','G','SW','S','ESW_e']

STAT = np.concatenate([C_S,C_M,P_M,P_E],axis=1)

print(STAT)

print('\n\n\n\n')
C_SD = np.zeros((N_mod,1))
P_SD = np.zeros((N_mod,1))
P_eSD = np.zeros((N_mod,1))		
	
C_SD[0], P_SD[0], P_eSD[0] = compute_std('eps_greedy_soft_weighted/conv_tot.txt', 'eps_greedy_soft_weighted/perc_tot.txt','eps_greedy_soft_weighted/perc_expl_3.txt')
C_SD[1], P_SD[1], P_eSD[1] = compute_std('eps_greedy_soft/conv_tot.txt', 'eps_greedy_soft/perc_tot.txt','eps_greedy_soft/perc_expl_3.txt')
C_SD[2], P_SD[2], P_eSD[2] = compute_std('eps_greedy_unif/conv_tot.txt', 'eps_greedy_unif/perc_tot.txt','eps_greedy_unif/perc_expl_3.txt')
C_SD[3], P_SD[3], P_eSD[3] = compute_std('greedy/conv_tot.txt', 'greedy/perc_tot.txt','greedy/perc_expl_3.txt')
C_SD[4], P_SD[4], P_eSD[4] = compute_std('softmax_weighted/conv_tot.txt', 'softmax_weighted/perc_tot.txt', 'softmax_weighted/perc_expl_3.txt')
C_SD[5], P_SD[5], P_eSD[5] = compute_std('softmax/conv_tot.txt', 'softmax/perc_tot.txt','softmax/perc_expl_3.txt')
C_SD[6], P_SD[6], P_eSD[6] = compute_std('eps_greedy_soft_weighted_e/conv_tot.txt', 'eps_greedy_soft_weighted_e/perc_no_expl_tot.txt','eps_greedy_soft_weighted_e/perc_expl_tot.txt')

STAT_SD = np.concatenate([C_SD,P_SD,P_eSD],axis=1)

print(STAT_SD)
