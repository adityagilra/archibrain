import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec
import pylab

np.set_printoptions(precision=2)

def compute_statistics(conv_str):

	C = np.loadtxt(conv_str)

	N_sim = np.shape(C)[0]
	C_s = 100*np.shape(C[np.where((C!=0)*(C<100000))])[0]/N_sim
	if C_s!=0:
		C_m = np.mean(C[(C!=0)*(C<100000)])
	else:
		C_m = None
		
	if C_s!=0:
		C_sd = np.std(C[(C!=0)*(C<100000)])
	else:
		C_sd = None		

	return C_s, C_m, C_sd
	
def compute_test_statistics(perc_str):

	P = np.loadtxt(perc_str)
	
	P_m = np.mean(P)
	P_sd = np.std(P)	
	
	return P_m, P_sd	
    
N_mod = 2

bet_l = 11
BET = np.array([0,2,4,6,8,10,12,14,16,18,20])

C_S = np.zeros((N_mod,bet_l))
C_M = np.zeros((N_mod,bet_l))	
C_SD = np.zeros((N_mod,bet_l))

for b_j in np.arange(np.shape(BET)[0]):
    string = 'eps_greedy_soft_weighted_t/conv_'+str(BET[b_j])+'.txt'
    C_S[0,b_j], C_M[0,b_j], C_SD[0,b_j] = compute_statistics(string)
    string = 'softmax_weighted/conv_'+str(BET[b_j])+'.txt'
    C_S[1,b_j], C_M[1,b_j], C_SD[1,b_j] = compute_statistics(string)

LABELS = ['eps-greedy-soft-weighted (P1)','softmax-weighted (P6)']

print('Cs:\n',C_S)
print('\n\nCm:\n',C_M)
print('\n\nCsd:\n',C_SD)

C_M = np.transpose(C_M)
C_SD = np.transpose(C_SD)

fontTitle = 32
fontTicks = 24
fontLabel = 30

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
print(colors)
COLORS = [colors[2],colors[4]]

figx = 16
figy = 12
fig = plt.figure(figsize=(figx,figy))

[plt.plot(BET,C_M[:,i],linewidth=6,label=LABELS[i],color=COLORS[i],marker='o',markersize=12) for i in np.arange(N_mod)]
[plt.fill_between(BET,np.maximum(C_M[:,i]-C_SD[:,i],0),C_M[:,i]+C_SD[:,i],alpha=0.2,color=COLORS[i]) for i in np.arange(N_mod)]
plt.legend(fontsize=fontLabel)
plt.ylabel('Mean Learning Time',fontsize=fontLabel)
plt.xlabel('Factor m', fontsize=fontLabel)
plt.xticks(BET,fontsize=fontTicks)
plt.yticks(fontsize=fontTicks)
plt.ylim([0,100000])

plt.show()

str_save = 'beta_analysis.pdf'
fig.savefig(str_save)

#fontTitle = 32
#fontTicks = 24
#fontLabel = 28

ind = np.arange(bet_l)
width=0.35

#C_S = np.transpose(C_S)
print(C_S)
print(np.shape(C_S))

fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(figx,figy))

rects1= ax.bar(ind-width/2, C_S[0,:]/100, width,color=COLORS[0],edgecolor='k')
rects2= ax.bar(ind+width/2, C_S[1,:]/100, width,color=COLORS[1],edgecolor='k')
ax.set_xlabel('Factor m',fontsize=fontLabel)
ax.set_ylabel('Fraction of Convergence Success',fontsize=fontLabel)
ax.set_xticks(ind)
ax.set_xticklabels(BET,fontsize=fontLabel)
ax.set_yticks([0,0.2,0.4,0.6,0.8,1])
ax.set_yticklabels((0,0.2,0.4,0.6,0.8,1),fontsize=fontTicks)
ax.legend((rects1[0], rects2[0]), ('eps-greedy-soft-weighted (P1)','softmax-weighted (P6)'),fontsize=fontLabel)
plt.ylim([0,1.3])

str_save = 'beta_barplot.pdf'

fig.savefig(str_save)
plt.show()


fig = plt.figure(figsize=(figx,figy))

P_e_m = np.zeros((N_mod,bet_l))
P_e_sd = np.zeros((N_mod,bet_l))
P_n_m = np.zeros((N_mod,bet_l))
P_n_sd = np.zeros((N_mod,bet_l))

for b_j in np.arange(np.shape(BET)[0]):

    string = 'eps_greedy_soft_weighted_t/perc_expl_'+str(BET[b_j])+'.txt'
    P_e_m[0,b_j], P_e_sd[0,b_j] = compute_test_statistics(string)
    string = 'eps_greedy_soft_weighted_t/perc_no_expl_'+str(BET[b_j])+'.txt'
    P_n_m[0,b_j], P_n_sd[0,b_j] = compute_test_statistics(string)
    
    string = 'softmax_weighted/perc_expl_'+str(BET[b_j])+'_tot.txt'
    P_e_m[1,b_j], P_e_sd[1,b_j] = compute_test_statistics(string)
    string = 'softmax_weighted/perc_no_expl_'+str(BET[b_j])+'_tot.txt'
    P_n_m[1,b_j], P_n_sd[1,b_j] = compute_test_statistics(string)
    
P_e_m = np.transpose(P_e_m)
P_e_sd = np.transpose(P_e_sd)
P_n_m = np.transpose(P_n_m)
P_n_sd = np.transpose(P_n_sd)

[plt.plot(BET,P_n_m[:,i],linewidth=6,label=LABELS[i],color=COLORS[i],marker='o',markersize=12) for i in np.arange(N_mod)]
[plt.fill_between(BET,np.maximum(P_n_m[:,i]-P_n_sd[:,i],0),P_n_m[:,i]+P_n_sd[:,i],alpha=0.2,color=COLORS[i]) for i in np.arange(N_mod)]
plt.legend(fontsize=fontLabel)
plt.ylabel('Mean Test Accuracy (Greedy policy)',fontsize=fontLabel)
plt.xlabel('Factor m', fontsize=fontLabel)
plt.xticks(BET,fontsize=fontTicks)
plt.yticks(fontsize=fontTicks)
plt.ylim([90,100])

plt.show()

str_save = 'beta_test_1.pdf'
fig.savefig(str_save)

fig = plt.figure(figsize=(figx,figy))

[plt.plot(BET,P_e_m[:,i],linewidth=6,label=LABELS[i],color=COLORS[i],marker='o',markersize=12) for i in np.arange(N_mod)]
[plt.fill_between(BET,np.maximum(P_e_m[:,i]-P_e_sd[:,i],0),P_e_m[:,i]+P_e_sd[:,i],alpha=0.2,color=COLORS[i]) for i in np.arange(N_mod)]
plt.legend(fontsize=fontLabel)
plt.ylabel('Mean Test Accuracy ($\epsilon$-greedy policy)',fontsize=fontLabel)
plt.xlabel('Factor m', fontsize=fontLabel)
plt.xticks(BET,fontsize=fontTicks)
plt.yticks(fontsize=fontTicks)
plt.ylim([90,100])

plt.show()

str_save = 'beta_test_2.pdf'
fig.savefig(str_save)
