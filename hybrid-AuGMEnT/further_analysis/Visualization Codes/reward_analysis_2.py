import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec
import pylab

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
   
figx = 15
figy = 13
fig = plt.figure(figsize=(figx,figy))

fontTitle = 32
fontTicks = 24
fontLabel = 32

N_int = 50
N_cum = 10

R_esw = np.loadtxt('eps_greedy_soft_weighted/reward_3.txt')
R_es = np.loadtxt('eps_greedy_soft/reward_3.txt')
R_eu = np.loadtxt('eps_greedy_unif/reward_3.txt')
R_g = np.loadtxt('greedy/reward_3.txt')
R_sw = np.loadtxt('softmax_weighted/reward_3.txt')
R_s = np.loadtxt('softmax/reward_3.txt')

R_esw_sd = np.std(R_esw,axis=0,keepdims=True)
R_es_sd = np.std(R_es,axis=0,keepdims=True)
R_eu_sd = np.std(R_eu,axis=0,keepdims=True)
R_g_sd = np.std(R_g,axis=0,keepdims=True)
R_sw_sd = np.std(R_sw,axis=0,keepdims=True)
R_s_sd = np.std(R_s,axis=0,keepdims=True)

R_esw = np.mean(R_esw,axis=0,keepdims=True)
R_es = np.mean(R_es,axis=0,keepdims=True)
R_eu = np.mean(R_eu,axis=0,keepdims=True)
R_g = np.mean(R_g,axis=0,keepdims=True)
R_sw = np.mean(R_sw,axis=0,keepdims=True)
R_s = np.mean(R_s,axis=0,keepdims=True)

print(np.shape(R_esw))
L= np.shape(R_esw)[1]

R_esw_sd = np.reshape(moving_average(R_esw_sd,N_cum),(-1,1))
R_es_sd = np.reshape(moving_average(R_es_sd,N_cum),(-1,1))
R_eu_sd = np.reshape(moving_average(R_eu_sd,N_cum),(-1,1))
R_g_sd = np.reshape(moving_average(R_g_sd,N_cum),(-1,1))
R_sw_sd = np.reshape(moving_average(R_sw_sd,N_cum),(-1,1))
R_s_sd = np.reshape(moving_average(R_s_sd,N_cum),(-1,1))

R_esw = np.reshape(moving_average(R_esw,N_cum),(-1,1))
R_es = np.reshape(moving_average(R_es,N_cum),(-1,1))
R_eu = np.reshape(moving_average(R_eu,N_cum),(-1,1))
R_g = np.reshape(moving_average(R_g,N_cum),(-1,1))
R_sw = np.reshape(moving_average(R_sw,N_cum),(-1,1))
R_s = np.reshape(moving_average(R_s,N_cum),(-1,1))

REW = np.concatenate([R_esw, R_es, R_eu, R_g, R_sw, R_s],axis=1)
print(np.shape(REW))
REW_SD = np.concatenate([R_esw_sd, R_es_sd, R_eu_sd, R_g_sd, R_sw_sd, R_s_sd],axis=1)
print(np.shape(REW))

it = np.arange(L)*N_int
OPT_REW = 0.1*np.ones((L))*6
LABELS = ['eps-greedy-soft-weighted (P1)','eps-greedy-soft (P3)','eps-greedy-unif (P4)','greedy (P5)','softmax-weighted (P6)','softmax (P7)']
N_mod = len(LABELS)

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
print(colors)
COLORS = [colors[0],colors[2],colors[1],colors[5],colors[3],colors[4],colors[6]]

plt.plot(it,OPT_REW,linewidth=6,label='OPTIMAL',color=COLORS[0])
#plt.fill_between(it,OPT_REW,OPT_REW,alpha=0.1)
[plt.plot(it[(N_cum-1):],REW[:,i],linewidth=4,label=LABELS[i],color=COLORS[i+1]) for i in np.arange(N_mod)]
#[plt.fill_between(it[(N_cum-1):],REW[:,i]-REW_SD[:,i],np.minimum(REW[:,i]+REW_SD[:,i],0.6),alpha=0.1) for i in np.arange(N_mod)]
plt.plot(it[(N_cum-1):],REW[:,0],linewidth=4,color=COLORS[1])
#plt.fill_between(it[(N_cum-1):],REW[:,0]-REW_SD[:,0],np.minimum(REW[:,0]+REW_SD[:,0],0.6),color='orange',alpha=0.1)
plt.legend(fontsize=fontLabel)
plt.ylabel('Mean Reward Per Trial',fontsize=fontLabel)
plt.xlabel('Number of Trials', fontsize=fontLabel)
plt.xticks(fontsize=fontTicks)
plt.yticks(fontsize=fontTicks)

plt.show()

strsave='mean_reward.pdf'
fig.savefig(strsave)
