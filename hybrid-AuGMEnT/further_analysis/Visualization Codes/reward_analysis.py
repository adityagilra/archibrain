import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec
import pylab

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
   
figx = 16
figy = 12
fig = plt.figure(figsize=(figx,figy))

fontTitle = 32
fontTicks = 24
fontLabel = 30

N_int = 50
N_cum = 4

R_esw = np.loadtxt('eps_greedy_soft_weighted/reward_3.txt')
R_es = np.loadtxt('eps_greedy_soft/reward_3.txt')
R_eu = np.loadtxt('eps_greedy_unif/reward_3.txt')
R_g = np.loadtxt('greedy/reward_3.txt')
R_sw = np.loadtxt('softmax_weighted/reward_3.txt')
R_s = np.loadtxt('softmax/reward_3.txt')

R_esw = np.mean(R_esw,axis=0,keepdims=True)
R_es = np.mean(R_es,axis=0,keepdims=True)
R_eu = np.mean(R_eu,axis=0,keepdims=True)
R_g = np.mean(R_g,axis=0,keepdims=True)
R_sw = np.mean(R_sw,axis=0,keepdims=True)
R_s = np.mean(R_s,axis=0,keepdims=True)

print(np.shape(R_esw))
L= np.shape(R_esw)[1]

R_esw = np.reshape(moving_average(R_esw,N_cum),(-1,1))
R_es = np.reshape(moving_average(R_es,N_cum),(-1,1))
R_eu = np.reshape(moving_average(R_eu,N_cum),(-1,1))
R_g = np.reshape(moving_average(R_g,N_cum),(-1,1))
R_sw = np.reshape(moving_average(R_sw,N_cum),(-1,1))
R_s = np.reshape(moving_average(R_s,N_cum),(-1,1))

REW = np.concatenate([R_esw, R_es, R_eu, R_g, R_sw, R_s],axis=1)
print(np.shape(REW))

it = np.arange(L)*N_int
OPT_REW = 0.1*np.ones((L))*6
LABELS = ['eps-greedy-soft-g','eps-greedy-soft','eps-greedy-unif','greedy','softmax-g','softmax']
N_mod = len(LABELS)

plt.plot(it,OPT_REW,linewidth=6,label='OPTIMAL')
[plt.plot(it[(N_cum-1):],REW[:,i],linewidth=4,label=LABELS[i]) for i in np.arange(N_mod)]
plt.plot(it[(N_cum-1):],REW[:,0],linewidth=4,color='orange')
plt.legend(fontsize=fontLabel)
plt.ylabel('Mean Reward Per Trial',fontsize=fontLabel)
plt.xlabel('Number of trials', fontsize=fontLabel)

plt.show()

strsave='mean_reward.png'
fig.savefig(strsave)
