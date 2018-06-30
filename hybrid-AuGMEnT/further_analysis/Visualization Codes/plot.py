import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec
import pylab

N_sim = 20
N_leng = 10

str_esw =  'eps_greedy_soft_weighted/error_3.txt'
X_esw = np.loadtxt(str_esw)
print(np.shape(X_esw))
X_esw = np.mean(X_esw,axis=0)
print(np.shape(X_esw))
X_esw = np.reshape(X_esw,(-1,N_leng))
print(np.shape(X_esw))
X_esw_sd = np.std(X_esw,axis=1)
X_esw = np.mean(X_esw,axis=1)
print(np.shape(X_esw))

str_es =  'eps_greedy_soft/error_3.txt'
X_es = np.loadtxt(str_es)
X_es = np.mean(X_es,axis=0)
X_es = np.reshape(X_es,(-1,N_leng))
X_es_sd = np.std(X_es,axis=1)
X_es = np.mean(X_es,axis=1)

str_eu =  'eps_greedy_unif/error_3.txt'
X_eu = np.loadtxt(str_eu)
X_eu = np.mean(X_eu,axis=0)
X_eu = np.reshape(X_eu,(-1,N_leng))
X_eu_sd = np.std(X_eu,axis=1)
X_eu = np.mean(X_eu,axis=1)

str_g =  'greedy/error_3.txt'
X_g = np.loadtxt(str_g)
X_g = np.mean(X_g,axis=0)
X_g = np.reshape(X_g,(-1,N_leng))
X_g_sd = np.std(X_g,axis=1)
X_g = np.mean(X_g,axis=1)

str_sw =  'softmax_weighted/error_3.txt'
X_sw = np.loadtxt(str_sw)
X_sw = np.mean(X_sw,axis=0)
X_sw = np.reshape(X_sw,(-1,N_leng))
X_sw_sd = np.std(X_sw,axis=1)
X_sw = np.mean(X_sw,axis=1)

str_s =  'softmax/error_3.txt'
X_s = np.loadtxt(str_s)
X_s = np.mean(X_s,axis=0)
X_s = np.reshape(X_s,(-1,N_leng))
X_s_sd = np.std(X_s,axis=1)
X_s = np.mean(X_s,axis=1)

X = np.concatenate([X_esw, X_es, X_eu, X_g, X_sw, X_s])
X_sd = np.concatenate([X_esw_sd, X_es_sd, X_eu_sd, X_g_sd, X_sw_sd, X_s_sd])

leng = np.shape(X_esw)[0]
X = np.transpose(np.reshape(X,(-1,leng)))
X_sd = np.transpose(np.reshape(X_sd,(-1,leng)))

print(np.shape(X))

LABELS = ['eps-greedy-soft-g','eps-greedy-soft','eps-greedy-unif','greedy','softmax-g','softmax']

N_ = np.shape(X)[0]
N_mod = np.shape(X)[1]

iters = np.arange(N_)*50*N_leng

fontTitle = 32
fontTicks = 24
fontLabel = 30
colors = ['orange','green','red','violet','brown','pink']


figx = 16
figy = 12
fig = plt.figure(figsize=(figx,figy))

#[plt.plot(iters,X[:,i],color=colors[i],linewidth=6,label=LABELS[i]) for i in np.arange(N_mod)]
#[plt.fill_between(iters,np.maximum(X[:,i]-X_sd[:,i],0),X[:,i]+X_sd[:,i],color=colors[i],alpha=0.3) for i in np.arange(N_mod)]

plt.plot(iters,np.zeros(np.shape(iters)),linewidth=6,label='OPTIMAL',color='blue')
[plt.plot(iters,X[:,i],linewidth=6,label=LABELS[i],color=colors[i]) for i in np.arange(N_mod)]
[plt.fill_between(iters,np.maximum(X[:,i]-X_sd[:,i],0),X[:,i]+X_sd[:,i],alpha=0.3,color=colors[i]) for i in np.arange(N_mod)]
plt.plot(iters,X[:,0],linewidth=6,color='orange')
plt.fill_between(iters,np.maximum(X[:,0]-X_sd[:,0],0),X[:,0]+X_sd[:,0],alpha=0.3,color=colors[0])

#plt.xlim([0,5000])
plt.legend(fontsize=fontLabel)
plt.ylabel('Mean Training Error',fontsize=fontLabel)
plt.xlabel('Number of trials', fontsize=fontLabel)
#plt.xticks([0,1000,2000,3000,4000,5000],fontsize=fontTicks)
#plt.yticks([0,0.2,0.4,0.6,0.8,1],fontsize=fontTicks)

plt.show()

str_save = 'policy_error.png'
fig.savefig(str_save)
