import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec
import pylab

N_sim = 10
N_leng = 200
limit = 10000

str_AUG_AX =  'AuGMEnTAX_error_3.txt'
X_AUG_AX = np.loadtxt(str_AUG_AX)
X_AUG_AX = np.reshape(X_AUG_AX,(-1,N_leng))
X_AUG_AX_sd = np.std(X_AUG_AX,axis=0)
X_AUG_AX = np.mean(X_AUG_AX,axis=0)

str_AUG_12AXS =  'AuGMEnT12AX-S_error.txt'
X_AUG_12AXS = np.loadtxt(str_AUG_12AXS)
X_AUG_12AXS = np.reshape(X_AUG_12AXS,(-1,N_leng*10))
X_AUG_12AXS_sd = np.std(X_AUG_12AXS,axis=0)
X_AUG_12AXS = np.mean(X_AUG_12AXS,axis=0)
X_AUG_12AXS = X_AUG_12AXS[:N_leng]
X_AUG_12AXS_sd = X_AUG_12AXS_sd[:N_leng]

str_AUG_12AX =  'AuGMEnT_LONG_12AX_error.txt'
X_AUG_12AX = np.loadtxt(str_AUG_12AX)
print(np.shape(X_AUG_12AX))
X_AUG_12AX = np.reshape(X_AUG_12AX,(-1,N_leng*100))
print(np.shape(X_AUG_12AX))
X_AUG_12AX_sd = np.std(X_AUG_12AX,axis=0)
X_AUG_12AX = np.mean(X_AUG_12AX,axis=0)
print(np.shape(X_AUG_12AX))
X_AUG_12AX = X_AUG_12AX[:N_leng]
X_AUG_12AX_sd = X_AUG_12AX_sd[:N_leng]
print(np.shape(X_AUG_12AX),'\n')


X = np.concatenate([X_AUG_AX,X_AUG_12AXS,X_AUG_12AX])
X_sd = np.concatenate([X_AUG_AX_sd,X_AUG_12AXS_sd,X_AUG_12AX_sd])

leng = np.shape(X_AUG_AX)[0]
X = np.transpose(np.reshape(X,(-1,leng)))
X_sd = np.transpose(np.reshape(X_sd,(-1,leng)))

print(np.shape(X))

LABELS = ['AX','12AX-S','12AX']

N_ = np.shape(X)[0]
N_mod = np.shape(X)[1]

iters = np.arange(N_)*50

fontTitle = 32
fontTicks = 26
fontLabel = 30

colors = ['#b5dab5','#7dbe7d','#46A346']


fig = plt.figure(figsize=(18,13))


[plt.fill_between(iters,np.maximum(X[:,i]-X_sd[:,i],0),X[:,i]+X_sd[:,i],color=colors[i],alpha=0.5) for i in np.arange(N_mod)]

[plt.plot(iters,X[:,i],color=colors[i],linewidth=6,label=LABELS[i]) for i in np.arange(N_mod)]

plt.xlim([0,5000])
plt.legend(fontsize=fontLabel)
plt.ylabel('Mean Training Error',fontsize=fontLabel)
plt.xlabel('Number of trials', fontsize=fontLabel)
tit = 'AuGMEnT Analysis: Training Error'
plt.title(tit,fontweight="bold",fontsize=fontTitle)
plt.xticks([0,2000,4000,6000,8000,10000],fontsize=fontTicks)
plt.yticks([0,0.2,0.4,0.6,0.8,1],fontsize=fontTicks)

plt.show()

str_save = 'AuGMEnT_analysis_error_plot.png'

fig.savefig(str_save)
