import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec
import pylab

N_sim = 10
N_leng = 200

str_AUG =  'AuGMEnT_long_12AX_error.txt'
X_AUG = np.loadtxt(str_AUG)
X_AUG = np.reshape(X_AUG,(-1,N_leng))
X_AUG_sd = np.std(X_AUG,axis=0)
X_AUG = np.mean(X_AUG,axis=0)

str_HER = 'HER_long_12AXerror.txt'
X_HER = np.loadtxt(str_HER)
X_HER = np.reshape(X_HER,(-1,N_leng))
X_HER_sd = np.std(X_HER,axis=0)
X_HER = np.mean(X_HER,axis=0)

str_LSTM = 'LSTM_long_12AX_error_2.txt'
X_LSTM = np.loadtxt(str_LSTM)
X_LSTM = np.reshape(X_LSTM,(-1,N_leng))
X_LSTM_sd = np.std(X_LSTM,axis=0)
X_LSTM = np.mean(X_LSTM,axis=0)

str_DNC = 'DNC_long_12AX_error.txt'
X_DNC = np.loadtxt(str_DNC)
X_DNC = np.reshape(X_DNC,(-1,N_leng))
X_DNC_sd = np.std(X_DNC,axis=0)
X_DNC = np.mean(X_DNC,axis=0)

X = np.concatenate([X_AUG,X_HER,X_LSTM,X_DNC])
X_sd = np.concatenate([X_AUG_sd,X_HER_sd,X_LSTM_sd,X_DNC_sd])
print(np.shape(X_AUG))
print(np.shape(X_HER))
print(np.shape(X_LSTM))
print(np.shape(X_DNC))

leng = np.shape(X_AUG)[0]
X = np.transpose(np.reshape(X,(-1,leng)))
X_sd = np.transpose(np.reshape(X_sd,(-1,leng)))

print(np.shape(X))

LABELS = ['AuGMEnT','HER','LSTM','DNC']

N_ = np.shape(X)[0]
N_mod = np.shape(X)[1]

iters = np.arange(N_)*50

fontTitle = 32
fontTicks = 24
fontLabel = 30
colors = ['#46A346','#525CB9','#A155E7','#B92828']


fig = plt.figure(figsize=(18,13))

[plt.plot(iters,X[:,i],color=colors[i],linewidth=6,label=LABELS[i]) for i in np.arange(N_mod)]
[plt.fill_between(iters,np.maximum(X[:,i]-X_sd[:,i],0),X[:,i]+X_sd[:,i],color=colors[i],alpha=0.3) for i in np.arange(N_mod)]

plt.xlim([0,5000])
plt.legend(fontsize=fontLabel)
plt.ylabel('Mean Training Error',fontsize=fontLabel)
plt.xlabel('Number of trials', fontsize=fontLabel)
tit = 'Training Error on 12AX Task'
plt.title(tit,fontweight="bold",fontsize=fontTitle)
plt.xticks([0,1000,2000,3000,4000,5000],fontsize=fontTicks)
plt.yticks([0,0.2,0.4,0.6,0.8,1],fontsize=fontTicks)

plt.show()

str_save = '12AX_error_plot_longer.png'

fig.savefig(str_save)
