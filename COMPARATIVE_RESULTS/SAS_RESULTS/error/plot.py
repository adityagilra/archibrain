import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec
import pylab

N_sim = 10
N_leng = 500

str_AUG =  'AuGMEnT_long_saccade_error_go.txt'
X_AUG = np.loadtxt(str_AUG)
X_AUG_sd = np.std(np.reshape(X_AUG,(-1,N_leng)),axis=0)
X_AUG = np.mean(np.reshape(X_AUG,(-1,N_leng)),axis=0)
str_HER = 'HER_long_saccadeerror_go.txt'
X_HER = np.loadtxt(str_HER)
X_HER_sd = np.std(np.reshape(X_HER,(-1,N_leng)),axis=0)
X_HER = np.mean(np.reshape(X_HER,(-1,N_leng)),axis=0)

str_LSTM = 'LSTM_long_saccade_error.txt'
X_LSTM = np.loadtxt(str_LSTM)
X_LSTM_sd = np.std(np.reshape(X_LSTM,(-1,N_leng)),axis=0)
X_LSTM = np.mean(np.reshape(X_LSTM,(-1,N_leng)),axis=0)

str_DNC = 'DNC_long_saccade_error.txt'
X_DNC = np.loadtxt(str_DNC)

X_DNC = np.mean(np.reshape(X_DNC,(-1,N_leng*50)),axis=0)
X_DNC_sd = np.std(np.reshape(X_DNC,(-1,50)),axis=1)
X_DNC = np.mean(np.reshape(X_DNC,(-1,50)),axis=1)


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
print(np.shape(X_sd))

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
tit = 'Training Error on SAS Task'
plt.title(tit,fontweight="bold",fontsize=fontTitle)
plt.xticks([0,1000,2000,3000,4000,5000],fontsize=fontTicks)
plt.yticks([0,0.2,0.4,0.6,0.8,1],fontsize=fontTicks)

plt.show()

str_save = 'SAS_error_plot.png'

fig.savefig(str_save)
