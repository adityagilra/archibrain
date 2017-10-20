import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec
import pylab


str_AUG_g =  'AuGMEnT_LONG_12AX_perc.txt'
X_AUG_g = np.loadtxt(str_AUG_g)
str_HER_g = 'HER_long_12AX_perc.txt'
X_HER_g = np.loadtxt(str_HER_g)
str_LSTM = 'LSTM_long_12AX_perc.txt'
X_LSTM = np.loadtxt(str_LSTM)
str_DNC = 'DNC_long_12AX_perc.txt'
X_DNC = np.loadtxt(str_DNC)

X = np.concatenate([X_AUG_g,X_HER_g,X_LSTM,X_DNC])
N_sim = np.shape(X_AUG_g)[0]
X = np.transpose(np.reshape(X,(-1,N_sim)))

print(X)

X_mean = np.mean(X,axis=0)
print('MEAN: \n',X_mean)

LABELS = ['AuGMEnT','HER','LSTM','DNC']
