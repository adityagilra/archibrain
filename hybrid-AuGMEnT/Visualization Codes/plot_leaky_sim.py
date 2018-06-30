import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec
import pylab

fontTitle = 32
fontTicks = 24
fontLabel = 28

folder = 'DATA'
task = 'seq_prediction'

AuG_type = 'leaky_AuG' 
str_conv = folder+'/'+AuG_type+'_'+task+'_ALL_CONV.txt'
l_AUG_conv = np.loadtxt(str_conv)
str_Q = folder+'/'+AuG_type+'_'+task+'_ALL_Q.txt'
l_AUG_Q = np.loadtxt(str_Q)
str_RPE = folder+'/'+AuG_type+'_'+task+'_ALL_RPE.txt'
l_AUG_RPE = np.loadtxt(str_RPE)
# str_resp = folder+'/'+AuG_type+'_'+task+'_ALL_RESP.txt'
# l_AUG_resp = np.loadtxt(str_resp)

d=5
END=5000
DELTA=50

l_AUG_Q = np.reshape(l_AUG_Q, (10,-1,2))
print(l_AUG_Q[0,-20:,:])
l_AUG_Q = l_AUG_Q[:,np.arange(1,np.shape(l_AUG_Q)[1]/(d+1)+1,dtype='int32')*(d+1)-1,:]
print(l_AUG_Q[0,-20:,:])
print(l_AUG_Q[1,-20:,:])
print(l_AUG_Q[2,-20:,:])
l_AUG_Q = l_AUG_Q[:,:END,:]
#print(np.shape(l_AUG_Q))
l_AUG_Q = np.max(l_AUG_Q,axis=2)
#print(np.shape(l_AUG_Q))
#print(l_AUG_Q[0,-20:])
l_AUG_Q = np.mean(np.reshape(l_AUG_Q, (10,-1,DELTA)), axis=2)

[plt.plot(l_AUG_Q[i,:]) for i in np.arange(10)]
plt.show()
