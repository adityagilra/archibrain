import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec
import pylab


str_AUG_AX =  'AuGMEnT_AX_conv_3.txt'
X_AUG_AX = np.loadtxt(str_AUG_AX)
str_AUG_12AXS =  'AuGMEnT_12AX-S_conv.txt'
X_AUG_12AXS = np.loadtxt(str_AUG_12AXS)
str_AUG_12AX =  'AuGMEnT_LONG_12AX_conv.txt'
X_AUG_12AX = np.loadtxt(str_AUG_12AX)

X = np.concatenate([X_AUG_AX,X_AUG_12AXS,X_AUG_12AX])
N_sim = np.shape(X_AUG_AX)[0]
X = np.transpose(np.reshape(X,(-1,N_sim)))

print(np.shape(X))

LABELS = ['AX','12AX-S','12AX']

N_sim = np.shape(X)[0]
N_mod = np.shape(X)[1]
ind = np.arange(N_mod)
conv_success = np.sum(np.where(X>0,1,0),axis=0)
frac_success = conv_success/N_sim

conv_mean = np.zeros((3))
conv_sd = np.zeros((3))
for i in np.arange(3):
	conv_s = conv_success[i]
	if conv_s!=0:
		conv_mean[i] = np.sum(np.where(X[:,i]>0,X[:,i],0))/conv_s
		conv_sd[i] = np.sqrt(np.sum((np.where(X[:,i]>0,X[:,i]-conv_mean[i],0))**2)/conv_s)

print(frac_success)
print(conv_mean)
print(conv_sd)

colors = ['#b5dab5','#7dbe7d','#46A346']

fontTitle = 30
fontTicks = 22
fontLabel = 26

#fig, (ax1, ax2) = plt.subplots(nrows=1,ncols=2,figsize=(25,12))
fig, (ax1, ax2) = plt.subplots(nrows=2,ncols=1,figsize=(12,30))

rects1= ax1.bar(ind, frac_success, color=colors,edgecolor='k')
ax1.set_ylabel('Fraction of Convergence Success',fontsize=fontLabel)
tit = 'AuGMEnT Percent of Success'
ax1.set_title(tit,fontweight="bold",fontsize=fontTitle)
ax1.set_xticks(ind)
ax1.set_xticklabels(LABELS,fontweight="bold",fontsize=fontLabel)
ax1.set_yticks([0,0.2,0.4,0.6,0.8,1])
ax1.set_yticklabels((0,0.2,0.4,0.6,0.8,1),fontsize=fontTicks)


rects2 = ax2.bar(ind, conv_mean,color=colors,yerr=conv_sd,edgecolor='k',error_kw=dict(elinewidth=5))
ax2.set_ylabel('Mean Convergence Time',fontsize=fontLabel)
tit = 'AuGMEnT Learning Time'
ax2.set_title(tit,fontweight="bold",fontsize=fontTitle)
ax2.set_xticks(ind)
ax2.set_xticklabels(LABELS,fontweight="bold",fontsize=fontLabel)
ax2.set_yticks([0,1500,3000,4500,6000])
ax2.set_yticklabels((0,1500,3000,4500,6000),fontsize=fontTicks)

plt.show()

str_save = '12AX_AuGMEnT_analysis_barplot.png'

fig.savefig(str_save)
