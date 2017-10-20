import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec
import pylab


str_AUG_l =  'leaky_conv_2.txt'
X_AUG_l = np.loadtxt(str_AUG_l)
str_AUG_12AX =  'AuGMEnT_12AX_conv.txt'
X_AUG_12AX = np.loadtxt(str_AUG_12AX)
str_HER = 'HER_12AX_conv.txt'
X_HER = np.loadtxt(str_HER)

print(np.shape(X_AUG_12AX))
print(np.shape(X_AUG_l))
print(np.shape(X_HER))

X = np.concatenate([X_AUG_12AX,X_AUG_l,X_HER])
N_sim = np.shape(X_AUG_l)[0]
X = np.transpose(np.reshape(X,(-1,N_sim)))

print(np.shape(X))

LABELS = ['AuGMEnT', 'Leaky-AuGMEnT', 'HER']

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

colors = ['#46A346','#46A375','#525CB9']

fontTitle = 30
fontTicks = 22
fontLabel = 26

#fig, (ax1, ax2) = plt.subplots(nrows=1,ncols=2,figsize=(25,10))
fig, (ax1, ax2) = plt.subplots(nrows=2,ncols=1,figsize=(14,20))

rects1= ax1.bar(ind, frac_success, color=colors,edgecolor='k')
ax1.set_ylabel('Fraction of Convergence Success',fontsize=fontLabel)
tit = '12AX: Percent of Success'
ax1.set_title(tit,fontweight="bold",fontsize=fontTitle)
ax1.set_xticks(ind)
ax1.set_xticklabels(LABELS,fontweight="bold",fontsize=fontLabel)
ax1.set_yticks([0,0.2,0.4,0.6,0.8,1])
ax1.set_yticklabels((0,0.2,0.4,0.6,0.8,1),fontsize=fontTicks)


rects2 = ax2.bar(ind, conv_mean,color=colors,yerr=conv_sd,edgecolor='k',error_kw=dict(elinewidth=5))
ax2.set_ylabel('Mean Convergence Time',fontsize=fontLabel)
tit = '12AX: Learning Time'
ax2.set_title(tit,fontweight="bold",fontsize=fontTitle)
ax2.set_xticks(ind)
ax2.set_xticklabels(LABELS,fontweight="bold",fontsize=fontLabel)
ax2.set_yticks([0,10000,20000,30000,40000])
ax2.set_yticklabels((0,10000,20000,30000,40000),fontsize=fontTicks)

plt.show()

str_save = 'leaky_AuGMEnT_analysis_barplot_2.png'
fig.savefig(str_save)
