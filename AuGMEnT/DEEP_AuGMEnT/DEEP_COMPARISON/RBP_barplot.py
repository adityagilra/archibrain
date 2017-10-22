import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec
import pylab

N_sim= 25

str_AUG_l =  'leaky_conv_2.txt'
X_AUG_l = np.loadtxt(str_AUG_l)
X_AUG_l = X_AUG_l[:N_sim]
str_AUG_DMC =  'DCM_conv_3.txt'
X_AUG_DMC = np.loadtxt(str_AUG_DMC)
X_AUG_DMC = X_AUG_DMC[:N_sim]
str_AUG_DMC_RBP =  'RBP_conv_3.txt'
X_AUG_DMC_RBP = np.loadtxt(str_AUG_DMC_RBP)
str_AUG_DMC_SRBP =  'SRBP_conv_3.txt'
X_AUG_DMC_SRBP = np.loadtxt(str_AUG_DMC_SRBP)
str_AUG_DMC_MRBP =  'MRBP_conv_3.txt'
X_AUG_DMC_MRBP = np.loadtxt(str_AUG_DMC_MRBP)
#str_HER = 'HER_12AX_conv.txt'
#X_HER = np.loadtxt(str_HER)

X = np.concatenate([X_AUG_l,X_AUG_DMC,X_AUG_DMC_RBP,X_AUG_DMC_SRBP,X_AUG_DMC_MRBP])
X = np.transpose(np.reshape(X,(-1,N_sim)))

print(np.shape(X))

LABELS = ['Leaky', 'Deep', 'Deep\n(RBP)', 'Deep\n(SRBP)', 'Deep\n(MRBP)']

N_sim = np.shape(X)[0]
N_mod = np.shape(X)[1]
ind = np.arange(N_mod)
conv_success = np.sum(np.where(X>0,1.,0),axis=0)
frac_success = conv_success/N_sim

conv_mean = np.zeros((5))
conv_sd = np.zeros((5))
for i in np.arange(5):
	conv_s = conv_success[i]
	if conv_s!=0:
		conv_mean[i] = np.sum(np.where(X[:,i]>0,X[:,i],0))/conv_s
		conv_sd[i] = np.sqrt(np.sum((np.where(X[:,i]>0,X[:,i]-conv_mean[i],0))**2)/conv_s)

print(frac_success)
print(conv_mean)
print(conv_sd)

colors = ['#46A375','#214d21','#255725','#225725','#225725']

fontTitle = 30
fontTicks = 20
fontLabel = 23

fig, (ax1, ax2) = plt.subplots(nrows=1,ncols=2,figsize=(25,10))
#fig, (ax1, ax2) = plt.subplots(nrows=2,ncols=1,figsize=(14,25))

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
ax2.set_yticks([0,25000,50000,75000,100000,125000,150000])
ax2.set_yticklabels((0,25000,50000,75000,100000,125000,150000),fontsize=fontTicks)

plt.show()

str_save = 'deep_AUGMENT_RBP.png'
fig.savefig(str_save)
