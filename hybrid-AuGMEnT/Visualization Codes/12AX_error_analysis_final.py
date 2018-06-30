import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec
import pylab


str_AUG =  'AuGMEnT_12AX_conv.txt'
X_AUG = np.loadtxt(str_AUG)
#X_AUG = np.zeros((100))
str_l = 'leaky_AuG_12AX_conv.txt'
X_l = np.loadtxt(str_l)
str_h = 'hybrid_AuG_12AX_conv.txt'
X_h = np.loadtxt(str_h)

X = np.concatenate([X_AUG,X_l,X_h])
N_sim = np.shape(X_AUG)[0]
X = np.transpose(np.reshape(X,(-1,N_sim)))

print(np.shape(X))

LABELS = ['AuGMEnT','Leaky\nAuGMEnT','Hybrid\nAuGMEnT']

N_sim = np.shape(X)[0]
N_mod = np.shape(X)[1]
ind = np.arange(N_mod)
conv_success = np.sum(np.where(X!=0,1,0),axis=0)
frac_success = conv_success/N_sim

conv_mean = np.zeros((N_mod))
conv_sd = np.zeros((N_mod))
for i in np.arange(N_mod):
	conv_s = conv_success[i]
	if conv_s!=0:
		conv_mean[i] = np.mean(X[np.where(X[:,i]!=0),i])
		conv_sd[i] = np.std(X[np.where(X[:,i]!=0),i])

print(frac_success)
print(conv_mean)
print(conv_sd)

fontTitle = 32
fontTicks = 24
fontLabel = 28

colors = ['green','#525CB9','#B92828']

fig, (ax1, ax2) = plt.subplots(nrows=1,ncols=2,figsize=(30,12))
#fig, (ax1, ax2) = plt.subplots(nrows=2,ncols=1,figsize=(15,30))

rects1= ax1.bar(ind, frac_success, color=colors,edgecolor='k')
ax1.set_ylabel('Fraction of Convergence Success',fontsize=fontLabel)
tit = 'Percent of Success'
ax1.set_title(tit,fontweight="bold",fontsize=fontTitle)
ax1.set_xticks(ind)
ax1.set_xticklabels(LABELS,fontweight="bold",fontsize=fontLabel)
ax1.set_yticks([0,0.2,0.4,0.6,0.8,1])
ax1.set_yticklabels((0,0.2,0.4,0.6,0.8,1),fontsize=fontTicks)


rects2 = ax2.bar(ind, conv_mean,color=colors,yerr=conv_sd,edgecolor='k',error_kw=dict(elinewidth=5))
ax2.set_ylabel('Mean Convergence Time',fontsize=fontLabel)
tit = 'Learning Time'
ax2.set_title(tit,fontweight="bold",fontsize=fontTitle)
ax2.set_xticks(ind)
ax2.set_xticklabels(LABELS,fontweight="bold",fontsize=fontLabel)
ax2.set_yticks([0,10000,20000,30000,40000, 50000])
ax2.set_yticklabels((0,10000,20000,30000,40000,50000),fontsize=fontTicks)

plt.subplots_adjust(hspace=0.35)

str_save = '12AX_barplot_hor_augment.png'

fig.savefig(str_save)

plt.show(block=False)

#########################################

fontTicks = 24

folder = 'DATA'
task = '12AX'

FIG_1=plt.figure(figsize=(15,10))

AuG_type = 'AuGMEnT' 
str_RPE = folder+'/'+AuG_type+'_'+task+'_ALL_RPE_cut.txt'
AUG_RPE = np.loadtxt(str_RPE)

AuG_type = 'leaky_AuG'
str_RPE = folder+'/'+AuG_type+'_'+task+'_ALL_RPE.txt'
l_AUG_RPE = np.loadtxt(str_RPE)

AuG_type = 'hybrid_AuG' 
str_RPE = folder+'/'+AuG_type+'_'+task+'_ALL_RPE.txt'
h_AUG_RPE = np.loadtxt(str_RPE)

delta = 2000
end = 150000

it = np.arange(end/delta)*delta

AUG_RPE = 0.5*AUG_RPE**2
AUG_RPE_mean = np.mean(AUG_RPE, axis=0)
AUG_RPE_mean = AUG_RPE_mean[:end]
AUG_RPE_mean = np.mean(np.reshape(AUG_RPE_mean,(-1,delta)),axis=1)
AUG_RPE_sd = np.std(AUG_RPE, axis=0)
AUG_RPE_sd = AUG_RPE_sd[:end]
AUG_RPE_sd = np.mean(np.reshape(AUG_RPE_sd,(-1,delta)),axis=1) 

l_AUG_RPE = 0.5*l_AUG_RPE**2
print(np.shape(l_AUG_RPE))
l_AUG_RPE_mean = np.mean(l_AUG_RPE, axis=0)
print(np.shape(l_AUG_RPE_mean))
l_AUG_RPE_mean = l_AUG_RPE_mean[:end]
print(np.shape(l_AUG_RPE_mean))
l_AUG_RPE_mean = np.mean(np.reshape(l_AUG_RPE_mean,(-1,delta)),axis=1) 
print(np.shape(l_AUG_RPE_mean))
l_AUG_RPE_sd = np.std(l_AUG_RPE, axis=0)
l_AUG_RPE_sd = l_AUG_RPE_sd[:end]
l_AUG_RPE_sd = np.mean(np.reshape(l_AUG_RPE_sd,(-1,delta)),axis=1) 

h_AUG_RPE = 0.5*h_AUG_RPE**2
h_AUG_RPE_mean = np.mean(h_AUG_RPE, axis=0)
h_AUG_RPE_mean = h_AUG_RPE_mean[:end]
h_AUG_RPE_mean = np.mean(np.reshape(h_AUG_RPE_mean,(-1,delta)),axis=1)
h_AUG_RPE_sd = np.std(h_AUG_RPE, axis=0)
h_AUG_RPE_sd = h_AUG_RPE_sd[:end]
h_AUG_RPE_sd = np.mean(np.reshape(h_AUG_RPE_sd,(-1,delta)),axis=1) 


plt.plot(it,AUG_RPE_mean,marker='o',label='AuGMEnT')
plt.fill_between(it,np.maximum(AUG_RPE_mean-AUG_RPE_sd,0),AUG_RPE_mean+AUG_RPE_sd,alpha=0.1)
plt.plot(it,l_AUG_RPE_mean,marker='o',color='r',label='Leaky AuGMEnT')
plt.fill_between(it,np.maximum(l_AUG_RPE_mean-l_AUG_RPE_sd,0),l_AUG_RPE_mean+l_AUG_RPE_sd,color='r',alpha=0.1)
plt.plot(it,h_AUG_RPE_mean,marker='o',color='purple',label='Hybrid AuGMEnT')
plt.fill_between(it,np.maximum(h_AUG_RPE_mean-h_AUG_RPE_sd,0),h_AUG_RPE_mean+h_AUG_RPE_sd,color='purple',alpha=0.1)
#tit = '12AX Task'
#plt.title(tit,fontweight="bold",fontsize=fontTitle)
plt.xlabel('Training Trials',fontsize=fontLabel)
plt.ylabel('Mean Energy Functional',fontsize=fontLabel)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.legend(fontsize=24)
plt.ylim([0,0.2])

strsave='12AX_error_analysis_final.png'
FIG_1.savefig(strsave)

plt.show(block=True)

#########################################

FIG_2=plt.figure(figsize=(15,10))

AuG_type = 'AuGMEnT' 
str_Q = folder+'/'+AuG_type+'_'+task+'_ALL_Q_cut.txt'
AUG_Q = np.loadtxt(str_Q)

AuG_type = 'leaky_AuG'
str_Q = folder+'/'+AuG_type+'_'+task+'_ALL_Q.txt'
l_AUG_Q = np.loadtxt(str_Q)

AuG_type = 'hybrid_AuG' 
str_Q = folder+'/'+AuG_type+'_'+task+'_ALL_Q.txt'
h_AUG_Q = np.loadtxt(str_Q)

END = 150000
DELTA = 2500

IT = np.arange(END/DELTA)*DELTA

AUG_Q = np.reshape(AUG_Q, (10,-1,2))
AUG_Q = AUG_Q[:,:END,:]

AUG_Q_max = np.max(AUG_Q,axis=2)
AUG_Q_MSE = np.zeros((END))
for i in np.arange(10):
	for j in np.arange(i+1,10):
		AUG_Q_MSE += (AUG_Q_max[i,:]-AUG_Q_max[j,:])**2
AUG_Q_MSE /= 45
AUG_Q_MSE = np.mean(np.reshape(AUG_Q_MSE, (-1,DELTA)), axis=1)


l_AUG_Q = np.reshape(l_AUG_Q, (10,-1,2))
l_AUG_Q = l_AUG_Q[:,:END,:]

l_AUG_Q_max = np.max(l_AUG_Q,axis=2)
l_AUG_Q_MSE = np.zeros((END))
for i in np.arange(10):
	for j in np.arange(i+1,10):
		l_AUG_Q_MSE += (l_AUG_Q_max[i,:]-l_AUG_Q_max[j,:])**2
l_AUG_Q_MSE /= 45
l_AUG_Q_MSE = np.mean(np.reshape(l_AUG_Q_MSE, (-1,DELTA)), axis=1)

h_AUG_Q = np.reshape(h_AUG_Q, (10,-1,2))
h_AUG_Q = h_AUG_Q[:,:END,:]

h_AUG_Q_max = np.max(h_AUG_Q,axis=2)
h_AUG_Q_MSE = np.zeros((END))
for i in np.arange(10):
	for j in np.arange(i+1,10):
		h_AUG_Q_MSE += (h_AUG_Q_max[i,:]-h_AUG_Q_max[j,:])**2
h_AUG_Q_MSE /= 45
h_AUG_Q_MSE = np.mean(np.reshape(h_AUG_Q_MSE, (-1,DELTA)), axis=1)

plt.plot(IT,AUG_Q_MSE,marker='o',linewidth=2,label='AuGMEnT')
plt.plot(IT,l_AUG_Q_MSE,marker='o',color='r',linewidth=2,label='Leaky AuGMEnT')
plt.plot(IT,h_AUG_Q_MSE,marker='o',color='purple',linewidth=2,label='Hybrid AuGMEnT')
#tit = '12AX Task'
#plt.title(tit,fontweight="bold",fontsize=fontTitle)
plt.xlabel('Training Iterations',fontsize=fontLabel)
plt.ylabel('Max Q-value MSE',fontsize=fontLabel)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.legend(fontsize=24)

plt.show(block=False)

strsave='12AX_error_analysis_max_final.png'
FIG_2.savefig(strsave)

#########################################

FIG_3=plt.figure(figsize=(15,10))

AUG_Q_MSE = np.zeros((END,2))
for i in np.arange(10):
	for j in np.arange(i+1,10):
		AUG_Q_MSE += (AUG_Q[i,:,:]-AUG_Q[j,:,:])**2
AUG_Q_MSE /= 45
AUG_Q_MSE = np.mean(AUG_Q_MSE,axis=1)
AUG_Q_MSE = np.mean(np.reshape(AUG_Q_MSE, (-1,DELTA)), axis=1)


l_AUG_Q_MSE = np.zeros((END,2))
for i in np.arange(10):
	for j in np.arange(i+1,10):
		l_AUG_Q_MSE += (l_AUG_Q[i,:,:]-l_AUG_Q[j,:,:])**2
l_AUG_Q_MSE /= 45
l_AUG_Q_MSE = np.mean(l_AUG_Q_MSE,axis=1)
l_AUG_Q_MSE = np.mean(np.reshape(l_AUG_Q_MSE, (-1,DELTA)), axis=1)


h_AUG_Q_MSE = np.zeros((END,2))
for i in np.arange(10):
	for j in np.arange(i+1,10):
		h_AUG_Q_MSE += (h_AUG_Q[i,:,:]-h_AUG_Q[j,:,:])**2
h_AUG_Q_MSE /= 45
h_AUG_Q_MSE = np.mean(h_AUG_Q_MSE,axis=1)
h_AUG_Q_MSE = np.mean(np.reshape(h_AUG_Q_MSE, (-1,DELTA)), axis=1)

plt.plot(IT,AUG_Q_MSE,marker='o',linewidth=2,label='AuGMEnT')
plt.plot(IT,l_AUG_Q_MSE,marker='o',color='r',linewidth=2,label='Leaky AuGMEnT')
plt.plot(IT,h_AUG_Q_MSE,marker='o',color='purple',linewidth=2,label='Hybrid AuGMEnT')
#tit = '12AX Task'
#plt.title(tit,fontweight="bold",fontsize=fontTitle)
plt.xlabel('Training Iterations',fontsize=fontLabel)
plt.ylabel('Mean Q-value MSE',fontsize=fontLabel)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.legend(fontsize=24)

plt.show(block=False)

strsave='12AX_error_analysis_mean_final.png'
FIG_3.savefig(strsave)
