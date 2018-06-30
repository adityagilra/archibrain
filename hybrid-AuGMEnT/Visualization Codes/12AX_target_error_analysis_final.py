import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec
import pylab

import os

os.chdir('..')

fontTitle = 36
fontTicks = 24
fontLabel = 28

fontTicks = 24

folder = 'DATA'
task = '12AX'

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

AUG_RPE = AUG_RPE[:,:end]
l_AUG_RPE = l_AUG_RPE[:,:end]
h_AUG_RPE = h_AUG_RPE[:,:end]

non_target_vec = []
target_vec = []
S_data = np.loadtxt('DATA/12AX_Sdata_1000000.txt')
for i in np.arange(end):
	s_pos = np.argmax(S_data[i,:]) 
	if s_pos!=5 and s_pos!=6:
		non_target_vec.append(i)
	else:
		target_vec.append(i)

non_target_vec = np.array(non_target_vec)
target_vec = np.array(target_vec)

it = np.arange(end/delta)*delta

AUG_RPE_target = np.copy(AUG_RPE)
l_AUG_RPE_target = np.copy(l_AUG_RPE)
h_AUG_RPE_target = np.copy(h_AUG_RPE)
AUG_RPE_nontarget = np.copy(AUG_RPE)
l_AUG_RPE_nontarget = np.copy(l_AUG_RPE)
h_AUG_RPE_nontarget = np.copy(h_AUG_RPE)

AUG_RPE = 0.5*AUG_RPE**2
AUG_RPE_mean = np.mean(AUG_RPE, axis=0)
AUG_RPE_mean = np.mean(np.reshape(AUG_RPE_mean,(-1,delta)),axis=1)
AUG_RPE_sd = np.std(AUG_RPE, axis=0)
AUG_RPE_sd = np.mean(np.reshape(AUG_RPE_sd,(-1,delta)),axis=1) 

l_AUG_RPE = 0.5*l_AUG_RPE**2
l_AUG_RPE_mean = np.mean(l_AUG_RPE, axis=0)
l_AUG_RPE_mean = np.mean(np.reshape(l_AUG_RPE_mean,(-1,delta)),axis=1) 
l_AUG_RPE_sd = np.std(l_AUG_RPE, axis=0)
l_AUG_RPE_sd = np.mean(np.reshape(l_AUG_RPE_sd,(-1,delta)),axis=1) 

h_AUG_RPE = 0.5*h_AUG_RPE**2
h_AUG_RPE_mean = np.mean(h_AUG_RPE, axis=0)
h_AUG_RPE_mean = np.mean(np.reshape(h_AUG_RPE_mean,(-1,delta)),axis=1)
h_AUG_RPE_sd = np.std(h_AUG_RPE, axis=0)
h_AUG_RPE_sd = np.mean(np.reshape(h_AUG_RPE_sd,(-1,delta)),axis=1) 

AUG_RPE_target[:,non_target_vec]=0
l_AUG_RPE_target[:,non_target_vec]=0
h_AUG_RPE_target[:,non_target_vec]=0

AUG_RPE_nontarget[:,target_vec]=0
l_AUG_RPE_nontarget[:,target_vec]=0
h_AUG_RPE_nontarget[:,target_vec]=0



FIG=plt.figure(figsize=(30,12))

plt.subplot(1,3,1)
plt.plot(it,AUG_RPE_mean,marker='o',color='green',label='AuGMEnT')
plt.fill_between(it,np.maximum(AUG_RPE_mean-AUG_RPE_sd,0),AUG_RPE_mean+AUG_RPE_sd,color='green',alpha=0.1)
plt.plot(it,l_AUG_RPE_mean,marker='o',color='b',label='Leaky AuGMEnT')
plt.fill_between(it,np.maximum(l_AUG_RPE_mean-l_AUG_RPE_sd,0),l_AUG_RPE_mean+l_AUG_RPE_sd,color='b',alpha=0.1)
plt.plot(it,h_AUG_RPE_mean,marker='o',color='r',label='Hybrid AuGMEnT')
plt.fill_between(it,np.maximum(h_AUG_RPE_mean-h_AUG_RPE_sd,0),h_AUG_RPE_mean+h_AUG_RPE_sd,color='r',alpha=0.1)
plt.xlabel('Training Trials',fontsize=fontLabel)
plt.ylabel('Mean Loss Function',fontsize=fontLabel)
plt.yticks(fontsize=24)
plt.legend(fontsize=24)
plt.ylim([0,0.2])
plt.xticks([0,50000,100000,150000],fontsize=24)
plt.text(-65000,0.21,'A)',fontsize=fontTitle,fontweight='bold')

###
AUG_RPE_target = 0.5*AUG_RPE_target**2
AUG_RPE_target_mean = np.mean(AUG_RPE_target, axis=0)
AUG_RPE_target_mean = np.reshape(AUG_RPE_target_mean,(-1,delta))
AUG_RPE_target_mean = np.sum(AUG_RPE_target_mean,axis=1)/np.count_nonzero(AUG_RPE_target_mean,axis=1)
AUG_RPE_target_sd = np.std(AUG_RPE_target, axis=0)
AUG_RPE_target_sd = np.reshape(AUG_RPE_target_sd,(-1,delta))
AUG_RPE_target_sd = np.sum(AUG_RPE_target_sd,axis=1)/np.count_nonzero(AUG_RPE_target_sd,axis=1)

l_AUG_RPE_target = 0.5*l_AUG_RPE_target**2
l_AUG_RPE_target_mean = np.mean(l_AUG_RPE_target, axis=0)
l_AUG_RPE_target_mean = np.reshape(l_AUG_RPE_target_mean,(-1,delta))
l_AUG_RPE_target_mean = np.sum(l_AUG_RPE_target_mean,axis=1)/np.count_nonzero(l_AUG_RPE_target_mean,axis=1)
l_AUG_RPE_target_sd = np.std(l_AUG_RPE_target, axis=0)
l_AUG_RPE_target_sd = np.reshape(l_AUG_RPE_target_sd,(-1,delta))
l_AUG_RPE_target_sd = np.sum(l_AUG_RPE_target_sd,axis=1)/np.count_nonzero(l_AUG_RPE_target_sd,axis=1)

h_AUG_RPE_target = 0.5*h_AUG_RPE_target**2
h_AUG_RPE_target_mean = np.mean(h_AUG_RPE_target, axis=0)
h_AUG_RPE_target_mean = np.reshape(h_AUG_RPE_target_mean,(-1,delta))
h_AUG_RPE_target_mean = np.sum(h_AUG_RPE_target_mean,axis=1)/np.count_nonzero(h_AUG_RPE_target_mean,axis=1)
h_AUG_RPE_target_sd = np.std(h_AUG_RPE_target, axis=0)
h_AUG_RPE_target_sd = np.reshape(h_AUG_RPE_target_sd,(-1,delta))
h_AUG_RPE_target_sd = np.sum(h_AUG_RPE_target_sd,axis=1)/np.count_nonzero(h_AUG_RPE_target_sd,axis=1)


plt.subplot(1,3,2)
plt.plot(it,AUG_RPE_target_mean,color='g',marker='o',label='AuGMEnT')
plt.fill_between(it,np.maximum(AUG_RPE_target_mean-AUG_RPE_target_sd,0),AUG_RPE_target_mean+AUG_RPE_target_sd,color='g',alpha=0.1)
plt.plot(it,l_AUG_RPE_target_mean,marker='o',color='b',label='Leaky AuGMEnT')
plt.fill_between(it,np.maximum(l_AUG_RPE_target_mean-l_AUG_RPE_target_sd,0),l_AUG_RPE_target_mean+l_AUG_RPE_target_sd,color='b',alpha=0.1)
plt.plot(it,h_AUG_RPE_target_mean,marker='o',color='r',label='Hybrid AuGMEnT')
plt.fill_between(it,np.maximum(h_AUG_RPE_target_mean-h_AUG_RPE_target_sd,0),h_AUG_RPE_target_mean+h_AUG_RPE_target_sd,color='r',alpha=0.1)
plt.xlabel('Training Trials',fontsize=fontLabel)
plt.ylabel('Mean Loss Function - Target Cues',fontsize=fontLabel)
plt.yticks(fontsize=24)
plt.legend(fontsize=24)
plt.xticks([0,50000,100000,150000],fontsize=24)
plt.ylim([0,0.2])
plt.text(-65000,0.21,'B)',fontsize=fontTitle,fontweight='bold')

###
print(np.shape(AUG_RPE_nontarget))
AUG_RPE_nontarget = 0.5*AUG_RPE_nontarget**2
AUG_RPE_nontarget_mean = np.mean(AUG_RPE_nontarget, axis=0)
print(np.shape(AUG_RPE_nontarget))
AUG_RPE_nontarget_mean = np.reshape(AUG_RPE_nontarget_mean,(-1,delta))
print(np.shape(AUG_RPE_nontarget))
AUG_RPE_nontarget_mean = np.sum(AUG_RPE_nontarget_mean,axis=1)/np.count_nonzero(AUG_RPE_nontarget_mean,axis=1)
print(np.shape(AUG_RPE_nontarget))
AUG_RPE_nontarget_sd = np.std(AUG_RPE_nontarget, axis=0)
AUG_RPE_nontarget_sd = np.reshape(AUG_RPE_nontarget_sd,(-1,delta))
AUG_RPE_nontarget_sd = np.sum(AUG_RPE_nontarget_sd,axis=1)/np.count_nonzero(AUG_RPE_nontarget_sd,axis=1)

l_AUG_RPE_nontarget = 0.5*l_AUG_RPE_nontarget**2
l_AUG_RPE_nontarget_mean = np.mean(l_AUG_RPE_nontarget, axis=0)
l_AUG_RPE_nontarget_mean = np.reshape(l_AUG_RPE_nontarget_mean,(-1,delta))
l_AUG_RPE_nontarget_mean = np.sum(l_AUG_RPE_nontarget_mean,axis=1)/np.count_nonzero(l_AUG_RPE_nontarget_mean,axis=1)
l_AUG_RPE_nontarget_sd = np.std(l_AUG_RPE_nontarget, axis=0)
l_AUG_RPE_nontarget_sd = np.reshape(l_AUG_RPE_nontarget_sd,(-1,delta))
l_AUG_RPE_nontarget_sd = np.sum(l_AUG_RPE_nontarget_sd,axis=1)/np.count_nonzero(l_AUG_RPE_nontarget_sd,axis=1)

h_AUG_RPE_nontarget = 0.5*h_AUG_RPE_nontarget**2
h_AUG_RPE_nontarget_mean = np.mean(h_AUG_RPE_nontarget, axis=0)
h_AUG_RPE_nontarget_mean = np.reshape(h_AUG_RPE_nontarget_mean,(-1,delta))
h_AUG_RPE_nontarget_mean = np.sum(h_AUG_RPE_nontarget_mean,axis=1)/np.count_nonzero(h_AUG_RPE_nontarget_mean,axis=1)
h_AUG_RPE_nontarget_sd = np.std(h_AUG_RPE_nontarget, axis=0)
h_AUG_RPE_nontarget_sd = np.reshape(h_AUG_RPE_nontarget_sd,(-1,delta))
h_AUG_RPE_nontarget_sd = np.sum(h_AUG_RPE_nontarget_sd,axis=1)/np.count_nonzero(h_AUG_RPE_nontarget_sd,axis=1)

plt.subplot(1,3,3)
plt.plot(it,AUG_RPE_nontarget_mean,color='g',marker='o',label='AuGMEnT')
plt.fill_between(it,np.maximum(AUG_RPE_nontarget_mean-AUG_RPE_nontarget_sd,0),AUG_RPE_nontarget_mean+AUG_RPE_nontarget_sd,color='g',alpha=0.1)
plt.plot(it,l_AUG_RPE_nontarget_mean,marker='o',color='b',label='Leaky AuGMEnT')
plt.fill_between(it,np.maximum(l_AUG_RPE_nontarget_mean-l_AUG_RPE_nontarget_sd,0),l_AUG_RPE_nontarget_mean+l_AUG_RPE_nontarget_sd,color='b',alpha=0.1)
plt.plot(it,h_AUG_RPE_nontarget_mean,marker='o',color='r',label='Hybrid AuGMEnT')
plt.fill_between(it,np.maximum(h_AUG_RPE_nontarget_mean-h_AUG_RPE_nontarget_sd,0),h_AUG_RPE_nontarget_mean+h_AUG_RPE_nontarget_sd,color='r',alpha=0.1)
plt.xlabel('Training Trials',fontsize=fontLabel)
plt.ylabel('Mean Loss Function - Non-Target Cues',fontsize=fontLabel)
plt.yticks(fontsize=24)
plt.xticks([0,50000,100000,150000],fontsize=24)
plt.legend(fontsize=24)
plt.ylim([0,0.2])
plt.text(-65000,0.21,'C)',fontsize=fontTitle,fontweight='bold')


plt.subplots_adjust(left=0.1,wspace=0.45)
#plt.tight_layout()

strsave='IMAGES/12AX_complete_error_analysis_final.png'
FIG.savefig(strsave)

plt.show()
