import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec
import pylab
import gc 
import os

os.chdir('..')

task = 'seq_prediction'

fontTitle = 32
fontTicks = 24
fontLabel = 28

M=4 
mem_vec_c=[]
for i in range(M):
	mem_vec_c.append('M'+str(i+1)+'-C')
mem_vec_l=[]
for i in range(M):
	mem_vec_l.append('M'+str(i+1)+'-L')
mem_vec_h=[]
for i in range(int(M/2)):
	mem_vec_h.append('M'+str(i+1)+'-L')
for i in range(int(M/2)):
	mem_vec_h.append('M'+str(i+1)+'-C')

from task_seq_prediction import get_dictionary

dic_stim3,_ = get_dictionary(3)
dic_stim8,_ = get_dictionary(8)


cues_vec_3 = []
values_vec = list(dic_stim3.values())
for l in values_vec:
	cues_vec_3.append(l+'+')
for l in values_vec:
	cues_vec_3.append(l+'-')
cues_vec_8 = []
values_vec = list(dic_stim8.values())
for l in values_vec:
	cues_vec_8.append(l+'+')
for l in values_vec:
	cues_vec_8.append(l+'-')


FIG = plt.figure(figsize=(30,25))

AuG_type = 'AuGMEnT'
tit_aug = 'AuGMEnT'

d = 3
S = d+2

savestr = 'DATA/'+AuG_type+'_'+task+'_'+'distr'+str(d)+'_weights_Vm.txt'
X_a3 = np.loadtxt(savestr)

plt.subplot(2,3,1)
plt.pcolor(np.flipud(X_a3),edgecolors='k', linewidths=1)
plt.set_cmap('bwr')		
cb = plt.colorbar()
cb.ax.tick_params(labelsize=fontLabel)
tit = tit_aug+' (L='+str(d+2)+')'
plt.title(tit,fontweight="bold",fontsize=fontTitle)
plt.xticks(np.linspace(0.5,M-0.5,M,endpoint=True),mem_vec_c,fontsize=fontTicks)
plt.yticks(np.linspace(0.5,2*S-0.5,2*S,endpoint=True),np.flipud(cues_vec_3),fontsize=fontTicks)
plt.ylabel('Transient Unit Labels', fontsize=fontLabel)

d = 8
S = d+2
savestr = AuG_type+'_'+task+'_'+'distr'+str(d)+'_weights_Vm.txt'
X_a8 = np.loadtxt(savestr)

plt.subplot(2,3,4)
plt.pcolor(np.flipud(X_a8),edgecolors='k', linewidths=1)
plt.set_cmap('bwr')		
cb = plt.colorbar()
cb.ax.tick_params(labelsize=fontLabel)
tit = tit_aug+' (L='+str(d+2)+')'
plt.title(tit,fontweight="bold",fontsize=fontTitle)
plt.xticks(np.linspace(0.5,M-0.5,M,endpoint=True),mem_vec_c,fontsize=fontTicks)
plt.yticks(np.linspace(0.5,2*S-0.5,2*S,endpoint=True),np.flipud(cues_vec_8),fontsize=fontTicks)
plt.ylabel('Transient Unit Labels', fontsize=fontLabel)
plt.xlabel('Memory Unit Labels', fontsize=fontLabel)



AuG_type = 'leaky_AuG'
tit_aug = 'Leaky AuGMEnT'
d = 3
S = d+2
savestr = AuG_type+'_'+task+'_'+'distr'+str(d)+'_weights_Vm.txt'
X_l3 = np.loadtxt(savestr)

plt.subplot(2,3,2)
plt.pcolor(np.flipud(X_l3),edgecolors='k', linewidths=1)
plt.set_cmap('bwr')		
cb = plt.colorbar()
cb.ax.tick_params(labelsize=fontLabel)
tit = tit_aug+' (L='+str(d+2)+')'
plt.title(tit,fontweight="bold",fontsize=fontTitle)
plt.xticks(np.linspace(0.5,M-0.5,M,endpoint=True),mem_vec_l,fontsize=fontTicks)
plt.yticks(np.linspace(0.5,2*S-0.5,2*S,endpoint=True),np.flipud(cues_vec_3),fontsize=fontTicks)

d = 8
S = d+2
savestr = AuG_type+'_'+task+'_'+'distr'+str(d)+'_weights_Vm.txt'
X_l8 = np.loadtxt(savestr)

plt.subplot(2,3,5)
plt.pcolor(np.flipud(X_l8),edgecolors='k', linewidths=1)
plt.set_cmap('bwr')		
cb = plt.colorbar()
cb.ax.tick_params(labelsize=fontLabel)
tit = tit_aug+' (L='+str(d+2)+')'
plt.title(tit,fontweight="bold",fontsize=fontTitle)
plt.xticks(np.linspace(0.5,M-0.5,M,endpoint=True),mem_vec_l,fontsize=fontTicks)
plt.yticks(np.linspace(0.5,2*S-0.5,2*S,endpoint=True),np.flipud(cues_vec_8),fontsize=fontTicks)
plt.xlabel('Memory Unit Labels', fontsize=fontLabel)



AuG_type = 'hybrid_AuG'
tit_aug = 'Hybrid AuGMEnT'

d = 3
S = d+2
savestr = AuG_type+'_'+task+'_'+'distr'+str(d)+'_weights_Vm.txt'
X_h3 = np.loadtxt(savestr)

plt.subplot(2,3,3)
plt.pcolor(np.flipud(X_h3),edgecolors='k', linewidths=1)
plt.set_cmap('bwr')		
cb = plt.colorbar()
cb.ax.tick_params(labelsize=fontLabel)
tit = tit_aug+' (L='+str(d+2)+')'
plt.title(tit,fontweight="bold",fontsize=fontTitle)
plt.xticks(np.linspace(0.5,M-0.5,M,endpoint=True),mem_vec_h,fontsize=fontTicks)
plt.yticks(np.linspace(0.5,2*S-0.5,2*S,endpoint=True),np.flipud(cues_vec_3),fontsize=fontTicks)

d = 8
S = d+2
savestr = AuG_type+'_'+task+'_'+'distr'+str(d)+'_weights_Vm.txt'
X_h8 = np.loadtxt(savestr)

plt.subplot(2,3,6)
plt.pcolor(np.flipud(X_h8),edgecolors='k', linewidths=1)
plt.set_cmap('bwr')		
cb = plt.colorbar()
cb.ax.tick_params(labelsize=fontLabel)
tit = tit_aug+' (L='+str(d+2)+')'
plt.title(tit,fontweight="bold",fontsize=fontTitle)
plt.xticks(np.linspace(0.5,M-0.5,M,endpoint=True),mem_vec_h,fontsize=fontTicks)
plt.yticks(np.linspace(0.5,2*S-0.5,2*S,endpoint=True),np.flipud(cues_vec_8),fontsize=fontTicks)
plt.xlabel('Memory Unit Labels', fontsize=fontLabel)

plt.subplots_adjust(left=None,right=None,hspace=0.3,wspace=0.3)
plt.show()

savestr = task+'_weights_analysis.png'
FIG.savefig(savestr)
