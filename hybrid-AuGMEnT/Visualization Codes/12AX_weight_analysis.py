import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec
import pylab
import gc
import os 

os.chdir('..')

tit_aug = 'Hybrid AuGMEnT'
AuG_type = 'Hybrid_AuG'

S = 8        		
R = 10			     
M = 20 			     	
A = 2	

task = '12AX'

cues_vec = ['1','2','A','B','C','X','Y','Z']
cues_vec_tot = ['1+','2+','A+','B+','C+','X+','Y+','Z+','1-','2-','A-','B-','C-','X-','Y-','Z-']
mem_vec=[]
for i in range(M):
	mem_vec.append('M'+str(i+1))
act_vec= ['L','R']

fontTitle = 50
fontTicks = 24
fontLabel = 48

fig1 = plt.figure(figsize=(26,22))

X = np.loadtxt('DATA/12AX_hyb_aug_memory_V_m.txt')


plt.pcolor(np.flipud(X),edgecolors='k', linewidths=1)
plt.set_cmap('bwr')		
cb = plt.colorbar()
cb.ax.tick_params(labelsize=fontTicks)
tit = tit_aug+' - $V^M$'
plt.title(tit,fontweight="bold",fontsize=fontTitle)
plt.xticks(np.linspace(0.5,M-0.5,M,endpoint=True),mem_vec,fontsize=fontTicks)
plt.yticks(np.linspace(0.5,2*S-0.5,2*S,endpoint=True),np.flipud(cues_vec_tot),fontsize=fontTicks)
plt.xlabel('Memory Unit Labels',fontsize=fontLabel,labelpad=100)
plt.ylabel('Transient Unit Labels',fontsize=fontLabel)

savestr = 'IMAGES/'+AuG_type+'_'+task+'_weights_Vm.png'
fig1.savefig(savestr)

fig2 = plt.figure(figsize=(15,22))
X = np.loadtxt('DATA/12AX_hyb_aug_memory_W_m.txt')

plt.pcolor(np.flipud(X),edgecolors='k', linewidths=1)
plt.set_cmap('bwr')		
cb = plt.colorbar()
cb.ax.tick_params(labelsize=fontTicks)
tit = tit_aug+' - $W^M$'
plt.title(tit,fontweight="bold",fontsize=fontTitle)
plt.xticks(np.linspace(0.5,A-0.5,A,endpoint=True),act_vec,fontsize=fontTicks)
plt.yticks(np.linspace(0.5,M-0.5,M,endpoint=True),np.flipud(mem_vec),fontsize=fontTicks)
plt.ylabel('Memory Unit Labels',fontsize=fontLabel,labelpad=100)
plt.xlabel('Activity Unit Labels',fontsize=fontLabel)


plt.tight_layout()
savestr = 'IMAGES/'+AuG_type+'_'+task+'_weights_Wm.png'
fig2.savefig(savestr)
plt.show()
