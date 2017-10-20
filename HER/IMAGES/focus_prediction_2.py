import numpy as np
from matplotlib import pyplot as plt
import pylab

task='12AX'

S = 8
cues_vec = ['1','2','A','B','C','X','Y','Z']

fontTitle = 28
fontTicks = 22
fontLabel = 22

str_pred = 'long_'+task+'_weights_prediction_2.txt'
W = np.loadtxt(str_pred)
print(np.shape(W))

fig2 = plt.figure(figsize=(20,16))
plt.pcolor(np.flipud(W),edgecolors='k', linewidths=1)
plt.set_cmap('Blues')			
plt.colorbar()
tit = 'PREDICTION WEIGHTS: Level 2'
plt.title(tit,fontweight="bold",fontsize=fontTitle)
dx = np.shape(W)[1]/(2*S)
plt.xticks(np.linspace(dx,np.shape(W)[1]-dx,S,endpoint=True),cues_vec,fontsize=fontTicks)
plt.yticks(np.linspace(0.5,S-0.5,S,endpoint=True),np.flipud(cues_vec),fontsize=fontTicks)

#plt.show()
savestr = task+'_weights_prediction.png'

fig2.savefig(savestr)	

W_x = np.shape(W)[1]
block_dim = int(W_x/8)
print(block_dim)

sub_block = W[0:2,block_dim*2:block_dim*4]
print(np.shape(sub_block))

lw = [1.,1.,1.,1.,2.,1.,1.,1.,2.,1.,
1.,1.,2.,1.,1.,1.,2.,1.,1.,1.,
2.,1.,1.,1.,2.,1.,1.,1.,2.,1.,
1.,1.,3.5,1.,1.,1.,2.,1.,1.,1.,
2.,1.,1.,1.,2.,1.,1.,1.,2.,1.,
1.,1.,2.,1.,1.,1.,2.,1.,1.,1.,
2.,1.,1.,1.]

fig3 = plt.figure(figsize=(12,7))
plt.pcolor(np.flipud(sub_block),edgecolors='k', linewidths=lw)
plt.set_cmap('Blues')			
plt.colorbar()
dx = np.shape(sub_block)[1]/(2*2)
plt.xticks(np.array([16,48]),['A','B'],fontsize=fontTicks)
plt.yticks(np.arange(2)+0.5,np.flipud(['1','2']),fontsize=fontTicks)

plt.show()

savestr = task+'_weights_prediction_zoom.png'
fig3.savefig(savestr)	
