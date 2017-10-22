import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec
import pylab
import gc 

## PLOTS
fontTitle = 30
fontTicks = 24
fontLabel = 28

H_r = 0
H_m = 10

conv = np.loadtxt('RBP_conv_2.txt')[-1]

#FA_HR = np.loadtxt('RBP_angle_hr.txt')
FA_HM = np.loadtxt('RBP_angle_hm_2.txt')
FA_R = np.loadtxt('RBP_angle_r_2.txt')
FA_M = np.loadtxt('RBP_angle_m_2.txt')


average_sample = 500
xs = (np.arange(np.shape(FA_R)[0])+1)*average_sample

figRBP = plt.figure(figsize=(30,12))			
plt.subplot(1,2,1)
plt.plot(xs,np.arccos(FA_R)*180/np.pi,'b',linewidth=6,label='1st layer')
plt.xlabel('Training Episodes',fontsize=fontLabel)
plt.ylabel('Angle [degrees]',fontsize=fontLabel)
plt.xticks([0,10000,20000,30000,40000],fontsize=fontTicks)
plt.title('Feedback Alignment: Regular Branch',fontsize=fontTitle,fontweight='bold')
plt.xlim((0,conv-1000))
plt.ylim((0,125))
plt.axhline(y=30, color='k', linestyle='--',alpha=0.2)
plt.axhline(y=60, color='k', linestyle='--',alpha=0.2)
plt.axhline(y=90, color='k', linestyle='--',alpha=0.8)
plt.axhline(y=120, color='k', linestyle='--',alpha=0.2)
plt.yticks([0,30,60,90,120],fontsize=fontTicks)

if H_r!=0:
	plt.subplot(1,2,1)
	plt.plot(xs,np.arccos(FA_HR)*180/np.pi,'r',linewidth=6,label='2nd layer')

leg = plt.legend(fontsize=fontLabel)
leg.draggable(state=True)

plt.subplot(1,2,2) 
plt.plot(xs,np.arccos(FA_M)*180/np.pi,'k',linewidth=6,label='1st layer')
plt.xlabel('Training Episodes',fontsize=fontLabel)
plt.xticks([0,10000,20000,30000,40000],fontsize=fontTicks)
plt.ylabel('Angle [degrees]',fontsize=fontLabel)
plt.title('Feedback Alignment: Memory Branch',fontsize=fontTitle,fontweight='bold')
plt.xlim((0,conv-1000))
plt.ylim((0,125))
plt.axhline(y=30, color='k', linestyle='--',alpha=0.2)
plt.axhline(y=60, color='k', linestyle='--',alpha=0.2)
plt.axhline(y=90, color='k', linestyle='--',alpha=0.8)
plt.axhline(y=120, color='k', linestyle='--',alpha=0.2)
plt.yticks([0,30,60,90,120],fontsize=fontTicks)

if H_m!=0:		
	plt.subplot(1,2,2)			
	plt.plot(xs,np.arccos(FA_HM)*180/np.pi,'g',linewidth=6,label='2nd layer')

leg = plt.legend(fontsize=fontLabel)
leg.draggable(state=True)

plt.show()

saveRBPcond = 'RBP_cond.png'	
figRBP.savefig(saveRBPcond)



