import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec
import pylab
import os

os.chdir("..")

fontTitle = 32
fontTicks = 24
fontLabel = 28

folder = 'DATA'
task = 'seq_prediction'

#################################################################################
FIG_0=plt.figure(figsize=(30,15))

d_vec = np.arange(11)

AuG_type='AuGMEnT'
str_conv = folder+'/'+AuG_type+'_'+task+'_CONV.txt'
str_sd = folder+'/'+AuG_type+'_'+task+'_SD.txt'
AUG_conv_vec_mean = np.loadtxt(str_conv)
AUG_conv_vec_sd = np.loadtxt(str_sd)

AuG_type = 'leaky_AuG' 
str_conv = folder+'/'+AuG_type+'_'+task+'_CONV.txt'
str_sd = folder+'/'+AuG_type+'_'+task+'_SD.txt'
l_AUG_conv_vec_mean = np.loadtxt(str_conv)
l_AUG_conv_vec_sd = np.loadtxt(str_sd)

AuG_type = 'hybrid_AuG'
str_conv = folder+'/'+AuG_type+'_'+task+'_CONV.txt'
str_sd = folder+'/'+AuG_type+'_'+task+'_SD.txt'
h_AUG_conv_vec_mean = np.loadtxt(str_conv)
h_AUG_conv_vec_sd = np.loadtxt(str_sd)

plt.subplot(1,2,1)
plt.plot(d_vec, AUG_conv_vec_mean,marker='o',label='AuGMEnT')
plt.fill_between(d_vec, AUG_conv_vec_mean-AUG_conv_vec_sd, AUG_conv_vec_mean+AUG_conv_vec_sd,alpha=0.1)
plt.plot(d_vec, l_AUG_conv_vec_mean,marker='o',color='r',label='Leaky AuGMEnT')
plt.fill_between(d_vec, l_AUG_conv_vec_mean-l_AUG_conv_vec_sd, l_AUG_conv_vec_mean+l_AUG_conv_vec_sd,color='r',alpha=0.1)
plt.plot(d_vec, h_AUG_conv_vec_mean,marker='o',color='purple',label='Hybrid AuGMEnT')
plt.fill_between(d_vec, h_AUG_conv_vec_mean-h_AUG_conv_vec_sd, h_AUG_conv_vec_mean+h_AUG_conv_vec_sd,color='purple',alpha=0.1)
tit = 'Sequence Prediction Task'
plt.title(tit,fontweight="bold",fontsize=fontTitle)
plt.xlabel('Number of Distractors',fontsize=fontLabel)
plt.ylabel('Mean Convergence Time',fontsize=fontLabel)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.ylim([0,500])
plt.legend(fontsize=24)

plt.subplot(1,2,2)
plt.plot(d_vec, AUG_conv_vec_mean,marker='o',label='AuGMEnT')
plt.fill_between(d_vec, AUG_conv_vec_mean-AUG_conv_vec_sd, AUG_conv_vec_mean+AUG_conv_vec_sd,alpha=0.3)
plt.plot(d_vec, l_AUG_conv_vec_mean,marker='o',color='r',label='Leaky AuGMEnT')
plt.fill_between(d_vec, l_AUG_conv_vec_mean-l_AUG_conv_vec_sd, l_AUG_conv_vec_mean+l_AUG_conv_vec_sd,color='r',alpha=0.3)
plt.plot(d_vec, h_AUG_conv_vec_mean,marker='o',color='purple',label='Hybrid AuGMEnT')
plt.fill_between(d_vec, h_AUG_conv_vec_mean-h_AUG_conv_vec_sd, h_AUG_conv_vec_mean+h_AUG_conv_vec_sd,color='purple',alpha=0.3)
tit = 'Sequence Prediction Task'
plt.title(tit,fontweight="bold",fontsize=fontTitle)
plt.xlabel('Number of Distractors',fontsize=fontLabel)
plt.ylabel('Mean Convergence Time',fontsize=fontLabel)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.legend(fontsize=24)

plt.show()

strsave='IMAGES/seq_pred_AuG_conv_FINAL.png'
FIG_0.savefig(strsave)
