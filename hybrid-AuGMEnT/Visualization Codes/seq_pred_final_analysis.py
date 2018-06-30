import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec
import pylab

fontTitle = 32
fontTicks = 24
fontLabel = 28

folder = 'DATA'
task = 'seq_prediction'

#################################################################################

d = 5

AuG_type='AuGMEnT'
str_conv = folder+'/'+AuG_type+'_'+task+'_ALL_CONV_2.txt'
AUG_conv = np.loadtxt(str_conv)
str_Q = folder+'/'+AuG_type+'_'+task+'_ALL_Q_2.txt'
AUG_Q = np.loadtxt(str_Q)
str_RPE = folder+'/'+AuG_type+'_'+task+'_ALL_RPE_2.txt'
AUG_RPE = np.loadtxt(str_RPE)
#str_resp = folder+'/'+AuG_type+'_'+task+'_ALL_RESP.txt'
#AUG_resp = np.loadtxt(str_resp)

AuG_type = 'leaky_AuG' 
str_conv = folder+'/'+AuG_type+'_'+task+'_ALL_CONV_2.txt'
l_AUG_conv = np.loadtxt(str_conv)
str_Q = folder+'/'+AuG_type+'_'+task+'_ALL_Q_2.txt'
l_AUG_Q = np.loadtxt(str_Q)
str_RPE = folder+'/'+AuG_type+'_'+task+'_ALL_RPE_2.txt'
l_AUG_RPE = np.loadtxt(str_RPE)
#str_resp = folder+'/'+AuG_type+'_'+task+'_ALL_RESP.txt'
#l_AUG_resp = np.loadtxt(str_resp)

AuG_type = 'hybrid_AuG'
str_conv = folder+'/'+AuG_type+'_'+task+'_ALL_CONV_2.txt'
h_AUG_conv = np.loadtxt(str_conv)
str_Q = folder+'/'+AuG_type+'_'+task+'_ALL_Q_2.txt'
h_AUG_Q = np.loadtxt(str_Q)
str_RPE = folder+'/'+AuG_type+'_'+task+'_ALL_RPE_2.txt'
h_AUG_RPE = np.loadtxt(str_RPE)
#str_resp = folder+'/'+AuG_type+'_'+task+'_ALL_RESP.txt'
#h_AUG_resp = np.loadtxt(str_resp)


delta = 10
end = 1500
it = np.arange(end/delta)*delta

AUG_RPE = 0.5*AUG_RPE**2
AUG_RPE_mean = np.mean(AUG_RPE, axis=0)
AUG_RPE_mean = AUG_RPE_mean[:end]
AUG_RPE_mean = np.mean(np.reshape(AUG_RPE_mean,(-1,delta)),axis=1) 
AUG_RPE_sd = np.std(AUG_RPE, axis=0)
AUG_RPE_sd = AUG_RPE_sd[:end]
AUG_RPE_sd = np.mean(np.reshape(AUG_RPE_sd,(-1,delta)),axis=1) 

l_AUG_RPE = 0.5*l_AUG_RPE**2
l_AUG_RPE_mean = np.mean(l_AUG_RPE, axis=0)
l_AUG_RPE_mean = l_AUG_RPE_mean[:end]
l_AUG_RPE_mean = np.mean(np.reshape(l_AUG_RPE_mean,(-1,delta)),axis=1) 
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

FIG_0=plt.figure(figsize=(15,10))

plt.plot(it,AUG_RPE_mean,marker='o',color='g',markersize=10,linewidth=2,label='AuGMEnT')
plt.fill_between(it,np.maximum(AUG_RPE_mean-AUG_RPE_sd,0),AUG_RPE_mean+AUG_RPE_sd,color='g',alpha=0.3)
plt.plot(it,l_AUG_RPE_mean,marker='o',color='b',markersize=10,linewidth=2,label='Leaky AuGMEnT')
plt.fill_between(it,np.maximum(l_AUG_RPE_mean-l_AUG_RPE_sd,0),l_AUG_RPE_mean+l_AUG_RPE_sd,color='b',alpha=0.3)
plt.plot(it,h_AUG_RPE_mean,marker='o',color='r',markersize=10,linewidth=2,label='Hybrid AuGMEnT')
plt.fill_between(it,np.maximum(h_AUG_RPE_mean-h_AUG_RPE_sd,0),h_AUG_RPE_mean+h_AUG_RPE_sd,color='r',alpha=0.3)
#tit = 'Sequence Prediction Task'
#plt.title(tit,fontweight="bold",fontsize=fontTitle)
plt.xlabel('Training Trials',fontsize=fontLabel)
plt.ylabel('Mean Energy Function',fontsize=fontLabel)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.legend(fontsize=24)



strsave='seq_pred_final_RPE.png'
FIG_0.savefig(strsave)

FIG_1=plt.figure(figsize=(15,10))

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


plt.plot(d_vec, AUG_conv_vec_mean,color='g',marker='o',markersize=10,linewidth=2.5,label='AuGMEnT')
plt.fill_between(d_vec, AUG_conv_vec_mean-AUG_conv_vec_sd, AUG_conv_vec_mean+AUG_conv_vec_sd,color='g',alpha=0.3)
plt.plot(d_vec, l_AUG_conv_vec_mean,marker='o',color='b',markersize=10,linewidth=2.5,label='Leaky AuGMEnT')
plt.fill_between(d_vec, l_AUG_conv_vec_mean-l_AUG_conv_vec_sd, l_AUG_conv_vec_mean+l_AUG_conv_vec_sd,color='b',alpha=0.3)
plt.plot(d_vec, h_AUG_conv_vec_mean,marker='o',color='r',markersize=10,linewidth=2.5,label='Hybrid AuGMEnT')
plt.fill_between(d_vec, h_AUG_conv_vec_mean-h_AUG_conv_vec_sd, h_AUG_conv_vec_mean+h_AUG_conv_vec_sd,color='r',alpha=0.3)
#tit = 'Sequence Prediction Task'
#plt.title(tit,fontweight="bold",fontsize=fontTitle)
plt.xlabel('Number of Distractors',fontsize=fontLabel)
plt.ylabel('Mean Convergence Time',fontsize=fontLabel)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.ylim([100,600])
plt.legend(fontsize=24)


strsave='seq_pred_final_conv.png'
FIG_1.savefig(strsave)


FIG_2=plt.figure(figsize=(15,10))

plt.semilogy(d_vec, AUG_conv_vec_mean,color='g',marker='o',markersize=10,linewidth=2.5,label='AuGMEnT')
plt.fill_between(d_vec, AUG_conv_vec_mean-AUG_conv_vec_sd, AUG_conv_vec_mean+AUG_conv_vec_sd,color='g',alpha=0.3)
plt.semilogy(d_vec, l_AUG_conv_vec_mean,marker='o',color='b',markersize=10,linewidth=2.5,label='Leaky AuGMEnT')
plt.fill_between(d_vec, l_AUG_conv_vec_mean-l_AUG_conv_vec_sd, l_AUG_conv_vec_mean+l_AUG_conv_vec_sd,color='b',alpha=0.3)
plt.semilogy(d_vec, h_AUG_conv_vec_mean,marker='o',color='r',markersize=10,linewidth=2.5,label='Hybrid AuGMEnT')
plt.fill_between(d_vec, h_AUG_conv_vec_mean-h_AUG_conv_vec_sd, h_AUG_conv_vec_mean+h_AUG_conv_vec_sd,color='r',alpha=0.3)
#tit = 'Sequence Prediction Task'
#plt.title(tit,fontweight="bold",fontsize=fontTitle)
plt.xlabel('Number of Distractors',fontsize=fontLabel)
plt.ylabel('Mean Convergence Time',fontsize=fontLabel)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.ylim([100,100000])
plt.legend(fontsize=24)

plt.show()

strsave='seq_pred_final_conv_all.png'
FIG_2.savefig(strsave)

