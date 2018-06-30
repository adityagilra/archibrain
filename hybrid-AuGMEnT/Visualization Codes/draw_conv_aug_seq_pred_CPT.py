import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec
import pylab

fontTitle = 32
fontTicks = 24
fontLabel = 28

#################################################################################
FIG_0=plt.figure(figsize=(16,15))

d_vec =               np.array([0,        1,        2,       3,      4,      5,       8,       10])

AUG_conv_vec_mean =   np.array([221.95, 284.4,   644.0,  2653.0,  3048.65, 4167.6,  8237.75,  9462.7])
AUG_conv_vec_sd =     np.array([71.8,    85.1,   461.7,  1687.2,  1599.3,  2444.5,   6418.3,  4530.2])

l_AUG_conv_vec_mean = np.array([257.35, 287.4,   563.0, 2289.9,   2670.8, 3275.8,   10770.6, 12926.7])
l_AUG_conv_vec_sd =   np.array([73.5,   95.9,    282.8, 1776.8,   1459.2, 1866.6,   8106.67, 6608.87])

plt.plot(d_vec, AUG_conv_vec_mean,marker='o')
plt.fill_between(d_vec, AUG_conv_vec_mean-AUG_conv_vec_sd, AUG_conv_vec_mean+AUG_conv_vec_sd,alpha=0.1,label='AuGMEnT')
plt.plot(d_vec, l_AUG_conv_vec_mean,marker='o',color='r')
plt.fill_between(d_vec, l_AUG_conv_vec_mean-l_AUG_conv_vec_sd, l_AUG_conv_vec_mean+l_AUG_conv_vec_sd,color='r',alpha=0.1,label='leaky-AuGMEnT')
tit = 'Sequence Prediction Task CPT'
plt.title(tit,fontweight="bold",fontsize=fontTitle)
plt.xlabel('Number of Distractors',fontsize=fontLabel)
plt.ylabel('Mean Convergence Time',fontsize=fontLabel)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.legend(fontsize=24)

plt.show()

strsave='seq_pred_CPT_AuG_conv.png'
FIG_0.savefig(strsave)


#################################################################################
FIG_1=plt.figure(figsize=(16,15))

d_vec = np.array([3])

lambda_vec = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]

l_AUG_conv_vec_mean = np.array([2653.0, 2386.5, 3105.9, 2289.9, 2181.0, 2250.1, 1798.2, 2934.9, 9889.3])
l_AUG_conv_vec_sd =   np.array([1687.2, 1553.3, 1901.7, 1776.8, 1175.2,  969.2, 380.0,   926.15,1624.3])

plt.plot(lambda_vec, l_AUG_conv_vec_mean,marker='o',color='g')
plt.fill_between(lambda_vec, l_AUG_conv_vec_mean-l_AUG_conv_vec_sd, l_AUG_conv_vec_mean+l_AUG_conv_vec_sd,color='g',alpha=0.3)
tit = 'Sequence Prediction CPT Task - leaky-AuGMEnT'
plt.title(tit,fontweight="bold",fontsize=fontTitle)
plt.xlabel('Leak Coefficient',fontsize=fontLabel)
plt.ylabel('Mean Convergence Time',fontsize=fontLabel)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)

plt.show()

strsave='seq_pred_CPT_AuG_leak_conv.png'
FIG_1.savefig(strsave)
