import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

np.set_printoptions(precision=2)

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
	
task = 'seq_pred'

fontTitle = 32
fontTicks = 24
fontLabel = 28

figx = 22
figx2 = 6
figy = 12

sz_min = 0.05
sz_max = 0.65

fig_3 = plt.figure(figsize=(figx,figy))

################################################################################
################################################################################
d = 3
S = d+2

AuG_type = 'hyb'
tit_aug = 'Hybrid AuGMEnT'

loadstr = task+'/'+AuG_type+'_'+'distr'+str(d)+'_Vm.txt'
X_h3 = np.loadtxt(loadstr)
AuG_type = 'AuG'
tit_aug = 'AuGMEnT'

loadstr = task+'/'+AuG_type+'_'+'distr'+str(d)+'_Vm.txt'
X_a3 = np.loadtxt(loadstr)

print(X_a3)
Y,X = np.shape(X_a3)
ax = fig_3.add_subplot(1,2,1)

dil = 2
plt.xlim((0,dil*X))
plt.ylim((0,Y))
plt.gca().set_aspect('equal', adjustable='box')
tit = tit_aug+' (L='+str(d+2)+')'
plt.title(tit,fontweight="bold",fontsize=fontTitle)
plt.ylabel('Transient Unit Labels', fontsize=fontLabel)
plt.xlabel('Memory Unit Labels', fontsize=fontLabel)
plt.xticks(np.linspace(0.5*dil,(M-0.5)*dil,M,endpoint=True),mem_vec_c,fontsize=fontTicks)
plt.yticks(np.linspace(0.5,2*S-0.5,2*S,endpoint=True),np.flipud(cues_vec_3),fontsize=fontTicks)
ax.grid(True)

MAX_a = np.max(np.abs(X_a3))
MIN_a = np.min(np.abs(X_a3))
print('MAX: ',MAX_a)
print('MIN: ',MIN_a)
MAX_h = np.max(np.abs(X_h3))
MIN_h = np.min(np.abs(X_h3))
print('MAX: ',MAX_h)
print('MIN: ',MIN_h)
MAX = np.maximum(MAX_a,MAX_h)
MIN = np.minimum(MIN_a,MIN_h)
print('MAX: ',MAX)
print('MIN: ',MIN)

print('\n\n\n')

for i in np.arange(X):
	for j in np.arange(Y):
		j_inv = Y - j -1
		dim = sz_min + sz_max*(np.abs(X_a3[j_inv,i])-MIN)/(MAX-MIN)
		if X_a3[j_inv,i]>=0:
			col = 'red'
		else:
			col = 'blue'
		ax.add_patch(patches.Rectangle(xy=((i+0.5-(dim/2))*dil, j+0.5-(dim/2)),width=dim*dil,height=dim,facecolor=col))

AuG_type = 'hyb'
tit_aug = 'Hybrid AuGMEnT'

loadstr = task+'/'+AuG_type+'_'+'distr'+str(d)+'_Vm.txt'
X_h3 = np.loadtxt(loadstr)

print(X_h3)
Y,X = np.shape(X_h3)
ax = fig_3.add_subplot(1,2,2)

dil = 2
plt.xlim((0,dil*X))
plt.ylim((0,Y))
plt.gca().set_aspect('equal', adjustable='box')
tit = tit_aug+' (L='+str(d+2)+')'
plt.title(tit,fontweight="bold",fontsize=fontTitle)
plt.ylabel('Transient Unit Labels', fontsize=fontLabel)
plt.xlabel('Memory Unit Labels', fontsize=fontLabel)
plt.xticks(np.linspace(0.5*dil,(M-0.5)*dil,M,endpoint=True),mem_vec_h,fontsize=fontTicks)
plt.yticks(np.linspace(0.5,2*S-0.5,2*S,endpoint=True),np.flipud(cues_vec_3),fontsize=fontTicks)
ax.grid(True)

print('\n\n\n')

for i in np.arange(X):
	for j in np.arange(Y):
		j_inv = Y - j -1
		dim = sz_min + sz_max*(np.abs(X_h3[j_inv,i])-MIN)/(MAX-MIN)
		if X_h3[j_inv,i]>=0:
			col = 'red'
		else:
			col = 'blue'
		ax.add_patch(patches.Rectangle(xy=((i+0.5-(dim/2))*dil, j+0.5-(dim/2)),width=dim*dil,height=dim,facecolor=col))

#scale = [0.01, 0.4, 0.8, 1.35]
#for j in np.arange(len(scale)):
#	dim = sz_min + sz_max*(scale[j]-MIN)/(MAX-MIN)
#	print('\n',dim)
#	print((0.5-(dim/2))*dil)
#	print(j+0.5-(dim/2))
#	ax.add_patch(patches.Rectangle(xy=((i+1.25+0.5-(dim/2))*dil, 2*(j+0.25)+1-(dim/2)),width=dim*dil,height=dim,facecolor='black',clip_on=False))
#	txt = '  value = '+str(scale[j])
#	ax.text(x=(i+1.6+0.5)*dil, y=2*(j+0.25)+1-0.11,s=txt,fontsize=fontTicks)
	
#ax.text(x=(i+1.25+0.5)*dil, y=2*(j+0.75+0.25)+1-0.11,s='SCALEBAR',fontsize=fontLabel,weight='bold')	
#plt.subplots_adjust(left=0.1, right=0.8, top=0.9, bottom=0.1)
		
plt.show()
strsave='seq_pred_d3.pdf'
fig_3.savefig(strsave)

####

#fig = plt.figure(figsize=(figx2,figy))
#ax = fig.add_subplot(1,1,1)
#scale = [0.01, 0.4, 0.8, 1.35]
#print(len(scale))
#for j in np.arange(len(scale)):
#	dim = sz_min + sz_max*(scale[j]-MIN)/(MAX-MIN)
#	print('\n',dim)
#	print((0.5-(dim/2))*dil)
#	print(j+0.5-(dim/2))
#	ax.add_patch(patches.Rectangle(xy=(0.5-((dim/2))*dil, j+1-(dim/2)),width=dim*dil,height=dim,facecolor='black'))

#plt.xlim(-0.2,1.5)	
#plt.ylim((0,len(scale)))	
#plt.axis('off')
#plt.show()

################################################################################
################################################################################

fig_8 = plt.figure(figsize=(figx,figy))

################################################################################
################################################################################
d = 8
S = d+2


AuG_type = 'hyb'
tit_aug = 'Hybrid AuGMEnT'

loadstr = task+'/'+AuG_type+'_'+'distr'+str(d)+'_Vm.txt'
X_h8 = np.loadtxt(loadstr)

AuG_type = 'AuG'
tit_aug = 'AuGMEnT'

loadstr = task+'/'+AuG_type+'_'+'distr'+str(d)+'_Vm.txt'
X_a8 = np.loadtxt(loadstr)

print(X_a8)
Y,X = np.shape(X_a8)
ax = fig_8.add_subplot(1,2,1)

dil = 3
plt.xlim((0,dil*X))
plt.ylim((0,Y))
#plt.gca().set_aspect('equal', adjustable='box')
tit = tit_aug+' (L='+str(d+2)+')'
plt.title(tit,fontweight="bold",fontsize=fontTitle)
plt.ylabel('Transient Unit Labels', fontsize=fontLabel)
plt.xlabel('Memory Unit Labels', fontsize=fontLabel)
plt.xticks(np.linspace(0.5*dil,(M-0.5)*dil,M,endpoint=True),mem_vec_c,fontsize=fontTicks)
plt.yticks(np.linspace(0.5,2*S-0.5,2*S,endpoint=True),np.flipud(cues_vec_8),fontsize=fontTicks)
ax.grid(True)

MAX_a = np.max(np.abs(X_a8))
MIN_a = np.min(np.abs(X_a8))
print('MAX: ',MAX_a)
print('MIN: ',MIN_a)
MAX_h = np.max(np.abs(X_h8))
MIN_h = np.min(np.abs(X_h8))
print('MAX: ',MAX_h)
print('MIN: ',MIN_h)
MAX = np.maximum(MAX_a,MAX_h)
MIN = np.minimum(MIN_a,MIN_h)
print('MAX: ',MAX)
print('MIN: ',MIN)

print('\n\n\n')

for i in np.arange(X):
	for j in np.arange(Y):
		j_inv = Y - j -1
		dim = sz_min + sz_max*(np.abs(X_a8[j_inv,i])-MIN)/(MAX-MIN)
		if X_a8[j_inv,i]>=0:
			col = 'red'
		else:
			col = 'blue'
		ax.add_patch(patches.Rectangle(xy=((i+0.5-(dim/2))*dil, j+0.5-(dim/2)),width=dil*dim,height=dim,facecolor=col))

AuG_type = 'hyb'
tit_aug = 'Hybrid AuGMEnT'

loadstr = task+'/'+AuG_type+'_'+'distr'+str(d)+'_Vm.txt'
X_h8 = np.loadtxt(loadstr)

print(X_h8)
Y,X = np.shape(X_h8)
ax = fig_8.add_subplot(1,2,2)

dil = 3
plt.xlim((0,dil*X))
plt.ylim((0,Y))
#plt.gca().set_aspect('equal', adjustable='box')
tit = tit_aug+' (L='+str(d+2)+')'
plt.title(tit,fontweight="bold",fontsize=fontTitle)
plt.ylabel('Transient Unit Labels', fontsize=fontLabel)
plt.xlabel('Memory Unit Labels', fontsize=fontLabel)
plt.xticks(np.linspace(0.5*dil,(M-0.5)*dil,M,endpoint=True),mem_vec_h,fontsize=fontTicks)
plt.yticks(np.linspace(0.5,2*S-0.5,2*S,endpoint=True),np.flipud(cues_vec_8),fontsize=fontTicks)
ax.grid(True)

print('\n\n\n')

for i in np.arange(X):
	for j in np.arange(Y):
		j_inv = Y - j -1
		dim = sz_min + sz_max*(np.abs(X_h8[j_inv,i])-MIN)/(MAX-MIN)
		if X_h8[j_inv,i]>=0:
			col = 'red'
		else:
			col = 'blue'
		ax.add_patch(patches.Rectangle(xy=((i+0.5-(dim/2))*dil, j+0.5-(dim/2)),width=dil*dim,height=dim,facecolor=col))

#scale = [0.01, 0.5, 1.0, 1.49]
#for j in np.arange(len(scale)):
#	dim = sz_min + sz_max*(scale[j]-MIN)/(MAX-MIN)
#	print('\n',dim)
#	print((0.5-(dim/2))*dil)
#	print(j+0.5-(dim/2))
#	ax.add_patch(patches.Rectangle(xy=((i+1.225+0.5-(dim/2))*dil, 4*(j+0.25)+1.5-(dim/2)),width=dim*dil,height=dim,facecolor='black',clip_on=False))
#	txt = '  value = '+str(scale[j])
#	ax.text(x=(i+1.6+0.5)*dil, y=4*(j+0.25)+1.5-0.2,s=txt,fontsize=fontTicks)
	
#ax.text(x=(i+1.23+0.5)*dil, y=4*(j+0.75+0.25)+1.5-0.2,s='SCALEBAR',fontsize=fontLabel,weight='bold')	
#plt.subplots_adjust(left=0.1, right=0.8, top=0.9, bottom=0.1)
plt.show()
strsave='seq_pred_d8.pdf'
fig_8.savefig(strsave)	
