import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from matplotlib import pyplot, lines

np.set_printoptions(precision=2)
	
S = 8        		
R = 10			     
M = 20 			     	
A = 2	

cues_vec = ['1','2','A','B','C','X','Y','Z']
cues_vec_tot = ['1+','2+','A+','B+','C+','X+','Y+','Z+','1-','2-','A-','B-','C-','X-','Y-','Z-']
mem_vec=[]
for i in range(M):
	mem_vec.append('M'+str(i+1))
act_vec= ['L','R']
	
task = '12AX'

fontTitle = 32
fontTicks = 24
fontLabel = 28

figx = 22
figx2 = 8
figy = 13

sz_min = 0.05
sz_max = 0.7

fig = plt.figure(figsize=(figx,figy))

################################################################################
################################################################################
AuG_type = 'hyb'
tit_aug = 'Hybrid AuGMEnT'

loadstr = task+'/'+AuG_type+'_V_m.txt'
Vm = np.loadtxt(loadstr)

Y,X = np.shape(Vm)
ax = fig.add_subplot(1,1,1)

dil = 1.5
plt.xlim((0,dil*X))
plt.ylim((0,Y))
plt.gca().set_aspect('equal', adjustable='box')
tit = tit_aug+' - $V^M$'
plt.title(tit,fontweight="bold",fontsize=fontTitle)
plt.ylabel('Transient Unit Labels', fontsize=fontLabel,labelpad=15)
plt.xlabel('Memory Unit Labels', fontsize=fontLabel,labelpad=60)
plt.xticks(np.linspace(0.5*dil,(M-0.5)*dil,M,endpoint=True),mem_vec,fontsize=fontTicks)
plt.yticks(np.linspace(0.5,2*S-0.5,2*S,endpoint=True),np.flipud(cues_vec_tot),fontsize=fontTicks)
ax.grid(True)
ax.annotate('', xy=(0, -0.08), xycoords='axes fraction', xytext=(0.5, -0.08), 
            arrowprops=dict(width=2))
ax.annotate('', xy=(0.5, -0.08), xycoords='axes fraction', xytext=(0.1, -0.08), 
            arrowprops=dict(width=2))
ax.annotate('', xy=(0.5, -0.08), xycoords='axes fraction', xytext=(1, -0.08), 
            arrowprops=dict(width=2))
ax.annotate('', xy=(1, -0.08), xycoords='axes fraction', xytext=(0.6, -0.08), 
            arrowprops=dict(width=2))

ax.text(x=5.8,y=-2.2,s='Leaky',fontsize=fontLabel)          
ax.text(x=20.5,y=-2.2,s='Non-Leaky',fontsize=fontLabel) 

ttl = ax.title
ttl.set_position([.5, 1.05])         

MAX = np.max(np.abs(Vm))
MIN = np.min(np.abs(Vm))

print('MAX: ',MAX)
print('MIN: ',MIN)

print('\n\n\n')

for i in np.arange(X):
	for j in np.arange(Y):
		j_inv = Y - j -1
		dim = sz_min + sz_max*(np.abs(Vm[j_inv,i])-MIN)/(MAX-MIN)
		if Vm[j_inv,i]>=0:
			col = 'red'
		else:
			col = 'blue'
		ax.add_patch(patches.Rectangle(xy=((i+0.5-(dim/2))*dil, j+0.5-(dim/2)),width=dim*dil,height=dim,facecolor=col))
		

#scale = [0.1, 1.5, 3.0, 4.5]
#for j in np.arange(len(scale)):
#	dim = sz_min + sz_max*(scale[j]-MIN)/(MAX-MIN)
#	print('\n',dim)
#	print((0.5-(dim/2))*dil)
#	print(j+0.5-(dim/2))
#	ax.add_patch(patches.Rectangle(xy=((i+2+0.5-(dim/2))*dil, 3*(j+0.25)+1.5-(dim/2)),width=dim*dil,height=dim,facecolor='black',clip_on=False))
#	txt = '  value = '+str(scale[j])
#	ax.text(x=(i+2.6+0.5)*dil, y=3*(j+0.25)+1.5-0.2,s=txt,fontsize=fontTicks)
	
#ax.text(x=(i+1.9+0.5)*dil, y=3*(j+0.75+0.25)+1.5-0.2,s='SCALEBAR',fontsize=fontLabel,weight='bold')		
#plt.subplots_adjust(left=0.1, right=0.8, top=0.9, bottom=0.1)

plt.show()
strsave='12AX_Vm.pdf'
fig.savefig(strsave)

################################################################################

fig = plt.figure(figsize=(figx2,figy))

loadstr = task+'/'+AuG_type+'_W_m.txt'
Wm = np.loadtxt(loadstr)

Y,X = np.shape(Wm)
ax = fig.add_subplot(1,1,1)

dil = 3
plt.xlim((0,dil*X))
plt.ylim((0,Y))
plt.gca().set_aspect('equal', adjustable='box')
tit = tit_aug+' - $W^M$'
plt.title(tit,fontweight="bold",fontsize=fontTitle)
plt.xticks(np.linspace(dil*0.5,dil*(A-0.5),A,endpoint=True),act_vec,fontsize=fontTicks)
plt.yticks(np.linspace(0.5,M-0.5,M,endpoint=True),np.flipud(mem_vec),fontsize=fontTicks)
plt.ylabel('Memory Unit Labels',fontsize=fontLabel,labelpad=70)
plt.xlabel('Activity Unit Labels',fontsize=fontLabel)
ax.grid(True)
ax.annotate('', xy=(-0.375, 0), xycoords='axes fraction', xytext=(-0.375, 0.5), 
            arrowprops=dict(width=2))
ax.annotate('', xy=(-0.375, 0.5), xycoords='axes fraction', xytext=(-0.375, 0.1), 
            arrowprops=dict(width=2))
ax.annotate('', xy=(-0.375, 0.5), xycoords='axes fraction', xytext=(-0.375, 1), 
            arrowprops=dict(width=2))
ax.annotate('', xy=(-0.375, 1), xycoords='axes fraction', xytext=(-0.375,0.6), 
            arrowprops=dict(width=2))

ax.text(x=-3.5,y=16,s='Leaky',fontsize=fontLabel,rotation=90)          
ax.text(x=-3.5,y=7,s='Non-Leaky',fontsize=fontLabel,rotation=90)
ttl = ax.title
ttl.set_position([.5, 1.05])

MAX = np.max(np.abs(Wm))
MIN = np.min(np.abs(Wm))

print('MAX: ',MAX)
print('MIN: ',MIN)

print('\n\n\n')

for i in np.arange(X):
	for j in np.arange(Y):
		j_inv = Y - j -1
		dim = sz_min + sz_max*(np.abs(Wm[j_inv,i])-MIN)/(MAX-MIN)
		if Wm[j_inv,i]>=0:
			col = 'red'
		else:
			col = 'blue'
		ax.add_patch(patches.Rectangle(xy=((i+0.5-(dim/2))*dil, j+0.5-(dim/2)),width=dim*dil,height=dim,facecolor=col))

#scale = [0.1, 0.5, 1.0, 2.0]
#for j in np.arange(len(scale)):
#	dim = sz_min + sz_max*(scale[j]-MIN)/(MAX-MIN)
#	print('\n',dim)
#	print((0.5-(dim/2))*dil)
#	print(j+0.5-(dim/2))
#	ax.add_patch(patches.Rectangle(xy=((i+1.2+0.5-(dim/2))*dil, 4*(j+0.25)+1.5-(dim/2)),width=dim*dil,height=dim,facecolor='black',clip_on=False))
#	txt = '  value = '+str(scale[j])
#	ax.text(x=(i+1.7+0.5)*dil, y=4*(j+0.25)+1.5-0.2,s=txt,fontsize=fontTicks)
	
#ax.text(x=(i+1.3+0.5)*dil, y=4*(j+0.75+0.25)+1.5-0.2,s='SCALEBAR',fontsize=fontLabel,weight='bold')		
#plt.subplots_adjust(left=0.1, right=0.75, top=0.9, bottom=0.1)

plt.show()
strsave='12AX_Wm.pdf'
fig.savefig(strsave)
