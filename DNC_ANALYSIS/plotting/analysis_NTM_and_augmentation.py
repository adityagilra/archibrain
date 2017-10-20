import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec
import pylab

def take_accuracy(filename,step=1000):
	
	acc = []
	it = []

	f = open(filename, mode="r")

	take=False
	for line in f:
		if line[0]=='E':
			curr_it = int(line[8:13])
			if np.remainder(curr_it,step)==0:
				foll=line[13]
				if foll==':':
					it.append(curr_it)
				else:
					curr_it = int(line[8:14])
					it.append(curr_it)
				take=True
		if take==True and line[0]=='[':
			l = line[1:-2].split(" ")
			l_np = np.array([c for c in l if len(c)>1]).astype(float)
			acc.append(l_np)
			take=False
	if len(it)!=len(acc):
		print('Loading error: number of iterations not equal with number of precizion measure \t',len(it),'!=',len(acc))
		it = it[:-1]

	f.close()
	
	return it, acc

def take_accuracy_NTM(filename,step=1000):
	
	acc = []
	it = []
	cont = 0
	taken_it = 0
	taken = False

	f = open(filename, mode="r")

	for line in f:
		if taken==True:
			l = line[1:-2].split(" ")
			l_np = np.array([c for c in l if len(c)>1]).astype(float)
			acc_prov = np.concatenate([acc_prov,l_np])
			acc.append(acc_prov)
			taken=False
		elif line[0]=='[':
			cont +=1
			if cont==int(step/100):
				l = line[1:-2].split(" ")
				l_np = np.array([c for c in l if len(c)>1]).astype(float)
				taken_it += 1
				iterat = taken_it*step
				if iterat<=100000:
					it.append(iterat)
					acc_prov=l_np
					taken=True
				cont = 0

	if len(it)!=len(acc):
		print('Loading error: number of iterations not equal with number of precizion measure \t',len(it),'!=',len(acc))

	f.close()
	
	return it, acc

fontTitle = 32
fontTicks = 24
fontLabel = 28

stp = 1000
INST = ['1st instance','2nd instance', '3rd instance', '4th instance', '5th instance',
'6th instance','7th instance', '8th instance', '9th instance', '10th instance']

#################################################################################
filename = "NTM_omniglot_run.txt"
report_interval = 100

FIG_0=plt.figure(figsize=(32,15))

it_NTM,acc_NTM = take_accuracy_NTM(filename,stp)
acc_NTM = np.transpose(acc_NTM)

#plt.subplot(1,2,1)
#[plt.plot(it_NTM,acc_i,marker='o',label=str(ist)) for ist,acc_i, in zip(INST,acc_NTM)]
#tit = 'One-shot NTM'
#plt.title(tit,fontweight="bold",fontsize=fontTitle)
#plt.xlabel('Training Episodes',fontsize=fontLabel)
#plt.ylabel('Classification Accuracy',fontsize=fontLabel)
#plt.xticks([0,20000,40000,60000,80000,100000],fontsize=15)
#plt.yticks([0,0.2,0.4,0.6,0.8,1],fontsize=15)
#plt.ylim([0,1])
#plt.axhline(y=0.2, color='k', linestyle='--',alpha=0.2)
#plt.axhline(y=0.4, color='k', linestyle='--',alpha=0.2)
#plt.axhline(y=0.6, color='k', linestyle='--',alpha=0.2)
#plt.axhline(y=0.8, color='k', linestyle='--',alpha=0.2)


filename = "omniglot_02.out"
report_interval = 1000

it_DNC,acc_DNC = take_accuracy(filename,stp)
acc_DNC = np.transpose(acc_DNC)

plt.subplot(1,2,1)
[plt.plot(it_DNC,acc_i,marker='o',label=str(ist)) for ist,acc_i, in zip(INST,acc_DNC)]
tit = 'DNC - without data augmentation'
plt.title(tit,fontweight="bold",fontsize=fontTitle)
plt.xlabel('Training Episodes',fontsize=fontLabel)
plt.ylabel('Classification Accuracy',fontsize=fontLabel)
plt.xticks([0,20000,40000,60000,80000,100000],fontsize=24)
plt.yticks([0,0.2,0.4,0.6,0.8,1],fontsize=24)
plt.ylim([0,1])
plt.axhline(y=0.2, color='k', linestyle='--',alpha=0.2)
plt.axhline(y=0.4, color='k', linestyle='--',alpha=0.2)
plt.axhline(y=0.6, color='k', linestyle='--',alpha=0.2)
plt.axhline(y=0.8, color='k', linestyle='--',alpha=0.2)

filename = "omniglot_08.out"
report_interval = 1000

it_DNC_aug,acc_DNC_aug = take_accuracy(filename,stp)
acc_DNC_aug = np.transpose(acc_DNC_aug)

plt.subplot(1,2,2)
[plt.plot(it_DNC_aug,acc_i,marker='o',label=str(ist)) for ist,acc_i, in zip(INST,acc_DNC_aug)]
tit = 'DNC - with data augmentation'
plt.title(tit,fontweight="bold",fontsize=fontTitle)
plt.xlabel('Training Episodes',fontsize=fontLabel)
plt.xticks([0,20000,40000,60000,80000,100000],fontsize=24)
plt.yticks([0,0.2,0.4,0.6,0.8,1],fontsize=24)
plt.ylim([0,1])
plt.axhline(y=0.2, color='k', linestyle='--',alpha=0.2)
plt.axhline(y=0.4, color='k', linestyle='--',alpha=0.2)
plt.axhline(y=0.6, color='k', linestyle='--',alpha=0.2)
plt.axhline(y=0.8, color='k', linestyle='--',alpha=0.2)

leg = plt.legend(fontsize=20,ncol=5,shadow=True)
leg.draggable(state=True)

plt.subplots_adjust(bottom=0.2)
plt.show()

strsave='omniglot_DNC.png'
FIG_0.savefig(strsave)

