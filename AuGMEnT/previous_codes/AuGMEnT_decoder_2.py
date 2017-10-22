## MAIN FILE FOR AuGMEnT TESTING
## Here are defined the settings for the AuGMEnT architecture and for the tasks to test.
## AUTHOR: Marco Martinolli
## DATE: 10.07.2017

import numpy as np
import matplotlib 
matplotlib.use('GTK3Cairo') 
from matplotlib import pyplot as plt
from matplotlib import gridspec
import pylab
import gc 
import activations as act


from TASKS.task_1_2AX_decoder import data_construction				
task = '12-AX'
	
cues_vec = ['1','2','A','B','C','X','Y','Z']
cues_vec_tot = ['1+','2+','A+','B+','C+','X+','Y+','Z+','1-','2-','A-','B-','C-','X-','Y-','Z-']
patt_vec = ['1','2', 'AX','AY','AZ', 'BX','BY','BZ', 'CX','CY','CZ']
pred_vec = ['L','R']

np.set_printoptions(precision=3)

class deep_AuGMEnT_decoder():

	def __init__(self,S,M,H,D,W_mem,W_hid,beta,leak):
 
		self.S = S
		self.M = M
		self.H = H

		self.D = D

		self.beta = beta
		self.memory_leak = leak

		self.memory_weights = W_mem
		self.hidden_weights = W_hid

		self.W_dec = 0.5*np.random.random((self.H,self.D)) - 0.25

		self.reset_memory()


	def reset_memory(self):
		self.cumulative_memory = 1e-6*np.ones((1,self.M))


	def decode(self, y_m):

		y_h = act.sigmoid(y_m,self.hidden_weights)

		decode_output = act.sigmoid(y_h, self.W_dec)
		#decode_output = act.softmax(y_output)

		return decode_output, y_h

	def update_memory(self, s_trans):

		y_m,self.cumulative_memory = act.sigmoid_acc_leaky(s_trans, self.memory_weights, self.cumulative_memory, self.memory_leak)

		return y_m		


	def update_weights(self, E, y_h, out):
		self.W_dec += self.beta*np.dot(np.transpose(y_h), E*out*(1-out))   # aggiungere derivata della parte sigmoidale E presynaptic as difference


	def define_transient(self, s,s_old):

		s_plus =  np.where(s<=s_old,0,1)
		s_minus = np.where(s_old<=s,0,1)
		s_trans = np.concatenate((s_plus,s_minus),axis=1)

		return s_trans


	def training(self,S_train,D_train,verbose=False):

		N_stimuli = np.shape(S_train)[0]
		zero = np.zeros((1,self.S))
		s_old = zero

		reset_case = ['1','2']
		pause_case = ['A','B','C']
		dic_stim = {0:'1',1:'2',2:'A',3:'B',4:'C',5:'X',6:'Y',7:'Z'}
		dic_dec = {0:'1',1:'2',
			2:'AX',3:'AY',4:'AZ',
			5:'BX',6:'BY',7:'BZ',
			8:'CX',9:'CY',10:'CZ'}	

		L = []
		ep = 0
		loss = 0
		loss_avg = 0

		m = 0
		for n in np.arange(N_stimuli):

			s = S_train[n:(n+1),:]
			s_print = dic_stim[np.argmax(s)]
			s_trans = self.define_transient(s, s_old)
			s_old = s

			if s_print in reset_case:
				if verbose:
					print('RESET \n')
				ep += 1
				self.reset_memory()

			y_m = self.update_memory(s_trans)
			#print(s_print)

			if s_print not in pause_case:

				d = D_train[m:(m+1),:]
				d_print = dic_dec[np.argmax(d)]
				#print(d,'\t',d_print)

				dec_out, y_h = self.decode(y_m)
				p = np.max(dec_out)
				r_print = dic_dec[np.argmax(dec_out)]
				if verbose:
					print('ITER: ',m+1,'\t PATTERN: ',d_print,'\t DECODED IN: ',r_print,'(',p,')')

				E = d - dec_out		# Decoder Error
				loss += -np.log(p)
				L.append(loss)
				if np.remainder(m,1000)==0 and m!=0:
					print('Iteration: ',m+1,'\t LOSS: ', loss_avg)
					loss_avg = 0
				else:
					loss_avg -= np.log(p)
				self.update_weights(E, y_h, dec_out)

				m += 1		

		return L


N = 2000000
p_c = 0.5

np.random.seed(1)
print('Dataset construction...')
S_tr,D_tr = data_construction(N,p_c)


prop_system = ['std','BP','RBP','SRBP','MRBP']
prop = 'RBP'
model_opt_system = ['base','deep','hier']
model_opt = 'deep'

weight_folder = 'WEIGHT_DATA'
image_folder = 'IMAGES'
str_weight_mem = weight_folder+'/'+model_opt+'_'+prop+'_W_mem.txt'
W_m = np.loadtxt(str_weight_mem)
str_weight_hid = weight_folder+'/'+model_opt+'_'+prop+'_W_hid.txt'
W_h = np.loadtxt(str_weight_hid)

## CONSTRUCTION OF THE AuGMEnT NETWORK
S = np.shape(S_tr)[1]        # dimension of the input = number of possible stimuli
M = np.shape(W_m)[1]	     # dimension of the memory units
H = np.shape(W_h)[1]
D = np.shape(D_tr)[1]

mem_vec=[]
for i in range(M):
	mem_vec.append('M'+str(i+1))	
hid_vec=[]
for i in range(H):
	hid_vec.append('H'+str(i+1))	

beta = 0.08			# weight update coefficient
leak = 0.68

verb = 1
	
do_training = True
do_plots = True

spec_opt = None

if model_opt == 'base' or model_opt == 'deep':
	model = deep_AuGMEnT_decoder(S, M, H, D, W_m, W_h, beta, leak)

if model_opt == 'hier':
	L = 3
	ALPHA = [0.1,0.5,0.9]
	BETA = [0.1,0.01,0.01]
	LEAK = [0.1,0.5,0.9]

## TRAINING
if do_training:
	L = model.training(S_tr,D_tr,verb)

## PLOTS
fontTitle = 26
fontTicks = 22
fontLabel = 22

fig = plt.figure(figsize=(10,8))
W_dec = model.W_dec
plt.pcolor(np.flipud(W_dec),edgecolors='k', linewidths=1)
plt.set_cmap('Greens')		
plt.colorbar()
tit = 'MEMORY DECODED WEIGHTS'
plt.title(tit,fontweight="bold",fontsize=fontTitle)			
plt.yticks(np.linspace(0.5,H-0.5,H,endpoint=True),np.flipud(hid_vec),fontsize=fontTicks)
plt.xticks(np.linspace(0.5,D-0.5,D,endpoint=True),patt_vec,fontsize=fontTicks)

plt.show()

savestr = image_folder+'/decoding_weights_hidden.png'
fig.savefig(savestr)
