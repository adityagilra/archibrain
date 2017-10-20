## CLASS OF LSTM ARCHITECTURE 
##
## AUTHOR: Marco Martinolli
## DATE: 11.04.2017

from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM
from keras import backend as K
from keras.regularizers import l2
from keras.optimizers import SGD,Adam
import numpy as np


from task_1_2AX import data_construction

class LSTM_arch():

	## Attribute
	# ------------
	# S - size of the input stimulus
	# H - number of hidden units
	# O - number of the output units
	# learn_rate - learning rate

	def __init__(self,S,H,O,learn_rate,batch_size=1,dt=1,dic_stim=None,dic_resp=None):	
	
		self.S = S
		self.H = H
		self.O = O

		self.dic_stim = dic_stim
		self.dic_resp = dic_resp

		self.b_sz = batch_size
		self.time_steps = dt

		opt = Adam(lr=learn_rate) 

		if self.O==2:
			ls_fct = 'binary_crossentropy'
		else:
			ls_fct = 'categorical_crossentropy'

		# LSTM model with stateful
		self.LSTM = Sequential()
		self.LSTM.add(LSTM(batch_input_shape=(self.b_sz,self.time_steps,self.S), units=self.H, activation='sigmoid'))
		self.LSTM.add(Dense(units=self.O, activation='softmax'))
		self.LSTM.compile(loss=ls_fct, optimizer=opt, metrics=['accuracy'])


	def training_saccade(self,S_train,O_train,stop=True):

		#self.LSTM.fit(S_train, O_train, epochs=ep, batch_size=self.b_sz, shuffle=False)	

		N = np.shape(S_train)[0]
		convergence = False
		E = np.zeros((N))
		conv_ep = np.array([0])

		trial_PL = np.zeros(50)
		trial_PR = np.zeros(50)
		trial_AL = np.zeros(50)
		trial_AR = np.zeros(50)	

		num_PL = 0
		num_PR = 0
		num_AL = 0
		num_AR = 0

		prop_PL = 0
		prop_PR = 0
		prop_AL = 0
		prop_AR = 0				

		for n in np.arange(N):

			# self.LSTM.reset_states()

			s = S_train[n:(n+1),:,:]				
			o = O_train[n,-1:,:]
		
			r = self.LSTM.predict(s,batch_size=self.b_sz)
			resp_ind = np.argmax(r)
			self.LSTM.fit(s, o, epochs=1, batch_size=self.b_sz, shuffle=False,verbose=0)

			o_print = self.dic_resp[repr(o.astype(int))]
			r_print = self.dic_resp[repr(resp_ind)]

			print('TRIAL: ',n,'\t OUT: ',o_print,'\t RESP: ',r_print)

			s_fix = np.reshape(s[0,1,:], (1,-1))
			s_fix = self.dic_stim[repr(s_fix.astype(int))]

			s_cue = np.reshape(s[0,2,:], (1,-1))
			s_cue = self.dic_stim[repr(s_cue.astype(int))]
			#print(s_fix, '\t', s_cue,'\t OUT: ',o_print,'\t RESP: ',r_print ,'(',r,')')
			if s_fix=='P' and s_cue=='L':
				num_PL += 1
				if r_print == o_print:
					trial_PL[(num_PL-1) % 50] = 1
				else:
					trial_PL[(num_PL-1) % 50] = 0
					E[n] = 1
				prop_PL = np.mean(trial_PL)
				#print('TRIAL ',n,'\t PL - ',prop_PL)
			if s_fix=='P' and s_cue=='R':
				num_PR += 1
				if r_print == o_print:
					trial_PR[(num_PR-1) % 50] = 1
				else:
					trial_PR[(num_PR-1) % 50] = 0
					E[n] = 1
				prop_PR = np.mean(trial_PR)
				#print('TRIAL ',n,'\t PR - ',prop_PR)
			if s_fix=='A' and s_cue=='L':
				num_AL += 1
				if r_print == o_print:
					trial_AL[(num_AL-1) % 50] = 1
				else:
					trial_AL[(num_AL-1) % 50] = 0
					E[n] = 1
				prop_AL = np.mean(trial_AL)
				#print('TRIAL ',n,'\t AL - ',prop_AL)
	
			if s_fix=='A' and s_cue=='R':
				num_AR += 1
				if r_print == o_print:
					trial_AR[(num_AR-1) % 50] = 1	
				else:
					trial_AR[(num_AR-1) % 50] = 0
					E[n] = 1
				prop_AR = np.mean(trial_AR)
				#print('TRIAL ',n,'\t AR - ',prop_AR)

			if convergence==False and prop_PL>=0.9 and prop_PR>=0.9 and prop_AL>=0.9 and prop_AR>=0.9:
				convergence = True
				conv_ep = np.array([n])
				if stop:
					break
					
			if np.remainder(n,1000)==0 and n!=0:
				print('TRIAL ',n,'\t PL:',prop_PL,'PR:',prop_PR,'AL:',prop_AL,'AR:',prop_AR)

		if convergence==True:
			print('SIMULATION MET CRITERION AT EPISODE ',conv_ep)

		return E, conv_ep
								

	def training_12AX(self,N,p_t,criterion='strong',stop=True):

		#self.LSTM.fit(S_train, O_train, epochs=ep, batch_size=self.b_sz, shuffle=False)	


		convergence = False
		E = np.zeros((N))
		conv_ep = np.array([0])

		correct = 0
		tr = 0

		for tr in np.arange(N):
			print('TRIAL N',tr+1)

			S_tr, O_tr = data_construction(1,p_t)
			N_stimuli = np.shape(S_tr)[0]
			
			for n in np.arange(N_stimuli):

				self.LSTM.reset_states()

				s = np.zeros((1,self.time_steps,8))
				s[0,-(n+1):,:] = S_tr[:(n+1),:]

				o = O_tr[n:(n+1),:]
			
				r = self.LSTM.predict(s,batch_size=self.b_sz)
				resp_ind = np.argmax(r)
				self.LSTM.fit(s, o, epochs=1, batch_size=self.b_sz, shuffle=False,verbose=0)
	
				o_print = self.dic_resp[repr(o.astype(int))]
				r_print = self.dic_resp[repr(resp_ind)]

				s = np.reshape(s[0,self.time_steps-1,:], (1,-1))
				s_print = self.dic_stim[repr(s.astype(int))]

				if r_print==o_print:
					correct += 1
				else:
					correct = 0
					E[tr] += 1

				print('\t S: ',s_print,'\t OUT: ',o_print,'\t RESP: ',r_print,'\t corr_acc:',correct)
			
				if criterion=='lenient':		
					if convergence==False and correct==2*25*6:
						convergence = True
						conv_ep = np.array([tr])
				elif criterion=='strong':
					if convergence==False and correct==1000:
						convergence = True
						conv_ep = np.array([tr])
			if convergence==True:
				#print('SIMULATION MET CRITERION AT ITERATION ',conv_ep)
				if stop:
					break

	
		return E, conv_ep
								
									

	def test_saccades(self,S_test,O_test):

		N = np.shape(S_test)[0]
		corr = 0			

		for n in np.arange(N):

			self.LSTM.reset_states()

			s = S_test[n:(n+1),:,:]				
			o = O_test[n,-1:,:]
		
			r = self.LSTM.predict(s,batch_size=self.b_sz)
			resp_ind = np.argmax(r)

			o_print = self.dic_resp[repr(o.astype(int))]
			r_print = self.dic_resp[repr(resp_ind)]

			if o_print==r_print:
				corr += 1
		
		perc = 100*float(corr)/float(N)				

		return perc


	def test_12AX(self,N,p_t=0.5):

		corr = 0

		for tr in np.arange(N):

			S_tst, O_tst = data_construction(1,p_t)
			N_stimuli = np.shape(S_tst)[0]

			corr_ep_bool = True 
			
			for n in np.arange(N_stimuli):

				self.LSTM.reset_states()

				s = np.zeros((1,self.time_steps,8))
				s[0,-(n+1):,:] = S_tst[:(n+1),:]

				o = O_tst[n:(n+1),:]


				r = self.LSTM.predict(s,batch_size=self.b_sz)
				resp_ind = np.argmax(r)

				o_print = self.dic_resp[repr(o.astype(int))]
				r_print = self.dic_resp[repr(resp_ind)]

				s = np.reshape(s[0,self.time_steps-1,:], (1,-1))
				s_print = self.dic_stim[repr(s.astype(int))]
	
				if o_print!=r_print:
					corr_ep_bool = False

			if corr_ep_bool:
				corr += 1
		
		perc = 100*float(corr)/float(N)				

		return perc
