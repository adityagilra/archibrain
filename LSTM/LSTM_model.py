## CLASS OF LSTM ARCHITECTURE 
##
## AUTHOR: Marco Martinolli
## DATE: 11.04.2017

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, LSTM
from keras import backend as K
from keras.regularizers import l2
from keras.optimizers import SGD
from keras.engine.topology import Layer, Merge
import numpy as np

class LSTM_arch():

	## Attribute
	# ------------
	# S - size of the input stimulus
	# H - number of hidden units
	# O - number of the output units
	# learn_rate - learning rate

	def __init__(self,S,H,O,learn_rate,dic_stim=None,dic_resp=None):

		self.S = S
		self.H = H
		self.O = O

		self.dic_stim = dic_stim
		self.dic_resp = dic_resp

		# LSTM model
		self.model = Sequential()
		self.model.add(LSTM(output_dim=self.H, input_dim=self.S, activation='sigmoid'))
		self.model.add(Dense(output_dim=self.O, init='zero', activation='sigmoid'))
		self.model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=learn_rate, momentum=0.0, decay=0.0, nesterov=False), metrics=['accuracy'])


	def training(self,S_train,O_train,reset_cond=None,verbose=1):

		N = np.shape(S_train)[0]
		
		for n in np.arange(N):
			s = S_train[n:(n+1),:]
			o = O_train[n:(n+1),:]
			
			s_print = self.dic_stim[repr(np.reshape(s,(1,-1)).astype(int))]
			o_print = self.dic_resp[repr(np.reshape(o,(1,-1)).astype(int))]
			#if s_print in reset_cond:				
			#	print('RESET')					# does LSTM need of manual RESETTING?
			#	self.model.reset_states()
			if verbose:
				print('ITER ',n,'\t S: ',s_print,'O: ',o_print)

			self.model.fit(s,o,nb_epoch=1)				
		

	def test(self,S_test,O_test,reset_cond=None,verbose=1):

		N = np.shape(S_test)[0]
		binary = (self.O==2)
		R = self.model.predict(S_test)
		
		Feedback_table = np.zeros((2,2))

		RESP_list = list(self.dic_resp.values())
		RESP_list = np.unique(RESP_list)
		RESP_list.sort()

		for n in np.arange(N):

			s = np.reshape(S_test[n:(n+1),:],(1,-1))
			o = np.reshape(O_test[n:(n+1),:],(1,-1))
			r = R[n,:]
			resp_ind = np.argmax(r)
			s_print = self.dic_stim[repr(s.astype(int))]
			o_print = self.dic_resp[repr(o.astype(int))]
			r_print = self.dic_resp[repr(resp_ind)]

			if (binary):

				if (verbose):
					print('TEST SAMPLE N.',n+1,'\t',s_print,'\t',o_print,'\t',r_print,'\n')

				if (o_print==RESP_list[0] and r_print==RESP_list[0]):
					Feedback_table[0,0] += 1
				elif (o_print==RESP_list[0] and r_print==RESP_list[1]):
					Feedback_table[0,1] += 1
				elif (o_print==RESP_list[1] and r_print==RESP_list[0]):
					Feedback_table[1,0] += 1
				elif (o_print==RESP_list[1] and r_print==RESP_list[1]):
					Feedback_table[1,1] += 1

		if binary:
			print("PERFORMANCE TABLE (output vs. response):\n",Feedback_table)
		
		corr = Feedback_table[0,0] + Feedback_table[1,1]
		print("Percentage of correct predictions: ", 100*corr/N) 			



	def training_saccade(self,S_train,O_train,reset_case,verbose=False):

		N_stimuli = np.shape(S_train)[0]
		zero = np.zeros((1,self.S))
		s_old = zero

		phase = 'start'
		fix = 0
		delay = 0
		r = None 
		abort = False
		resp = False
	
		for n in np.arange(N_stimuli):	
			
			if abort==True and jump!=0:
				jump-=1
				if jump==0:
					abort = False
			else:
				s = S_train[n:(n+1),:]
				o = O_train[n:(n+1),:]
				s_print = self.dic_stim[repr(s.astype(int))]
				o_print = self.dic_resp[repr(o.astype(int))]
				s_inst = s
				s_trans = self.define_transient(s_inst, s_old)
				s_old = s_inst
							
				if s_print=='empty' and phase!='delay':			# empty screen, begin of the trial
					phase = 'start'
				elif  phase=='start' and (s_print=='P' or s_print=='A'):   	# fixation mark appers
					phase = 'fix'	
					num_fix = 0
					attempts = 0
				elif (s_print=='AL' or s_print=='AR' or s_print=='PL' or s_print=='PR'): 	# location cue
					phase = 'cue'
				elif (s_print=='P' or s_print=='A') and phase=='cue':	   	# memory delay
					phase = 'delay'
				elif s_print=='empty' and phase=='delay':	 # go = solve task
					phase = 'go'
					num_attempts = 1
					resp = False

				if o_print!='None':			
					r_print = self.dic_resp[repr(resp_ind)]
					print('ITER: ',n+1,'-',phase,'\t STIM: ',s_print,'\t OUT: ',o_print,'\t RESP: ', r_print,'\t\t Q: ',Q)
				else:
					r_print = 'None'
					print('ITER: ',n+1,'-',phase,'\t STIM: ',s_print)
				
				
				if phase=='fix':
					attempts+=1
					if o_print!=r_print:
						num_fix = 0    # no fixation	
					else: 
						num_fix+=1     # fixation		

				if phase=='go':
					if o_print!=r_print:
						if r_print!='F':
							resp = True		
					else: 
						resp = True				

				if phase=='fix' and num_fix<2 and attempts==2:
					while num_fix<2 and attempts<10:
						fix = self.try_again(s,o_print,phase)
						attempts += 1
						if fix==False:
							num_fix = 0
						else:
							num_fix += 1
					if attempts==10 and r!=self.rew_pos:
						print('No fixation. ABORT')
						abort = True
						jump = 4  

				if phase=='go' and resp==False:
					attempts = 1
					while resp==False and attempts<8:
						resp = self.try_again(s,o_print,phase)
						attempts += 1
				if resp==True:
					resp = self.try_again(s,o_print,phase)
						
