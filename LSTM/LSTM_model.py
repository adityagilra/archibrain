## CLASS OF LSTM ARCHITECTURE 
##
## AUTHOR: Marco Martinolli
## DATE: 11.04.2017

from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM
from keras import backend as K
from keras.regularizers import l2
from keras.optimizers import SGD, Adam, RMSprop
import numpy as np

class LSTM_arch():

	## Attribute
	# ------------
	# S - size of the input stimulus
	# H - number of hidden units
	# O - number of the output units
	# learn_rate - learning rate

	def __init__(self,S,H,O,learn_rate,batch_size=1,dt=1,dic_stim=None,dic_resp=None,task=None):

		np.random.seed(1234)		
	
		self.S = S
		self.H = H
		self.O = O

		self.dic_stim = dic_stim
		self.dic_resp = dic_resp

		self.b_sz = batch_size
		self.time_steps = dt

		self.task = task

		if task == '12-AX' or task == 'saccade' or task == 'sequence_prediction':
			# LSTM model with stateful
			self.LSTM = Sequential()
			self.LSTM.add(LSTM(batch_input_shape=(self.b_sz,self.time_steps,self.S), units=self.H, activation='sigmoid',stateful=True))
			self.LSTM.add(Dense(units=self.O, activation='softmax'))

			opt = Adam(lr=learn_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
			self.LSTM.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

		elif task == 'copy':
			# stacked LSTM model
			self.LSTM = Sequential()
			self.LSTM.add(LSTM(H[0], input_shape=(None, self.S), stateful=False, return_sequences=True, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True))
			self.LSTM.add(LSTM(H[1], stateful=False, return_sequences=True, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True))
			self.LSTM.add(LSTM(H[2], stateful=False, return_sequences=True, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True))
			self.LSTM.add((Dense(self.O)))
			self.LSTM.add(Activation('sigmoid'))

			opt = RMSprop(lr=learn_rate, rho=0.9, epsilon=1e-08, decay=0.0)
			self.LSTM.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

		elif task == 'copy_repeat':
			# stacked LSTM model
			self.LSTM = Sequential()
			self.LSTM.add(LSTM(H[0], input_shape=(None, self.S), stateful=False, return_sequences=True, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True))
			self.LSTM.add(LSTM(H[1], stateful=False, return_sequences=True, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True))
			self.LSTM.add(LSTM(H[2], stateful=False, return_sequences=True, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True))
			self.LSTM.add((Dense(self.O)))
			self.LSTM.add(Activation('sigmoid'))

			opt = RMSprop(lr=learn_rate, rho=0.9, epsilon=1e-08, decay=0.0)
			self.LSTM.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])


	def training(self,S_train,O_train,ep=3,max_iters=100):

		#self.LSTM.fit(S_train, O_train, epochs=ep, batch_size=self.b_sz, shuffle=False)	
		task = self.task
		if task == '12-AX' or task == 'saccade' or task == 'sequence_prediction':

			N = np.shape(S_train)[0]
			convergence = False
			E = np.zeros((N))
			conv_iter = np.array([0])

			if task=='12AX':
				correct = 0
				conv2 = False
				conv_iter_2 = np.array([0]) 

			if task=='saccade':

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
				
			for e in np.arange(ep):

				self.LSTM.reset_states()

				for n in np.arange(N):

					s = S_train[n:(n+1),:,:]				
					o = O_train[n:(n+1),:]

					self.LSTM.fit(s, o, epochs=1, batch_size=self.b_sz, shuffle=False)

					r = self.LSTM.predict(s,batch_size=self.b_sz)
					resp_ind = np.argmax(r)

					o_print = self.dic_resp[repr(o.astype(int))]
					r_print = self.dic_resp[repr(resp_ind)]	

					if task=='12AX':
		
						s = np.reshape(s[0,self.time_steps-1,:], (1,-1))
						s_print = self.dic_stim[repr(s.astype(int))]

						if r_print==o_print:
							correct += 1
						else:
							correct = 0
							E[n] += 1
						
						if conv2==False and correct==2*25*6:
							conv2 = True
							conv_iter_2 = np.array([n]) 

						if correct==1000 and convergence==False:
							convergence = True
							conv_iter = np.array([n])
							break
					
					if task=='saccade':

						s_fix = np.reshape(s[0,1,:], (1,-1))
						s_fix = self.dic_stim[repr(s_fix.astype(int))]

						s_cue = np.reshape(s[0,2,:], (1,-1))
						s_cue = self.dic_stim[repr(s_cue.astype(int))]

						if s_fix=='P' and s_cue=='L':
							print('PL')
							num_PL += 1
							if r_print == o_print:
								trial_PL[(num_PL-1) % 50] = 1
							else:
								trial_PL[(num_PL-1) % 50] = 0
								E[n] = 1
							prop_PL = np.mean(trial_PL)

						if s_fix=='P' and s_cue=='R':
							print('PR')
							num_PR += 1
							if r_print == o_print:
								trial_PR[(num_PR-1) % 50] = 1
							else:
								trial_PR[(num_PR-1) % 50] = 0
								E[n] = 1
							prop_PR = np.mean(trial_PR)

						if s_fix=='A' and s_cue=='L':
							print('AL')
							num_AL += 1
							if r_print == o_print:
								trial_AL[(num_AL-1) % 50] = 1
							else:
								trial_AL[(num_AL-1) % 50] = 0
								E[n] = 1
							prop_AL = np.mean(trial_AL)

						if s_fix=='A' and s_cue=='R':
							print('AR')
							num_AR += 1
							if r_print == o_print:
								trial_AR[(num_AR-1) % 50] = 1	
							else:
								trial_AR[(num_AR-1) % 50] = 0
								E[n] = 1
							prop_AR = np.mean(trial_AR)

						if convergence==False and prop_PL>=0.9 and prop_PR>=0.9 and prop_AL>=0.9 and prop_AR>=0.9:
							convergence = True
							conv_iter = np.array([n])

		elif task == 'copy' or task == 'copy_repeat':
			acc = []
			loss = []
			iters = []

			try:
				for i in range(max_iters):
					x = S_train[i*self.b_sz : (i+1)*self.b_sz]
					y = O_train[i*self.b_sz : (i+1)*self.b_sz]
					scores = self.LSTM.train_on_batch(x, y)

					if (i+1)%100 == 0:
						loss.append(scores[0])
						acc.append(scores[1])
						iters.append(i)
						print('Training Iteration ', str(i), ': Loss = ', str(scores[0]), '\t Accuracy = ', str(scores[1]))

			except KeyboardInterrupt:
				pass

			return acc, loss, iters

		if task=='12AX':

			if conv2==True:
				print('SIMULATION MET (LENIENT) CRITERION AT ITERATION ',conv_iter_2)
			if convergence==True:
				print('SIMULATION MET (HARD) CRITERION AT ITERATION ',conv_iter)
		
			return E, conv_iter, conv_iter_2

		if task=='saccade':

			if convergence==True:
				print('SIMULATION MET CRITERION AT ITERATION ',conv_iter)

			return E, conv_iter
								


									

	def test(self,S_test,O_test,verbose=1,max_iters=100):

		task = self.task
		if task == '12-AX' or task == 'saccade' or task == 'sequence_prediction':

			N = np.shape(S_test)[0]
			binary = (self.O==2)
			
			Feedback_table = np.zeros((2,2))

			RESP_list = list(self.dic_resp.values())
			RESP_list = np.unique(RESP_list)
			RESP_list.sort()

			#scores = LSTM.evaluate(S_test,O_test,batch_size=self.b_sz)
			#print('Keras Evaluation:\n', scores)

			R = self.LSTM.predict(S_test,batch_size=self.b_sz)

			corr = 0

			for n in np.arange(N):
				
				s = S_test[n:(n+1),self.time_steps-1,:]
				r = R[n]
				s = np.reshape(s,(1,-1))
				o = np.reshape(O_test[n:(n+1),:],(1,-1))
				resp_ind = np.argmax(r)
				s_print = self.dic_stim[repr(s.astype(int))]
				o_print = self.dic_resp[repr(o.astype(int))]
				r_print = self.dic_resp[repr(resp_ind)]

				if (o_print == r_print):
					corr += 1
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
			
			print("Percentage of correct predictions: ", 100*corr/N)

		elif task == 'copy' or task == 'copy_repeat':

			total_acc = 0
			total_loss = 0

			try:
				for i in range(max_iters):
					x = S_test[i*self.b_sz : (i+1)*self.b_sz]
					y = O_test[i*self.b_sz : (i+1)*self.b_sz]
					scores = self.LSTM.evaluate(x, y, verbose=0)
					total_loss += scores[0]
					total_acc += scores[1]

					if (i+1)%10 == 0:
						print('Test Iteration ', str(i), ': Loss = ', str(scores[0]), '\t Accuracy = ', str(scores[1]))

			except KeyboardInterrupt:
				pass

			return float(total_loss)/float(i), float(total_acc)/float(i)

			