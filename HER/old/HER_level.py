## CLASS FOR SINGLE LAYER OF HER ARCHITECTURE
## The structure of each level includes:
##	- a Working Memory (WM), which registers the stimulus according to specific gating rules 
##	- a prediction block, that provides an (error) prediction
##	- an error block, which computes the prediction error (prediction - output)
##	- the outcome block, that presents the output from feedback/error at lower level
## AUTHOR: Marco Martinolli
## DATE: 21.02.2017

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.regularizers import l2

import numpy as np

class HER_level():
	
	## Inputs
	# ------------
	# s: 1-d array Mx1, input stimulus
	# f: 1-d array Fx1, feedback/error signal
	# M: scalar, number of neurons for the memory representation of the stimulus
	# P: scalar, number of neurons for the prediction/output vector
	
	## Variables
	# ------------
	# X: 2-d array SxM, weight matrix for memory representation of the input stimulus
	# r: 1-d array Mx1, item representation in the WM
	# W: 2-d array MxP, weight matrix for error prediction
	# p: 1-d array Px1, prediction vector
	# o: 1-d array Px1, ouput vector
	# e: 1-d array Px1, prediction error
	# a: 1-d array Px1, activity filter based on the response


	# M is the dimension of the stimulus vector, i.e. the number of stimuli to present 
	# P is equal to the double of the number of categories, i.e. for a binary classification 		#   problem(C1,C2) the prediction/ouput will have the form 
	#   [C1_corr, C1_wrong, C2_corr, C2_wrong]
	def __init__(self,ss,M,P):
		
		self.prediction_branch = Sequential()
		
		# working memory
		self.prediction_branch.add( Dense(M, input_dim=ss, activation='relu' ))

		# prediction block 
		self.prediction_branch.add(Dropout(0.5))
		self.prediction_branch.add( Dense(P, input_dim=M, activation='relu',W_regularizer=l2(0.01)))

		self.prediction_branch.compile(loss='mse', 
					       optimizer='sgd', 
					       metrics=['accuracy'])	

				
class HER_base(HER_level):
	
	# adding the response dynamics
	
	## Inputs
	# ------------
	# gamma: scalar, gain parameter for response determination in softmax function
	
	## Variables
	# ------------
	# U: 1-d array Ux1, response vector	

	def __init__(self,ss,M,P,gam):

		super(HER_base,self).__init__(ss,M,P)
		self.gamma=gam
			


	def training(self,S_tr,O_tr):
	
		# online training
		for t in np.arange(np.shape(O_tr)[0]):
		
			s = S_tr[t:(t+1),:]	
			o = O_tr[t:(t+1),:]			

			self.prediction_branch.fit(s,o,nb_epoch=20,batch_size=1,verbose=0)



	def test(self,S_test,O_test):
		
		# testing
		N = np.shape(O_test)[0]
		err = 0
		for t in np.arange(N):			
			
			#print('\n\n\n\nITERATION ',t)
			s = S_test[t:(t+1),:]
			o = O_test[t:(t+1),:]							

			p = self.prediction_branch.predict(s)
			p = np.transpose(p)
			#print('PREDICTION: \n',p,'   Shape:',np.shape(p) )				

			# response u computed as (p_corr - p_wrong) for each possibility
			U = np.zeros((int(np.shape(p)[0]/2),1))			
			for i,u in enumerate(U):	
				U[i] = (p[2*i]-p[2*i+1]) 
			#print('Response Vector: ',U)

			# response probability p_U obtained with softmax 			
			p_U = np.exp(self.gamma*U)
			p_U = p_U/np.sum(p_U)	
			#print('Response Probability Vector: ', p_U)

			# response selection
			p_cum = 0
			for i,p_u in enumerate(p_U): 	
				if (np.random.random() <= (p_u + p_cum)):		
					resp = i
					break
				else:		
					p_cum += p_u
			
			s_print = np.where( np.array_equiv(s,[[1,0]]), '1', '2')
			o_print = np.where( np.array_equiv(o,[[1,0,0,1]]), 'L', 'R')
			r_print = np.where( resp==0, 'L', 'R' ) 			
			
			if (o_print!=r_print):
				print('\n\nError Number: ',err+1,'\n')
				print('Desired Output: ',o_print,'\n')
				print('Error response vector: ',U,'\n')
				print('Error probability vector: \n',p_U,'\n')
				err+=1	
			#print('Stimulus:  ',s_print,'  Target:  ',o_print,'   Response: ',r_print)

		print('TEST: ', err,' errors on ',N, ' samples')



			


