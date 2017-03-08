## CLASS FOR SINGLE LAYER OF HER ARCHITECTURE
## The structure of each level includes:
##	- a Working Memory (WM), which registers the stimulus according to specific gating rules 
##	- a prediction block, that provides an (error) prediction
##	- an error block, which computes the prediction error (prediction - output)
##	- the outcome block, that presents the output from feedback/error at lower level
## Version 2.0: each layer is customized to handle top-down and bottom-ups.
## AUTHOR: Marco Martinolli
## DATE: 27.02.2017

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras import backend as K
from keras.regularizers import l2
from keras.optimizers import SGD
from keras.engine.topology import Layer, Merge

import numpy as np

from keras_HER import WMBlock, PredBlock, ModBlock

def compute_response(p):

	gam = 12
	leng_resp = K.variable(np.shape(p)[1])/2

	# response u computed as (p_corr - p_wrong) for each possibility
	U = np.zeros((leng_resp,1))			
	for i,u in enumerate(U):	
		U[i] = (p[0,2*i]-p[0,2*i+1]) 

	#print('Response Vector: ',U)
	# response probability p_U obtained with softmax 			
	p_U = np.exp(gam*U)
	p_U = p_U/np.sum(p_U)	
	#print('Response Probability Vector: ', p_U)

	# response selection
	p_cum = 0
	for i,p_u in enumerate(p_U): 	
		if (np.random.random() <= (p_u + p_cum)):		
			resp_ind = i
			break
		else:		
			p_cum += p_u
	return (resp_ind)


def filtered_loss_function(y_true,y_pred):

	resp_ind = compute_response(y_pred)		
	
	a = np.zeros((np.shape(o)))			
	a[0,2*self.resp_ind] = 1
	a[0,2*selfresp_ind+1] = 1

	return K.mean((a/2)*K.square(y_true-y_pred), axis=-1)


class HER_level():
	
	## Inputs
	# ------------
	# l: int, id number of the level in the HER architecture 
	# ss: int, dimension of the input stimulus
	# M: int, number of neurons for the memory representation of the stimulus
	# P: int, number of neurons for the prediction/output vector
	# beta: real, gain parameter for the memory gating dynamics
	
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
	# P is equal to the double of the number of categories, i.e. for a binary classification 		
	#   problem(C1,C2) the prediction/ouput will have the form 
	#   [C1_corr, C1_wrong, C2_corr, C2_wrong]
	
	def __init__(self,l,S,M,P,alpha,beta,reg_value=0.01,loss_fct='mse',pred_activ_fct='linear'):
		
		self.S = S
		self.P = P
		self.M = M
		self.level = l
		self.alpha = alpha
		self.beta = beta

		self.r = None
		
		# memory_branch = S --> WM 
		# I need it to have the intermediate ouput r for the bottom-up step
		self.memory_branch = Sequential()
		self.memory_branch.add( WMBlock(output_dim=self.M, input_dim=self.S, activation='linear',W_regularizer=l2(reg_value)))
		self.memory_branch.compile(loss= loss_fct, optimizer='sgd', metrics=['accuracy'])	
		
		# prediction_branch = WM --> PredBlock
		self.prediction_branch = Sequential()
		self.prediction_branch.add(PredBlock(output_dim=self.P,input_dim=self.M,activation=pred_activ_fct,W_regularizer=l2(reg_value)))
		self.prediction_branch.compile(loss= loss_fct, optimizer=SGD(lr=self.alpha), metrics=['accuracy'])

		# combined_branch = S ---> WM --> PredBlock
		self.combined_branch = Sequential()
		self.combined_branch.add(WMBlock(output_dim=self.M, input_dim=self.S, activation='linear',W_regularizer=l2(reg_value)))
		self.combined_branch.add(Dropout(0.2))
		self.combined_branch.add(PredBlock(output_dim=self.P,activation=pred_activ_fct,W_regularizer=l2(reg_value)))
		self.combined_branch.compile(loss = loss_fct, optimizer=SGD(lr=self.alpha), metrics=['accuracy'])	
		#self.combined_branch.compile(loss = filtered_loss_function, optimizer='sgd', metrics=['accuracy'])

		# modulator block WM ---> ModBlock
		self.modulator_branch = Sequential()
		self.modulator_branch.add(ModBlock(output_dim=self.P, input_dim=self.M, activation=pred_activ_fct, W_regularizer=l2(reg_value),trainable=None))
		self.modulator_branch.compile(loss= loss_fct, optimizer='sgd', metrics=['accuracy'])
		
		# modulated prediction = prediction + modulator
		self.modulated_prediction_branch = Sequential()
		self.modulated_prediction_branch.add(Merge([self.prediction_branch, self.modulator_branch], mode='sum'))		
		self.modulated_prediction_branch.compile(loss= loss_fct, optimizer=SGD(lr=self.alpha), metrics=['accuracy'])

	def empty_memory(self):
		self.r = None
			
	def memory_gating(self,s):			
		
		if (self.r is None):
			self.r = self.memory_branch.predict(s)	
		else:
			v = self.memory_branch.predict(s)
			p_storing = np.exp(self.beta*v)/(np.exp(self.beta*v)+np.exp(self.beta*self.r))
			for i in np.arange(len(self.r)):
				if np.random.random()<=p_storing[0,i]:
					self.r[0,i] = v[0,i]                       # IS THIS CORRECT???	
		return self.r

	def compute_error(self, p,o):            ## a' ????
		return (o-p)

	def bottom_up(self, e):
		return (np.dot(np.transpose(self.r),e))

	def top_down(self, P_err):

		bias_vec = np.zeros((self.P,))
		ww = [P_err, bias_vec]	
		self.modulator_branch.set_weights(ww)

				
class HER_base(HER_level):
	
	# adding the response dynamics
	
	## Inputs
	# ------------
	# gam: real, gain parameter for response determination in softmax function
	
	## Variables
	# ------------
	# U: 1-d array Ux1, response vector	

	def __init__(self,l,S,M,P,alpha,beta,gam,reg_value=0.01,loss_fct='mse',pred_activ_fct='linear'):
		
		if l!=0:
			sys.exit("HER_base class called to level different than the first.")	
		
		super(HER_base,self).__init__(l,S,M,P,alpha,beta,reg_value,loss_fct,pred_activ_fct)
		self.gamma = gam


	def compute_error(self,p,o,resp_ind):

		a = np.zeros((np.shape(o)))			
		a[0,2*resp_ind] = 1
		a[0,2*resp_ind+1] = 1

		return a*(o-p)


	def compute_response(self, p):

		# response u computed as (p_corr - p_wrong) for each possibility
		U = np.zeros((int(np.shape(p)[1]/2),1))			
		for i,u in enumerate(U):	
			U[i] = (p[0,2*i]-p[0,2*i+1]) 

		print('Response Vector: ',U)
		# response probability p_U obtained with softmax 			
		p_U = np.exp(self.gamma*U)
		p_U = p_U/np.sum(p_U)	
		print('Response Probability Vector: ', p_U)

		# response selection
		p_cum = 0
		for i,p_u in enumerate(p_U): 	
			if (np.random.random() <= (p_u + p_cum)):		
				resp_ind = i
				break
			else:		
				p_cum += p_u
		return resp_ind



	def base_training(self,S_tr,O_tr):
	
		# online training
		for t in np.arange(np.shape(O_tr)[0]):
		
			s = S_tr[t:(t+1),:]	
			o = O_tr[t:(t+1),:]			

			self.combined_branch.fit(s,o,nb_epoch=20,batch_size=1,verbose=0)



	def base_test_binary(self,S_test,O_test,dic_stim,dic_resp,verbose=0):
		
		# testing
		N = np.shape(O_test)[0]

		Feedback_table = np.zeros((2,2))
		RESP_list = list(dic_resp.values())
			
		for t in np.arange(N):			
			
			s = S_test[t:(t+1),:]
			o = O_test[t:(t+1),:]
			o_print = dic_resp[repr(o)]

			p=self.combined_branch.predict(s)			

			resp_ind = self.compute_response(p)
			r_print = np.where(resp_ind==0, RESP_list[0], RESP_list[1])			
			
			if (verbose) :					
				print('\nTEST N.',t+1,'\t ',dic_stim[repr(s)],'\t',o_print,'\t', r_print)			
			
			
			if (o_print==RESP_list[0] and r_print==RESP_list[0]):
				Feedback_table[0,0] +=1
			elif (o_print==RESP_list[0] and r_print==RESP_list[1]):
				Feedback_table[0,1] +=1
			elif (o_print==RESP_list[1] and r_print==RESP_list[0]):
				Feedback_table[1,0] +=1
			elif (o_print==RESP_list[1] and r_print==RESP_list[1]):
				Feedback_table[1,1] +=1
		
		print('Table: \n', Feedback_table)
