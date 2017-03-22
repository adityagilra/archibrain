## CLASS FOR BASE LAYER OF HER ARCHITECTURE
##
## The base level inherits the general structure from the standard HER_level class but includes also the response dynamics of the system.
## Response rules are taken rom the Supplementary Material of the paper "Frontal cortex function derives from hierarchical predictive coding", W. Alexander, J. Brown, equations (9) and (10).
##
## AUTHOR: Marco Martinolli
## DATE: 10.03.2017


from HER_level import HER_level
import numpy as np
				
class HER_base(HER_level):
	
	## Inputs
	# ------------
	# gam: real, gain parameter for response determination in softmax function
	
	## Variables
	# ------------
	# U: 1-d array Ux1, response vector	

	def __init__(self,l,S,P,alpha,beta,gam,elig_decay_const,reg_value=0.01,loss_fct='mse',mem_activ_fct='linear',pred_activ_fct='linear'):
		
		if l!=0:
			sys.exit("HER_base class called to level different than the first.")	
		
		super(HER_base,self).__init__(l,S,P,alpha,beta,elig_decay_const,reg_value,loss_fct,mem_activ_fct,pred_activ_fct)
		self.gamma = gam


	def compute_error(self,p,o,resp_ind):

		a = np.zeros((np.shape(o)))			
		a[0,(2*resp_ind):(2*resp_ind+2)] = 1

		return a*(o-p)


	def compute_response(self, p):

		# response u computed as (p_corr - p_wrong) for each possibility
		U = np.zeros((int(np.shape(p)[1]/2),1))			
		for i,u in enumerate(U):	
			U[i] = (p[0,2*i]-p[0,2*i+1]) 

		#print('Response Vector: ',np.transpose(U))
		# response probability p_U obtained with softmax 			
		p_U = np.exp(self.gamma*U)
		p_U = p_U/np.sum(p_U)	
		#print('Response Probability Vector: ', np.transpose(p_U))

		# response selection
		p_cum = 0
		random_value = np.random.random()
		for i,p_u in enumerate(p_U): 	
			if (random_value <= (p_u + p_cum)):		
				resp_ind = i
				break
			else:		
				p_cum += p_u
		return resp_ind



	def base_training(self,S_train,O_train,dic_stim,dic_resp):		
		
		N_samples = np.shape(S_train)[0]
		d = np.zeros((1,np.shape(S_train)[1]))	
		delete = ''

		for i in np.arange(N_samples):
			print('\n-----------------------------------------------------------------------\n')
			s = S_train[i:(i+1),:]		
			o = O_train[i:(i+1),:]				
		
			# eligibility trace dynamics
			d = d*self.elig_decay_const
			d[0,(np.where(s==1))[1]] = 1
			print('Eligibility Trace: ',d)

			# memory gating dynamics
			r = self.memory_gating(s,gate='max')
			
			if dic_stim is not None:			
				print('TRAINING ITER:',i,'   s: ',dic_stim[repr(s.astype(int))],'   r0:', dic_stim[repr(r.astype(int))])
						
			p = self.prediction_branch.predict(r)

			resp_i = self.compute_response(p)
			e = self.compute_error(p, o, resp_i)


			### TRAINING OF PREDICTION WEIGHTS
			W_p = self.prediction_branch.get_weights()
			W_p[0] = W_p[0] + self.alpha* np.dot(np.transpose(r), e)
			self.prediction_branch.set_weights(W_p)					
			W_m = self.memory_branch.get_weights()			
			W_m[0] = W_m[0] + np.dot(np.dot(W_p[0], np.transpose(e))*np.transpose(r)  ,d)
			self.memory_branch.set_weights(W_m)		

			print('Output: ', dic_resp[repr(o)]  ,'   Response: ', dic_resp[repr(resp_i)])
			print('Prediction: ', p)
			print('Prediction Error: ', e)
			print('Memory Matrix:\n', W_m[0])
			print('Prediction Matrix:\n', W_p[0])



	def base_test(self,S_test,O_test,dic_stim,dic_resp,verbose=0):
		
		# testing
		N = np.shape(O_test)[0]

		Feedback_table = np.zeros((2,2))
		RESP_list = list(dic_resp.values())
		RESP_list.sort()
		#print(RESP_list)

		binary = (len(RESP_list)/2==2)
		
		for t in np.arange(N):			
			
			s = S_test[t:(t+1),:]
			o = O_test[t:(t+1),:]
			o_print = dic_resp[repr(o)]

			p=self.prediction_branch.predict(s)			

			resp_ind = self.compute_response(p)
			
			r_print = np.where(resp_ind==0, RESP_list[0], RESP_list[2])			
			
			if (verbose) :					
				print('\nTEST N.',t,'\t ',dic_stim[repr(s.astype(int))],'\t',o_print,'\t', r_print)			
			
			if (binary):
				if (o_print==RESP_list[0] and r_print==RESP_list[0]):
					Feedback_table[0,0] +=1
				elif (o_print==RESP_list[0] and r_print==RESP_list[2]):
					Feedback_table[0,1] +=1
				elif (o_print==RESP_list[2] and r_print==RESP_list[0]):
					Feedback_table[1,0] +=1
				elif (o_print==RESP_list[2] and r_print==RESP_list[2]):
					Feedback_table[1,1] +=1
		
		if (binary):	
			print('Table: \n', Feedback_table)
			print('Percentage of correct predictions: ', 100*(Feedback_table[0,0]+Feedback_table[1,1])/np.sum(Feedback_table),'%')

