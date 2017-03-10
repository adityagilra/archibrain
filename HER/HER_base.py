## CLASS FOR BASE LAYER OF HER ARCHITECTURE
## The base level inherits the general structure from the standard HER_level class but includes also the response dynamics of the system.
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

	def __init__(self,l,S,M,P,alpha,beta,gam,reg_value=0.01,loss_fct='mse',pred_activ_fct='linear',drop_perc=0.3):
		
		if l!=0:
			sys.exit("HER_base class called to level different than the first.")	
		
		super(HER_base,self).__init__(l,S,M,P,alpha,beta,reg_value,loss_fct,pred_activ_fct,drop_perc)
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



	def base_test(self,S_test,O_test,dic_stim,dic_resp,verbose=0):
		
		# testing
		N = np.shape(O_test)[0]

		Feedback_table = np.zeros((2,2))
		RESP_list = list(dic_resp.values())
		RESP_list.sort()
		#print(RESP_list)

		binary = (len(RESP_list)==2)
		
		for t in np.arange(N):			
			
			s = S_test[t:(t+1),:]
			o = O_test[t:(t+1),:]
			o_print = dic_resp[repr(o)]

			p=self.combined_branch.predict(s)			

			resp_ind = self.compute_response(p)
			
			r_print = np.where(resp_ind==0, RESP_list[0], RESP_list[1])			
			
			if (verbose) :					
				print('\nTEST N.',t+1,'\t ',dic_stim[repr(s)],'\t',o_print,'\t', r_print)			
			
			if (binary):
				if (o_print==RESP_list[0] and r_print==RESP_list[0]):
					Feedback_table[0,0] +=1
				elif (o_print==RESP_list[0] and r_print==RESP_list[1]):
					Feedback_table[0,1] +=1
				elif (o_print==RESP_list[1] and r_print==RESP_list[0]):
					Feedback_table[1,0] +=1
				elif (o_print==RESP_list[1] and r_print==RESP_list[1]):
					Feedback_table[1,1] +=1
		
		if (binary):	
			print('Table: \n', Feedback_table)
			print('Percentage of correct predictions: ', 100*(Feedback_table[0,0]+Feedback_table[1,1])/np.sum(Feedback_table),'%')

