## MAIN FILE FOR AuGMEnT TESTING
## Here are defined the settings for the AuGMEnT architecture and for the tasks to test.
## AUTHOR: Marco Martinolli
## DATE: 10.07.2017

from AuGMEnT_model_final import AuGMEnT, deep_AuGMEnT, hierarchical_AuGMEnT 				

import numpy as np
import matplotlib 
matplotlib.use('GTK3Cairo') 
from matplotlib import pyplot as plt
from matplotlib import gridspec
import pylab
import gc 

from sys import version_info

np.set_printoptions(precision=3)

task_dic ={'1':'task 12-AX',
	   '2':'saccade/anti-saccade task'}

py3 = version_info[0] > 2 # creates boolean value for test that Python major version > 2
task_selection = input("\nPlease select a task: \n\t 1: task 12-AX\n\t 2: saccade/anti-saccade task\n Enter id number:  ")
print("\nYou have selected: ", task_dic[task_selection],'\n\n')

#########################################################################################################################################
#######################   TASK 1-2 AX
#########################################################################################################################################

if (task_selection!="2"):
	
	from TASKS.task_1_2AX import data_construction				
	task = '12-AX'
	
	cues_vec = ['1','2','A','B','C','X','Y','Z']
	cues_vec_tot = ['1+','2+','A+','B+','C+','X+','Y+','Z+','1-','2-','A-','B-','C-','X-','Y-','Z-']
	pred_vec = ['L','R']

	N = 1000000
	perc_tr = 0.8
	p_c = 0.5

	np.random.seed(1)

	print('Dataset construction...')
	S_tr,O_tr,S_test,O_test,dic_stim,dic_resp = data_construction(N,p_c,perc_tr)

	reset_cond = ['1','2']	

	## CONSTRUCTION OF THE AuGMEnT NETWORK
	S = np.shape(S_tr)[1]        # dimension of the input = number of possible stimuli
	R = 3			     # dimension of the regular units
	M = 4			     # dimension of the memory units
	A = 2			     # dimension of the activity units = number of possible responses

	# value parameters were taken from the 
	lamb = 0.12    			# synaptic tag decay 
	beta = 0.08			# weight update coefficient
	discount = 0.9			# discount rate for future rewards
	alpha = 1-lamb*discount 	# synaptic permanenc
	eps = 0.05			# fraction of softmax modality for activity selection
	g = 1
	leak = 0.68

	# reward settings
	rew_system = ['RL','PL','SRL','BRL']
	rew = 'BRL'
	prop_system = ['std','BP','RBP','SRBP','MRBP']
	prop = 'RBP'

	verb = 1
	
	do_training = True
	do_test = False
	do_plots = True

	model_opt_system = ['base','deep','hier']
	model_opt = 'deep'
	spec_opt = None

	if model_opt == 'base':

		model = AuGMEnT(S,R,M,A,alpha,beta,discount,eps,g,leak,rew,dic_stim,dic_resp,prop)

	if model_opt == 'deep':

		M = 10		
		H_m = 20
		H_r = 20

		if H_m!=0:
			mem_opt = 'deep_mem'
		else:
			mem_opt = 'shallow_mem'
		if H_r!=0:
			contr_opt = 'deep_contr'
		else:
			contr_opt = 'shallow_contr'		
		
		perc_active = 1
		if perc_active!=1:
			spec_opt='spec'
		else:
			spec_opt='no_spec'			

		model = deep_AuGMEnT(S,R,M,H_r,H_m,A,alpha,beta,discount,eps,g,leak,perc_active,rew,dic_stim,dic_resp,prop)

	if model_opt == 'hier':
		L = 3
		ALPHA = [0.1,0.5,0.9]
		BETA = [0.1,0.01,0.01]
		LEAK = [0.1,0.5,0.9]

		model = hierarchical_AuGMEnT(L,S,R,M,A,ALPHA,BETA,discount,eps,g,LEAK,rew,dic_stim,dic_resp,prop)

	## TRAINING
	data_folder = 'DATA'
	image_folder = 'IMAGES'
	weight_folder = 'WEIGHT_DATA'
	if do_training:

		average_sample=10
		if model_opt=='deep' and (prop=='RBP' or prop=='SRBP' or prop=='MRBP'):
			E,conv_iter,RBP_R,RBP_M,RBP_H_R,RBP_H_M = model.training(N,S_tr,O_tr,average_sample,reset_cond,'strong',True,verb)


			np.savetxt(data_folder+'/'+model_opt+'_'+prop+'_'+spec_opt+'_'+mem_opt+'_'+contr_opt+'_'+task+'_error.txt', E)
			np.savetxt(data_folder+'/'+model_opt+'_'+prop+'_'+spec_opt+'_'+mem_opt+'_'+contr_opt+'_'+task+'_conv.txt', conv_iter)
			np.savetxt(data_folder+'/'+model_opt+'_'+prop+'_'+spec_opt+'_'+mem_opt+'_'+contr_opt+'_'+task+'_RBP_r.txt', RBP_R)
			np.savetxt(data_folder+'/'+model_opt+'_'+prop+'_'+spec_opt+'_'+mem_opt+'_'+contr_opt+'_'+task+'_RBP_m.txt', RBP_M)
			if H_r!=0:
				np.savetxt(data_folder+'/'+model_opt+'_'+prop+'_'+spec_opt+'_'+mem_opt+'_'+contr_opt+'_'+task+'_RBP_hr.txt', RBP_H_R)
			if H_m!=0:
				np.savetxt(data_folder+'/'+model_opt+'_'+prop+'_'+spec_opt+'_'+mem_opt+'_'+contr_opt+'_'+task+'_RBP_hm.txt', RBP_H_M)
		else:
			E,conv_iter = model.training(S_tr,O_tr,reset_cond,'strong',True,verb)

			np.savetxt(data_folder+'/'+model_opt+'_'+prop+'_'+mem_opt+'_'+contr_opt+'_'+task+'_error.txt', E)
			np.savetxt(data_folder+'/'+model_opt+'_'+prop+'_'+mem_opt+'_'+contr_opt+'_'+task+'_conv.txt', conv_iter)

		str_weight_mem = weight_folder+'/'+model_opt+'_'+prop+'_W_mem.txt'
		np.savetxt(str_weight_mem, model.V_m)
		str_weight_hid = weight_folder+'/'+model_opt+'_'+prop+'_W_hid.txt'
		np.savetxt(str_weight_hid, model.W_m)


		if do_plots and model_opt=='deep' and (prop=='RBP' or prop=='SRBP' or prop=='MRBP'):

			xs = np.arange(np.shape(RBP_R)[0])*average_sample

			figRBP = plt.figure(figsize=(20,25))	
			if H_r!=0:
				plt.subplot(2,2,1)
				plt.plot(xs,np.arccos(RBP_H_R)*180/np.pi,'r')
				plt.xlabel('Training Episodes')
				plt.ylabel('Angle [degrees]')
				plt.title('RBP angle condition: Hidden Regular Units')
			if H_m!=0:		
				plt.subplot(2,2,2)			
				plt.plot(xs,np.arccos(RBP_H_M)*180/np.pi,'g')
				plt.xlabel('Training Episodes')
				plt.ylabel('Angle [degrees]')
				plt.title('RBP angle condition: Hidden Memory Units')			
			plt.subplot(2,2,3)
			plt.plot(xs,np.arccos(RBP_R)*180/np.pi,'b')
			plt.xlabel('Training Episodes')
			plt.ylabel('Angle [degrees]')
			plt.title('RBP angle condition: Regular Units')
			plt.subplot(2,2,4)
			plt.plot(xs,np.arccos(RBP_M)*180/np.pi,'k')
			plt.xlabel('Training Episodes')
			plt.ylabel('Angle [degrees]')
			plt.title('RBP angle condition: Memory Units')
			plt.show()
			saveRBPcond = image_folder+'/AuG_'+model_opt+'_'+prop+'_'+task+'_RBP_cond_'+spec_opt+'.png'	
			figRBP.savefig(saveRBPcond)

	## TEST
	if do_test:
		print('TEST...\n')
		model.test(S_test,O_test,reset_cond,0)
	

	## PLOTS
	fontTitle = 26
	fontTicks = 22
	fontLabel = 22
		
	if do_plots:

		N = len(E)
		bin = 100000
		END = np.floor(N/bin).astype(int)
		E = E[:END*bin]
		N = len(E)

		E_bin = np.reshape(E,(-1,bin))
		E_bin = np.sum(E_bin,axis=1)

		figE = plt.figure(figsize=(20,8))
		N_round = np.around(N/1000).astype(int)*1000
		
		plt.subplot(1,2,1)
		plt.bar(bin*np.arange(len(E_bin)),E_bin,width=bin,color='green',edgecolor='black', alpha=0.6)
		plt.axvline(x=4492, linewidth=5, ls='dashed', color='b')
		if conv_iter!=0:		
			plt.axvline(x=conv_iter, linewidth=5, color='g')
		tit = '12AX: Training Convergence'
		plt.title(tit,fontweight="bold",fontsize=fontTitle)		
		plt.xlabel('Training Iterations',fontsize=fontLabel)
		plt.ylabel('Number of Errors per bin',fontsize=fontLabel)		
		plt.xticks(np.linspace(0,N_round,5,endpoint=True),fontsize=fontTicks)	
		plt.yticks(fontsize=fontTicks)	
		text = 'Bin = '+str(bin)
		plt.figtext(x=0.38,y=0.78,s=text,fontsize=fontLabel,bbox={'facecolor':'white', 'alpha':0.5, 'pad':10})

		plt.subplot(1,2,2)
		LOSS = 0.5*np.mean(np.reshape(E,(-1,average_sample)),axis=1)
		plt.plot(np.arange(np.shape(LOSS)[0])*average_sample,LOSS, color='green',linewidth=7, alpha=0.6)
		plt.axvline(x=4492/6, linewidth=5, ls='dashed', color='b')
		if conv_iter!=0:		
			plt.axvline(x=conv_iter/6, linewidth=5, color='g')
		tit = '12AX: Loss Function'
		plt.title(tit,fontweight="bold",fontsize=fontTitle)			
		plt.xticks(np.linspace(0,N_round,5,endpoint=True),fontsize=fontTicks)
		plt.yticks(fontsize=fontTicks)
		plt.xlabel('Training Trials',fontsize=fontLabel)
		plt.ylabel('Average Loss Function',fontsize=fontLabel)
		plt.show()

		savestr = image_folder+'/AuG_'+model_opt+'_'+prop+'_'+task+'error.png'	
		figE.savefig(savestr)


#########################################################################################################################################
#######################   TASK SACCADES/ANTI-SACCADES
#########################################################################################################################################

if (task_selection=="2"):
	
	from TASKS.task_saccades import data_construction
	task = 'saccade_no_transient'

	cues_vec = ['P','A','L','R']
	cues_vec_tot = ['P+','A+','L+','R+']
	pred_vec = ['L','F','R']

	N_trial = 20000 
	perc_tr = 0.8
	S_tr,O_tr,S_test,O_test,dic_stim,dic_resp = data_construction(N=N_trial,perc_training=perc_tr)

	reset_cond = ['empty']	

	## CONSTRUCTION OF THE AuGMEnT NETWORK
	S = np.shape(S_tr)[1]        # dimension of the input = number of possible stimuli
	R = 3			     # dimension of the regular units
	M = 4 			     # dimension of the memory units
	A = 3			     # dimension of the activity units = number of possible responses
	
	# value parameters were taken from the 
	lamb = 0.2    			# synaptic tag decay 
	beta = 0.15			# weight update coefficient
	discount = 0.9			# discount rate for future rewards
	alpha = 1-lamb*discount 	# synaptic permanence	
	eps = 0.025			# percentage of softmax modality for activity selection
	leak = 1  			# additional parameter: leaking decay of the integrative memory
	g=1

	# reward settings
	rew_system = ['RL','PL','SRL']
	rew = 'SRL'
	shape_fac = 0.2
	verb = 1
	prop_system = ['std','BP','RBP','SRBP','MRBP']
	prop = 'MRBP'
	
	do_training = True
	do_test = True
	do_plots = True				

	model_opt_system = ['base','deep','hier']
	model_opt = 'hier'

	spec_opt = None

	if model_opt == 'base':

		model = AuGMEnT(S,R,M,A,alpha,beta,discount,eps,g,leak,rew,dic_stim,dic_resp,prop)

	if model_opt == 'deep':
		
		H_m = 20
		H_r = 20
		if H_m !=0:
			mem_opt ='deep_mem'
		else:
			mem_opt ='shallow_mem'
		if H_r !=0:
			contr_opt ='deep_contr'
		else:
			contr_opt ='shallow_contr'		
		
		perc_active = 1
		if perc_active!=1:
			spec_opt='spec'
		else:
			spec_opt='no_spec'			

		model = deep_AuGMEnT(S,R,M,H_r,H_m,A,alpha,beta,discount,eps,g,leak,perc_active,rew,dic_stim,dic_resp,prop)

	if model_opt == 'hier':
		
		L = 3
		ALPHA = [0.2, 0.2, 0.2]
		BETA = [0.05,0.05,0.05]
		LEAK = [0.5,0.5,0.5]

		model = hierarchical_AuGMEnT(L,S,R,M,A,ALPHA,BETA,discount,eps,g,LEAK,rew,dic_stim,dic_resp,prop)

	## TRAINING
	data_folder = 'DATA'
	if do_training:	
		training_trial = np.round(N_trial*perc_tr).astype(int)
		E_fix,E_go,conv_iter = model.training_saccade(training_trial,S_tr,O_tr,reset_cond,verb,shape_fac)	

	## TEST
	if do_test:
		model.test_saccade(S_test,O_test,reset_cond,0)

	## PLOTS
	# plot of the memory weights
	image_folder = 'IMAGES'

	fontTitle = 26
	fontTicks = 22
	fontLabel = 22

	
	if do_plots:

		N = len(E_fix)
		bin = round(N*0.02)
		print(bin)
		E_fix_bin = np.reshape(E_fix,(-1,bin))
		E_fix_bin = np.sum(E_fix_bin,axis=1)
		E_fix_cum = np.cumsum(E_fix)
		E_fix_norm = 100*E_fix_cum/(np.arange(N)+1)
		C_fix = np.where(E_fix==0,1,0)
		C_fix_cum = 100*np.cumsum(C_fix)/(np.arange(N)+1)

		E_go_bin = np.reshape(E_go,(-1,bin))
		E_go_bin = np.sum(E_go_bin,axis=1)
		E_go_cum = np.cumsum(E_go)
		E_go_norm = 100*E_go_cum/(np.arange(N)+1)
		C_go = np.where(E_go==0,1,0)
		C_go_cum = 100*np.cumsum(C_go)/(np.arange(N)+1)

		figE = plt.figure(figsize=(22,8))
		plt.subplot(1,2,1)
		plt.bar(bin*np.arange(len(E_go_bin)),E_go_bin,width=bin,color='green',edgecolor='black',label='go',alpha=0.6)
		plt.bar(bin*np.arange(len(E_fix_bin)),E_fix_bin,width=bin,color='orange',edgecolor='black',label='fix',alpha=0.6)
		plt.axvline(x=4100, linewidth=5, ls='dashed', color='green')
		plt.axvline(x=225, linewidth=5, ls='dashed', color='orange')
		if conv_iter!=0:
			plt.axvline(x=conv_iter, linewidth=5, color='green')
		tit = 'SAS: Training Convergence'
		plt.title(tit,fontweight="bold",fontsize=fontTitle)		
		plt.xlabel('Training Trials',fontsize=fontLabel)
		plt.ylabel('Number of Errors per bin',fontsize=fontLabel)		
		plt.xticks(np.linspace(0,N,5,endpoint=True),fontsize=fontTicks)	
		plt.yticks(fontsize=fontTicks)	
		text = 'Bin = '+str(bin)
		plt.figtext(x=0.37,y=0.78,s=text,fontsize=fontLabel,bbox={'facecolor':'white', 'alpha':0.5, 'pad':10})

		plt.subplot(1,2,2)
		plt.axvline(x=225, linewidth=5, ls='dashed', color='orange')
		plt.axvline(x=4100, linewidth=5, ls='dashed', color='green')
		if conv_iter!=0:
			plt.axvline(x=conv_iter, linewidth=5, color='green')
		plt.plot(np.arange(N), E_go_cum, color='green',linewidth=7,label='go',alpha=0.6)
		plt.plot(np.arange(N), E_fix_cum, color='orange',linewidth=7,label='fix',alpha=0.6)
		tit = 'SAS: Cumulative Training Error'
		plt.title(tit,fontweight="bold",fontsize=fontTitle)			
		plt.xticks(np.linspace(0,N,5,endpoint=True),fontsize=fontTicks)
		plt.yticks(fontsize=fontTicks)
		plt.xlabel('Training Trials',fontsize=fontLabel)
		plt.ylabel('Cumulative Error',fontsize=fontLabel)
		plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,fontsize=fontTitle)
		plt.show()

		savestr = image_folder+'/'+task+'_error_'+rew+'.png'
		if M==0:
			savestr = image_folder+'/'+task+'_error_'+rew+'_nomemory.png'		
		figE.savefig(savestr)
