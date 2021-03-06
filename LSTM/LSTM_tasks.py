## MAIN FILE FOR LSTM TESTING
## Here are defined the settings for the AuGMEnT architecture and for the tasks to test.
## AUTHOR: Marco Martinolli
## DATE: 11.04.2017

import tensorflow as tf
import numpy as np
import matplotlib 
matplotlib.use('GTK3Cairo') 
from matplotlib import pyplot as plt
from matplotlib import gridspec
import pylab
import gc 
from keras.models import load_model
import h5py

import sys
sys.path.append("..")
sys.path.append("LSTM")

from LSTM_model import LSTM_arch


def run_task(task, params_bool=None, params_task=None):
	if(task == '3'):
		LSTM_task_1_2AX(params_bool, params_task)

	elif(task == '4'):
		LSTM_task_saccades(params_bool, params_task)

	elif(task == '5'):
		LSTM_task_seq_prediction(params_bool, params_task)

	elif(task == '6'):
		LSTM_task_copy(params_bool, params_task)

	elif(task == '7'):
		LSTM_task_copy_repeat(params_bool, params_task)
		
	else:
		print('The task is not valid for LSTM\n')


#########################################################################################################################################
#######################   TASK 1-2 AX
#########################################################################################################################################

def LSTM_task_1_2AX(params_bool, params_task):

	from TASKS.task_1_2AX import data_construction, data_modification_for_LSTM
	task = '12-AX'
	
	np.random.seed(1234)	

	cues_vec = ['1','2','A','B','X','Y']
	pred_vec = ['L','R']

	print('Dataset construction...')

	if params_task is None:
		N = 20000
		p_c = 0.5
		perc_tr = 0.8
	else:
		N = int(params_task[0]) if params_task[0] != '' else 20000
		p_c = float(params_task[1]) if params_task[1] != '' else 0.5
		perc_tr = float(params_task[2]) if params_task[2] != '' else 0.8

	S_tr,O_tr,S_tst,O_tst,dic_stim,dic_resp = data_construction(N,p_c,perc_tr,model='2')
	
	print('Dataset construction done!')

	dt = 10	
	S_train_3D,O_train = data_modification_for_LSTM(S_tr,O_tr,dt)
	print('Training dataset modification done!')

	S_test_3D,O_test = data_modification_for_LSTM(S_tst,O_tst,dt)
	print('Test dataset modification done!')

	## CONSTRUCTION OF THE LSTM NETWORK
	S = np.shape(S_tr)[1]        # dimension of the input = number of possible stimuli
	H = 10	 		     # number of the hidden units
	O = np.shape(O_tr)[1]				     # dimension of the activity units = number of possible responses
	
	# value parameters were taken from the 
	alpha = 0.001			# learning rate
	hysteresis = 0.5	# hysteresis coefficient ???  s.t. context layer update: c(t) = 0.5 h(t-1) + 0.5 c(t-1)
	toll_err = 0.1		# tolerance error ???

	b_sz = 1
	
	verb = 0
	
	if params_bool is None:
		do_training = True
		do_test = True
		do_weight_plots = True
		do_error_plots = True
		
	else:
		do_training = params_bool[0]
		do_test = params_bool[1]
		do_weight_plots = params_bool[2]	
		do_error_plots = params_bool[3]	

	model = LSTM_arch(S,H,O,alpha,b_sz,dt,dic_stim,dic_resp,task)
	
	## TRAINING
	folder = 'LSTM/DATA'
	N_trial=N
	if do_training:	

		print('TRAINING...\n')	

		E,conv_iter,conv_iter_2 = model.training(S_train_3D,O_train,ep=1)
		
		str_err = folder+'/'+task+'_error_units'+str(H)+'.txt'
		np.savetxt(str_err,E)
		str_conv = folder+'/'+task+'_conv_units'+str(H)+'_.txt'
		np.savetxt(str_conv,conv_iter)
		str_conv_2 = folder+'/'+task+'_conv_2_units'+str(H)+'.txt'
		np.savetxt(str_conv_2,conv_iter_2)
			
		str_save = folder+'/'+task+'_weights_units'+str(H)+'.h5'
		model.LSTM.save_weights(str_save)

		print('\nSaved model to disk.')

	else:

		str_err = folder+'/'+task+'_error_units'+str(H)+'.txt'
		E = np.loadtxt(str_err)
		str_conv = folder+'/'+task+'_conv_units'+str(H)+'.txt'
		conv_iter = np.loadtxt(str_conv)
		str_conv_2 = folder+'/'+task+'_conv_2_units'+str(H)+'.txt'
		conv_iter_2 = np.loadtxt(str_conv_2)

		str_load = folder+'/'+task+'_weights_'+str(N_trial)+'_2.h5'
		model.LSTM.load_weights(str_load)
		
		# print(np.shape(model.LSTM.get_weights()[0]))
		# print(np.shape(model.LSTM.get_weights()[1]))
		# print(np.shape(model.LSTM.get_weights()[2]))
		# print(np.shape(model.LSTM.get_weights()[3]))
		# print(np.shape(model.LSTM.get_weights()[4]))

		print('\nLoaded weights from disk.')			

	print('\n------------------------------------------------------------------------------------------			\n----------------------------------------------------------------------------------------------\n')

	## TEST
	if do_test:
		print('TEST...\n')
		model.test(S_test_3D,O_test)
	
	print(conv_iter)
	print(conv_iter_2)

	## PLOTS
	# plot of the memory weights
	folder = 'LSTM/IMAGES'
	fontTitle = 26
	fontTicks = 22
	fontLabel = 22

	if do_error_plots:

		N = len(E)
		bin = np.around(N*0.02).astype(int)
		END = np.floor(N/bin).astype(int)
		E = E[:END*bin]
		N = len(E)

		N_round = np.around(N/1000).astype(int)*1000

		E_bin = np.reshape(E,(-1,bin))
		E_bin = np.sum(E_bin,axis=1)
		E_cum = np.cumsum(E)
		E_norm = 100*E_cum/(np.arange(N)+1)
		C = np.where(E==0,1,0)
		C_cum = 100*np.cumsum(C)/(np.arange(N)+1)

		fig= plt.figure(figsize=(20,8))
		plt.subplot(1,2,1)
		plt.bar(bin*np.arange(len(E_bin))/6,E_bin,width=bin/6,color='red',edgecolor='black', alpha=0.6)
		plt.axvline(x=2902/6, linewidth=5, ls='dashed', color='b')
		plt.axvline(x=48000/6, linewidth=5, ls='dashed', color='r')
		plt.axvline(x=conv_iter_2/6, linewidth=5, color='r')
		tit = '12AX: Training Convergence'
		plt.title(tit,fontweight="bold",fontsize=fontTitle)		
		plt.xlabel('Training Trials',fontsize=fontLabel)
		plt.ylabel('Number of Errors per bin',fontsize=fontLabel)		
		plt.xticks(np.linspace(0,N_round/6,5,endpoint=True),fontsize=fontTicks)	
		plt.yticks(fontsize=fontTicks)	
		text = 'Bin = '+str(np.around(bin/6).astype(int))
		plt.figtext(x=0.38,y=0.78,s=text,fontsize=fontLabel,bbox={'facecolor':'white', 'alpha':0.5, 'pad':10})

		plt.subplot(1,2,2)
		plt.plot(np.arange(N)/6, E_cum, color='red',linewidth=7, alpha=0.6)
		plt.axvline(x=2902/6, linewidth=5, ls='dashed', color='b')
		plt.axvline(x=48000/6, linewidth=5, ls='dashed', color='r')
		plt.axvline(x=conv_iter_2/6, linewidth=5, color='r')
		tit = '12AX: Cumulative Training Error'
		plt.title(tit,fontweight="bold",fontsize=fontTitle)			
		plt.xticks(np.linspace(0,N_round/6,5,endpoint=True),fontsize=fontTicks)
		plt.yticks(fontsize=fontTicks)
		plt.xlabel('Training Trials',fontsize=fontLabel)
		plt.ylabel('Cumulative Error',fontsize=fontLabel)
		plt.show()
		savestr = folder+'/'+task+'_error_units'+str(H)+'.png'
		fig.savefig(savestr)



#########################################################################################################################################
#######################   TASK SACCADES/ANTI-SACCADES
#########################################################################################################################################

def LSTM_task_saccades(params_bool, params_task):
	
	from TASKS.task_saccades import data_construction, data_modification_for_LSTM
	task = 'saccade'

	cues_vec = ['1','2','A','B','X','Y']
	pred_vec = ['L','R']
	
	np.random.seed(1234)

	
	print('Dataset construction...')

	if params_task is None:
		N_trial = 20000 
		perc_tr = 0.8
	else:
		N_trial = int(params_task[0]) if params_task[0] != '' else 20000
		perc_tr = float(params_task[1]) if params_task[1] != '' else 0.8

	S_tr,O_tr,S_tst,O_tst,dic_stim,dic_resp = data_construction(N=N_trial,perc_training=perc_tr,model='2')

	dt = 6 # 6 phases: start,fix,cue,delay,delay,gp
	S_train_3D,O_train = data_modification_for_LSTM(S_tr,O_tr,dt)
	S_test_3D,O_test = data_modification_for_LSTM(S_tst,O_tst,dt)

	## CONSTRUCTION OF THE LSTM NETWORK
	S = np.shape(S_tr)[1]        # dimension of the input = number of possible stimuli
	H = 4 		     # number of the hidden units
	O = np.shape(O_tr)[1]			     # dimension of the activity units = number of possible responses
	
	# value parameters were taken from the 
	alpha = 0.1		# learning rate
	hysteresis = 0.5	# hysteresis coefficient ???  s.t. context layer update: c(t) = 0.5 h(t-1) + 0.5 c(t-1)
	toll_err = 0.1		# tolerance error ???
	b_sz  = 1

	verb = 1
	
	if params_bool is None:
		do_training = True
		do_test = True
		do_weight_plots = True
		do_error_plots = True
		
	else:
		do_training = params_bool[0]
		do_test = params_bool[1]
		do_weight_plots = params_bool[2]	
		do_error_plots = params_bool[3]		

	model = LSTM_arch(S,H,O,alpha,b_sz,dt,dic_stim,dic_resp,task)

	## TRAINING
	folder = 'LSTM/DATA'
	if do_training:	

		print('TRAINING...\n')	

		E,conv_iter = model.training(S_train_3D,O_train,ep=1)
		
		str_err = folder+'/'+task+'_error.txt'
		np.savetxt(str_err,E)
		str_conv = folder+'/'+task+'_conv.txt'
		np.savetxt(str_conv,conv_iter)
		print(conv_iter)
	
		str_save = folder+'/'+task+'_weights.h5'
		model.LSTM.save_weights(str_save)

		print('\nSaved model to disk.')

	else:

		str_err = folder+'/'+task+'_error.txt'
		E = np.loadtxt(str_err)
		str_conv = folder+'/'+task+'_conv.txt'
		conv_iter = np.loadtxt(str_conv)
		print(conv_iter)
		str_load = folder+'/'+task+'_weights.h5'
		model.LSTM.load_weights(str_load)

		print('\nLoaded weights from disk.')			

	print('\n------------------------------------------------------------------------------------------			\n----------------------------------------------------------------------------------------------\n')

	## TEST
	if do_test:
		print('TEST...\n')
		model.test(S_test_3D,O_test,verbose=verb)

	## PLOTS
	# plot of the memory weights
	folder = 'LSTM/IMAGES'
	
	fontTitle = 26
	fontTicks = 22
	fontLabel = 22

	if do_error_plots:

		N = len(E)
		bin = round(N*0.02)
		E_bin = np.reshape(E,(-1,bin))
		E_bin = np.sum(E_bin,axis=1)
		E_cum = np.cumsum(E)
		E_norm = 100*E_cum/(np.arange(N)+1)
		C = np.where(E==0,1,0)
		C_cum = 100*np.cumsum(C)/(np.arange(N)+1)

		figE = plt.figure(figsize=(20,8))
		plt.subplot(1,2,1)
		plt.bar(bin*np.arange(len(E_bin)),E_bin,width=bin,color='red',edgecolor='black', alpha=0.6)
		plt.axvline(x=conv_iter, linewidth=5, color='r')
		plt.axvline(x=4100, linewidth=5, ls='dashed', color='g')
		plt.axvline(x=1837, linewidth=5, color='g')
		tit = 'SAS: Training Convergence'
		plt.title(tit,fontweight="bold",fontsize=fontTitle)		
		plt.xlabel('Training Trials',fontsize=fontLabel)
		plt.ylabel('Number of Errors per bin',fontsize=fontLabel)		
		plt.xticks(np.linspace(0,N,5,endpoint=True),fontsize=fontTicks)	
		plt.yticks(fontsize=fontTicks)	
		text = 'Bin = '+str(bin)
		plt.figtext(x=0.38,y=0.78,s=text,fontsize=fontLabel,bbox={'facecolor':'white', 'alpha':0.5, 'pad':10})

		plt.subplot(1,2,2)
		plt.plot(np.arange(N), E_cum, color='red',linewidth=7, alpha=0.6)
		plt.axvline(x=conv_iter, linewidth=5, color='r')
		plt.axvline(x=4100, linewidth=5, ls='dashed', color='g')
		plt.axvline(x=1837, linewidth=5, color='g')
		tit = 'SAS: Cumulative Training Error'
		plt.title(tit,fontweight="bold",fontsize=fontTitle)			
		plt.xticks(np.linspace(0,N,5,endpoint=True),fontsize=fontTicks)
		plt.yticks(fontsize=fontTicks)
		plt.xlabel('Training Trials',fontsize=fontLabel)
		plt.ylabel('Cumulative Error',fontsize=fontLabel)
		plt.show()
		savestr = folder+'/'+task+'_error_2.png'
		figE.savefig(savestr)


#########################################################################################################################################
#######################   TASK SEQUENCE PREDICTION
#########################################################################################################################################

def LSTM_task_seq_prediction(params_bool, params_task):

	from TASKS.task_seq_prediction import data_construction, data_modification_for_LSTM
	task = 'sequence_prediction'
	
	np.random.seed(1234)	

	cues_vec = ['A','B','C','X']
	pred_vec = ['D','B','C','Y']

	print('Dataset construction...')

	if params_task is None:
		N = 20000
		p_c = 0.5
		perc_tr = 0.8
	else:
		N = int(params_task[0]) if params_task[0] != '' else 20000
		p_c = float(params_task[1]) if params_task[1] != '' else 0.5
		perc_tr = float(params_task[2]) if params_task[2] != '' else 0.8

	S_tr,O_tr,S_tst,O_tst,dic_stim,dic_resp = data_construction(N,p_c,perc_tr,model='2')
	
	print('Dataset construction done!')

	dt = 3	
	S_train_3D,O_train = data_modification_for_LSTM(S_tr,O_tr,dt)
	print('Training dataset modification done!')

	S_test_3D,O_test = data_modification_for_LSTM(S_tst,O_tst,dt)
	print('Test dataset modification done!')

	## CONSTRUCTION OF THE LSTM NETWORK
	S = np.shape(S_tr)[1]        # dimension of the input = number of possible stimuli
	H = 20	 		     # number of the hidden units
	O = np.shape(O_tr)[1]			     # dimension of the activity units = number of possible responses
	
	# value parameters were taken from the 
	alpha = 0.002		# learning rate
	hysteresis = 0.5	# hysteresis coefficient ???  s.t. context layer update: c(t) = 0.5 h(t-1) + 0.5 c(t-1)
	toll_err = 0.1		# tolerance error ???

	b_sz = 1
	
	verb = 0
	
	if params_bool is None:
		do_training = True
		do_test = True
		do_weight_plots = True
		do_error_plots = True
		
	else:
		do_training = params_bool[0]
		do_test = params_bool[1]
		do_weight_plots = params_bool[2]	
		do_error_plots = params_bool[3]	

	model = LSTM_arch(S,H,O,alpha,b_sz,dt,dic_stim,dic_resp,task)
	
	## TRAINING
	folder = 'LSTM/DATA'
	N_trial=N
	if do_training:	

		print('TRAINING...\n')	

		E,conv_iter,conv_iter_2 = model.training(S_train_3D,O_train,ep=1)
		
		str_err = folder+'/'+task+'_error_units'+str(H)+'.txt'
		np.savetxt(str_err,E)
		str_conv = folder+'/'+task+'_conv_units'+str(H)+'_.txt'
		np.savetxt(str_conv,conv_iter)
		str_conv_2 = folder+'/'+task+'_conv_2_units'+str(H)+'.txt'
		np.savetxt(str_conv_2,conv_iter_2)
			
		str_save = folder+'/'+task+'_weights_units'+str(H)+'.h5'
		model.LSTM.save_weights(str_save)

		print('\nSaved model to disk.')

	else:

		str_err = folder+'/'+task+'_error_units'+str(H)+'.txt'
		E = np.loadtxt(str_err)
		str_conv = folder+'/'+task+'_conv_units'+str(H)+'.txt'
		conv_iter = np.loadtxt(str_conv)
		str_conv_2 = folder+'/'+task+'_conv_2_units'+str(H)+'.txt'
		conv_iter_2 = np.loadtxt(str_conv_2)

		str_load = folder+'/'+task+'_weights_'+str(N_trial)+'_2.h5'
		model.LSTM.load_weights(str_load)
		
		# print(np.shape(model.LSTM.get_weights()[0]))
		# print(np.shape(model.LSTM.get_weights()[1]))
		# print(np.shape(model.LSTM.get_weights()[2]))
		# print(np.shape(model.LSTM.get_weights()[3]))
		# print(np.shape(model.LSTM.get_weights()[4]))

		print('\nLoaded weights from disk.')			

	print('\n------------------------------------------------------------------------------------------			\n----------------------------------------------------------------------------------------------\n')

	## TEST
	if do_test:
		print('TEST...\n')
		model.test(S_test_3D,O_test)
	
	print(conv_iter)
	print(conv_iter_2)

	## PLOTS
	# plot of the memory weights
	folder = 'LSTM/IMAGES'
	fontTitle = 26
	fontTicks = 22
	fontLabel = 22

	if do_error_plots:

		N = len(E)
		bin = np.around(N*0.02).astype(int)
		END = np.floor(N/bin).astype(int)
		E = E[:END*bin]
		N = len(E)

		N_round = np.around(N/1000).astype(int)*1000

		E_bin = np.reshape(E,(-1,bin))
		E_bin = np.sum(E_bin,axis=1)
		E_cum = np.cumsum(E)
		E_norm = 100*E_cum/(np.arange(N)+1)
		C = np.where(E==0,1,0)
		C_cum = 100*np.cumsum(C)/(np.arange(N)+1)

		fig= plt.figure(figsize=(20,8))
		plt.subplot(1,2,1)
		plt.bar(bin*np.arange(len(E_bin))/3,E_bin,width=bin/3,color='red',edgecolor='black', alpha=0.6)
		plt.axvline(x=conv_iter_2/3, linewidth=5, color='r')
		tit = 'Seq Predict: Training Convergence'
		plt.title(tit,fontweight="bold",fontsize=fontTitle)		
		plt.xlabel('Training Trials',fontsize=fontLabel)
		plt.ylabel('Number of Errors per bin',fontsize=fontLabel)		
		plt.xticks(np.linspace(0,N_round/3,5,endpoint=True),fontsize=fontTicks)	
		plt.yticks(fontsize=fontTicks)	
		text = 'Bin = '+str(np.around(bin).astype(int))
		plt.figtext(x=0.38,y=0.78,s=text,fontsize=fontLabel,bbox={'facecolor':'white', 'alpha':0.5, 'pad':10})

		plt.subplot(1,2,2)
		plt.plot(np.arange(N)/3, E_cum, color='red',linewidth=7, alpha=0.6)
		plt.axvline(x=conv_iter_2/3, linewidth=5, color='r')
		tit = 'Seq Predict: Cumulative Training Error'
		plt.title(tit,fontweight="bold",fontsize=fontTitle)			
		plt.xticks(np.linspace(0,N_round/3,5,endpoint=True),fontsize=fontTicks)
		plt.yticks(fontsize=fontTicks)
		plt.xlabel('Training Trials',fontsize=fontLabel)
		plt.ylabel('Cumulative Error',fontsize=fontLabel)
		plt.show()
		savestr = folder+'/'+task+'_error_units'+str(H)+'.png'
		fig.savefig(savestr)


#########################################################################################################################################
#######################   TASK COPY
#########################################################################################################################################
def LSTM_task_copy(params_bool, params_task):

	from TASKS.task_copy import data_construction
	task = 'copy'

	if params_task is None:
		size = 8
		min_length = 1
		max_length = 20
		training_iters = 200000
	else:
		size = int(params_task[0]) if params_task[0] != '' else 8
		min_length = int(params_task[1]) if params_task[1] != '' else 1
		max_length = int(params_task[2]) if params_task[2] != '' else 20
		training_iters = int(params_task[3]) if params_task[3] != '' else 200000

	end_marker = True
	test_iters = 1000
	batch_size = 1

	print('Dataset construction...')

	S_train, O_train = data_construction(max_iters=training_iters, batch_size=batch_size, min_length=min_length, \
									max_length=max_length, size=size, end_marker=end_marker)
	S_test, O_test = data_construction(max_iters=test_iters, batch_size=batch_size, min_length=min_length, \
									max_length=max_length, size=size, end_marker=end_marker)

	print('Dataset construction done!')

	S = size+1
	H = [256, 256, 256]
	O = size+1

	learn_rate = 3e-5

	verb = 0
	
	if params_bool is None:
		do_training = True
		do_test = True
		do_weight_plots = True
		do_error_plots = True
		
	else:
		do_training = params_bool[0]
		do_test = params_bool[1]
		do_weight_plots = params_bool[2]	
		do_error_plots = params_bool[3]	

	model = LSTM_arch(S=S,H=H,O=O,learn_rate=learn_rate,batch_size=batch_size,task=task)

	folder = 'LSTM/DATA'
	if do_training:

		print('TRAINING...\n')	
		
		acc, loss, iters = model.training(S_train, O_train, max_iters=training_iters)
		str_loss = folder+'/'+task+'_training_loss.txt'
		np.savetxt(str_loss,loss)
		str_acc = folder+'/'+task+'_training_acc.txt'
		np.savetxt(str_acc,acc)

		str_save = folder+'/'+task+'_weights_units.h5'
		model.LSTM.save_weights(str_save)

		print('\nSaved model to disk.')

	else:

		str_loss = folder+'/'+task+'_training_loss.txt'
		loss = np.loadtxt(str_loss)
		str_acc = folder+'/'+task+'_training_acc.txt'
		acc = np.loadtxt(str_acc)

		str_load = folder+'/'+task+'_weights_units.h5'
		model.LSTM.load_weights(str_load)

		print('\nLoaded weights from disk.')

	if do_test:

		print('TESTING...\n')

		test_loss, test_acc = model.test(S_test, O_test, max_iters=test_iters)

		print('\nAverage Test Loss = ' + str(test_loss))
		print('Average Test Accuracy = ' + str(test_acc))

	example_input, example_output = data_construction(max_iters=1, batch_size=batch_size, min_length=min_length, \
									max_length=max_length, size=size, end_marker=end_marker)
	predicted_output = model.LSTM.predict(example_input)

	print('\nExample input:')
	print(np.asarray(example_input, dtype=np.float32))
	print('\nExample output:')
	print(np.asarray(example_output, dtype=np.float32))
	print('\nPredicted output:')
	print(np.asarray(predicted_output > 0.5, dtype=np.float32))

	folder = 'LSTM/IMAGES'

	if do_error_plots:

		fig = plt.figure(figsize=(20,8))
		plt.subplot(1,2,1)
		plt.plot(iters, acc, 'b-', linewidth=2, alpha=0.8)
		plt.title('Training Accuracy (LSTM - Copy Task)')
		plt.ylabel('Accuracy')
		plt.xlabel('Iteration')
		
		plt.subplot(1,2,2)
		plt.plot(iters, loss, 'b-', linewidth=2, alpha=0.8)
		plt.title('Training Loss (LSTM - Copy Task)')
		plt.ylabel('Loss')
		plt.xlabel('Iteration')
		plt.show()

		savestr = folder+'/'+task+'_training_error_plot.png'
		fig.savefig(savestr)



#########################################################################################################################################
#######################   TASK COPY REPEAT
#########################################################################################################################################
def LSTM_task_copy_repeat(params_bool, params_task):

	from TASKS.task_copy_repeat import data_construction
	task = 'copy_repeat'

	if params_task is None:
		size = 8
		min_length = 1
		max_length = 20
		min_repeats = 2
		max_repeats = 5
		training_iters = 200000

		
	else:
		size = int(params_task[0]) if params_task[0] != '' else 8
		min_length = int(params_task[1]) if params_task[1] != '' else 1
		max_length = int(params_task[2]) if params_task[2] != '' else 20
		min_repeats = int(params_task[3]) if params_task[3] != '' else 2
		max_repeats = int(params_task[4]) if params_task[4] != '' else 5
		training_iters = int(params_task[5]) if params_task[5] != '' else 200000

	end_marker = True
	test_iters = 1000
	batch_size = 10

	print('Dataset construction...')

	S_train, O_train = data_construction(max_iters=training_iters, batch_size=batch_size, min_length=min_length, max_length=max_length, \
								min_repeats=min_repeats, max_repeats=max_repeats, size=size, end_marker=end_marker)
	S_test, O_test = data_construction(max_iters=test_iters, batch_size=batch_size, min_length=min_length, max_length=max_length, \
								min_repeats=min_repeats, max_repeats=max_repeats, size=size, end_marker=end_marker)

	print('Dataset construction done!')

	S = size+2
	H = [512, 512, 512]
	O = size+2

	learn_rate = 3e-5

	verb = 0
	
	if params_bool is None:
		do_training = True
		do_test = True
		do_weight_plots = True
		do_error_plots = True
		
	else:
		do_training = params_bool[0]
		do_test = params_bool[1]
		do_weight_plots = params_bool[2]	
		do_error_plots = params_bool[3]	

	model = LSTM_arch(S=S,H=H,O=O,learn_rate=learn_rate,batch_size=batch_size,task=task)

	folder = 'LSTM/DATA'
	if do_training:

		print('TRAINING...\n')	
		
		acc, loss, iters = model.training(S_train, O_train, max_iters=training_iters)
		str_loss = folder+'/'+task+'_training_loss.txt'
		np.savetxt(str_loss,loss)
		str_acc = folder+'/'+task+'_training_acc.txt'
		np.savetxt(str_acc,acc)

		str_save = folder+'/'+task+'_weights_units.h5'
		model.LSTM.save_weights(str_save)

		print('\nSaved model to disk.')

	else:

		str_loss = folder+'/'+task+'_training_loss.txt'
		loss = np.loadtxt(str_loss)
		str_acc = folder+'/'+task+'_training_acc.txt'
		acc = np.loadtxt(str_acc)

		str_load = folder+'/'+task+'_weights_units.h5'
		model.LSTM.load_weights(str_load)

		print('\nLoaded weights from disk.')

	if do_test:

		print('TESTING...\n')

		test_loss, test_acc = model.test(S_test, O_test, max_iters=test_iters)

		print('\nAverage Test Loss = ' + str(test_loss))
		print('Average Test Accuracy = ' + str(test_acc))

	example_input, example_output = data_construction(max_iters=1, batch_size=batch_size, min_length=min_length, max_length=max_length, \
								min_repeats=min_repeats, max_repeats=max_repeats, size=size, end_marker=end_marker)
	predicted_output = model.LSTM.predict(example_input)

	print('\nExample input:')
	print(np.asarray(example_input, dtype=np.float32))
	print('\nExample output:')
	print(np.asarray(example_output, dtype=np.float32))
	print('\nPredicted output:')
	print(np.asarray(predicted_output > 0.5, dtype=np.float32))

	folder = 'LSTM/IMAGES'

	if do_error_plots:

		fig = plt.figure(figsize=(20,8))
		plt.subplot(1,2,1)
		plt.plot(iters, acc, 'b-', linewidth=2, alpha=0.8)
		plt.title('Training Accuracy (LSTM - Repeat Copy Task)')
		plt.ylabel('Accuracy')
		plt.xlabel('Iteration')
		
		plt.subplot(1,2,2)
		plt.plot(iters, loss, 'b-', linewidth=2, alpha=0.8)
		plt.title('Training Loss (LSTM - Repeat Copy Task)')
		plt.ylabel('Loss')
		plt.xlabel('Iteration')
		plt.show()

		savestr = folder+'/'+task+'_training_error_plot.png'
		fig.savefig(savestr)