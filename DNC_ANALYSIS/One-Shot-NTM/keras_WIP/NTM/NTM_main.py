## MAIN FILE FOR AuGMEnT TESTING
## Here are defined the settings for the AuGMEnT architecture and for the tasks to test.
## AUTHOR: Marco Martinolli
## DATE: 27.03.2017

from NTM_model_keras_2 import NTM

import numpy as np
import matplotlib 
matplotlib.use('GTK3Cairo') 
from matplotlib import pyplot as plt
from matplotlib import gridspec
import pylab
import gc 

from sys import version_info

task_dic ={'1':'copy task',
	   '2':'copy-repeat task',
	   '3':'omniglot task',
	   '4':'omniglot task (episodes)'}

py3 = version_info[0] > 2 	# creates boolean value for test that Python major version > 2
task_selection = input("\nPlease select a task: \n\t 1: copy task\n\t 2: copy-repeat task \n\t 3: omniglot task \n\t 4: omniglot task - episodes \n Enter id number:  ")
print("\nYou have selected: ", task_dic[task_selection],'\n\n')

#########################################################################################################################################
#######################   TASK COPY
#########################################################################################################################################
 
if (task_selection=="1"):
	
	from TASKS.task_copy_2 import data_construction
	task = 'copy'

	length = 6
	size = 5
	end_marker = True	

	N_train = 100000
	N_test = 4000

	X_train, Y_train = data_construction(N_train, length=length, size=size, end_marker=end_marker)
	X_test, Y_test = data_construction(N_test, length=length, size=size, end_marker=end_marker)		
	
	S = [np.shape(X_train)[1], np.shape(X_train)[2]]
	H = 256
	N = 128
	M = 40
	O = [np.shape(Y_train)[1], np.shape(Y_train)[2]]
	
	num_heads = 1
	gamma =	0.99

	lr = 3e-5
	momentum = 0.9
	decay = 0
	mini_batch = 1

	optim = 'RMSprop'
		
	time_depth = True
		
	do_training = False
	do_test = False
	do_plots = True
	do_check = False

	verb = 1

	model = NTM(S, O, H, N, M, num_heads, lr, decay, momentum, gamma, mini_batch, optim, time_depth)
	
	if do_check:
		np.set_printoptions(precision=2)
		X, Y = data_construction(3, length=3, size=4, end_marker=end_marker)
		print(X)
		print(np.shape(X))
		print(Y)
		print(np.shape(Y))
		print('---------------------------------------------------------------------------')
		model2 = NTM([np.shape(X)[1], np.shape(X)[2]], [np.shape(Y)[1], np.shape(Y)[2]], 10, 5, 4, 1, 3e-5, 0, 0.9, 0.99, 16,'RMSprop', True)		
		model2.check(X, Y, verb)
		
	if do_training:
		E,least_usage = model.training(X_train, Y_train)
	
	if do_test:
		LOSS, ACC, least_usage = model.test(X_test, Y_test)

		example_input, example_output = data_construction(1, length=length, size=size, end_marker=end_marker)
		predicted_output = model.NTM_to_fit.predict([example_input,model.MEMORY,model.read_weights,model.usage_weights,least_usage])

		print('\nExample input:')
		print(example_input)
		print('\nExample output:')
		print(example_output)
		print('\nPredicted output:')
		print(predicted_output)

		iters = np.arange(np.shape(ACC)[0])*mini_batch

		if do_plots:
			fig = plt.figure(figsize=(20,8))
			plt.subplot(1,2,1)
			plt.plot(iters, ACC, 'b-', linewidth=2, alpha=0.8)
			plt.title('Training Accuracy (1L-NTM - Copy Task)')
			plt.ylabel('Accuracy')
			plt.xlabel('Iteration')	

			plt.subplot(1,2,2)
			plt.plot(iters, LOSS, 'r-', linewidth=2, alpha=0.8)
			plt.title('Training Loss (1L-NTM - Copy Task)')
			plt.ylabel('Loss')
			plt.xlabel('Iteration')
			plt.show()

	if do_plots:
		from keras.utils import plot_model
		plot_model(model.NTM, to_file='NTM_model.png',show_shapes=False,show_layer_names=True)


#########################################################################################################################################
#######################   TASK COPY-REPEAT
#########################################################################################################################################

if (task_selection=="2"):

	from TASKS.task_copy_repeat import data_construction
	task='copy_repeat'

	length = 5
	size = 8
	repeats = 3
	max_repeats = 10
	unary = False
	end_marker = True

	N_train = 120
	N_test = 40

	X_train, Y_train = data_construction(N_train, length=length, size=size, repeats=repeats, max_repeats=max_repeats, unary=unary, end_marker=end_marker)
	X_test, Y_test = data_construction(N_test, length=length, size=size, repeats=repeats, max_repeats=max_repeats, unary=unary, end_marker=end_marker)

	S = np.shape(X_train)[2]
	H = 10
	N = 12
	M = 5
	O = np.shape(Y_train)[2]
	
	num_heads = 1
	lr = 0.0001
	momentum = 0.9
	decay = 0.95
	alpha_init = 0.5
	gamma =	0.99
	mini_batch = 16

	optim = 'Adam'
		
	do_training = True
	do_test = False
	do_plots = False

	verb = 1

	model = NTM(S, O, H, N, M, num_heads, lr, decay, alpha_init, gamma, mini_batch, optim)

	if do_training:
		E = model.training(X_train,Y_train,verb)
	
	if do_test:
		model.test(X_test,Y_test)

#########################################################################################################################################
#######################   OMNIGLOT TASK
#########################################################################################################################################

if (task_selection=="3"):

	from TASKS.omniglot_task import construct_dataset
	task = 'omniglot'
	
	N = 100  	#  number of character types (20 episodes each)
	N_new = 20 	#  number of character types NOT used in learning (20 episodes each)
	p_tr = 0.8	
	path_to_img = 'omniglot-master/python/images_background/'
	path_to_img_new = 'omniglot-master/python/images_evaluation/'
	
	X_train, Y_train, X_test, Y_test, X_new, Y_new, label_dictionary = construct_dataset(N, N_new, p_tr, path_to_img, path_to_img_new)

	S = np.shape(X_train)[1]
	H = 200
	N = 128
	M = 40
	O = np.shape(Y_train)[1]
	
	num_heads = 4
	lr = 0.0001
	alpha_init = 0.5
	gamma =	0.99
		
	do_training = True
	do_test = False
	do_plots = False

	model = NTM(S, O, H, N, M, num_heads, lr, decay, alpha_init, gamma, mini_batch, optim, label_dictionary)

	#print(Y_train)
	print(label_dictionary)

	if do_training:
		E = model.training(X_train,Y_train,verb)
	
	if do_test:
		model.test(X_test,Y_test)

#########################################################################################################################################
#######################   OMNIGLOT TASK - EPISODES
#########################################################################################################################################

if (task_selection=="4"):

	from TASKS.omniglot_task_2 import construct_dataset
	task = 'omniglot'
	
	N_episodes = 100000 	#  number of character types (20 episodes each)
	N_episodes_new = 100 	#  number of character types NOT used in learning (20 episodes each)
	
	N_char = 5
	N_images = N_char*10
	
	path_to_img = 'omniglot-master/python/images_background/'
	path_to_img_new = 'omniglot-master/python/images_evaluation/'

	S = 20*20+N_char
	H = 200
	N = 128
	M = 40
	O = N_char
	
	num_heads = 1
	lr = 0.0001
	momentum = 0.9
	decay = 0.95
	gamma =	0.99
	mini_batch = 1		 # or 16?

	optim = 'RMSprop'
		
	check = False
	do_training = True
	do_test = True
	do_plots = True

	verb = 1

	model = NTM(S, O, H, N, M, num_heads, lr, decay, gamma, mini_batch, optim)

	if do_training:

		E = model.training_episodes(N_episodes,N_char,N_images,path_to_img,verb)

	if do_test:
		model.test_episodes(N_episodes_new,N_char,N_images,path_to_img_new,verb)

	fig = plt.figure(figsize=(10,8))
	plt.plot(E,'r-', linewidth=4, alpha=0.8)
	plt.title('Training Accuracy (1L-NTM - Omniglot Task)')
	plt.ylabel('Classification Errors')
	plt.xlabel('Episodes')	

	plt.show()
