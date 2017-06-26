# define models and tasks
model_dic = {
	'0': 'AuGMEnT',
	'1': 'HER',
	'2': 'LSTM'
}

task_dic = {
	'0':'task 1-2', 
	'1':'task AX_CPT',
	'2':'task 12-AX_S',
	'3':'task 12-AX',
	'4':'saccade/anti-saccade task',
	'5':'sequence prediction task',
	'6':'copy task',
	'7':'repeat copy task'
}

def check_validity(model_selection, task_selection):
	if model_selection == '0':
		if task_selection == '6' or task_selection == '7':
			return False
	elif model_selection == '1':
		if task_selection == '6' or task_selection == '7':
			return False
	elif model_selection == '2':
		if task_selection == '0' or task_selection == '1' or task_selection == '2':
			return False
	return True


def main():
	# select model
	model_selection = input("\nPlease select a model: \n\t 0: AuGMEnT \n\t 1: HER \n\t 2: LSTM\nEnter model number: ")

	# select task
	task_selection = input("\nPlease select a task: \n\t 0: task 1-2 \n\t 1: task AX_CPT \n\t 2: task 12-AX-S\n" + 
					"\t 3: task 12-AX\n\t 4: saccade/anti-saccade task\n\t 5: sequence prediction task\n" + 
					"\t 6: copy task\n\t 7: repeat copy task \nEnter task number: ")

	# check model and task selection is valid
	if(not check_validity(model_selection, task_selection)):
		print('Your selection is invalid.')
		return

	print("\nSelected model: ", model_dic[model_selection])
	print("\nSelected task: ", task_dic[task_selection],'\n\n')

	# boolean parameters for simulation steps
	enter_params_bool = input("Skip steps for task simulation? (Yes/No, default No): ")
	params_bool_new = None

	if(enter_params_bool != ''):
		if(enter_params_bool[0] == 'y' or enter_params_bool[0] == 'Y'):
			params_bool = []
			params_bool_new = []
			params_bool.append((input("Do training? (Yes/No, default: Yes): ")))
			params_bool.append((input("Do test? (Yes/No, default: Yes): ")))
			params_bool.append((input("Do weight plots? (Yes/No, default: Yes): ")))
			params_bool.append((input("Do error plots? (Yes/No, default: Yes): ")))
			for p in params_bool:
				if(p != ''):
					if(p[0] == 'y' or p[0] == 'Y'):
						params_bool_new.append(True)
					else:
						params_bool_new.append(False)
				else:
					params_bool_new.append(True)


	# task specific parameters
	enter_params_task = input("\nChange parameters for task building? (Yes/No, default No): ")
	params_task = None

	if(enter_params_task != ''):
		if(task_selection == '0'):
			if(enter_params_task[0] == 'y' or enter_params_task[0] == 'Y'):
				params_task = []
				params_task.append(input("Number of trials (default: N = 4000): "))
				params_task.append(input("Probability of '1' (default: p1 = 0.7): "))
				params_task.append(input("Training percentage (default: tr_perc = 0.8): "))

		elif(task_selection == '1'):
			if(enter_params_task[0] == 'y' or enter_params_task[0] == 'Y'):
				params_task = []
				params_task.append(input("Number of trials (default: N = 40000): "))
				params_task.append(input("Probability of target (default: p_target = 0.2): "))
				params_task.append(input("Training percentage (default: tr_perc = 0.8): "))

		elif(task_selection == '2'):
			if(enter_params_task[0] == 'y' or enter_params_task[0] == 'Y'):
				params_task = []
				params_task.append(input("Number of trials (default: N = 100000): "))
				params_task.append(input("Probability of digit (default: p_digit = 0.1): "))
				params_task.append(input("Probability of wrong response (default: p_wrong = 0.15): "))
				params_task.append(input("Probability of correct response (default: p_correct = 0.25): "))
				params_task.append(input("Training percentage (default: tr_perc = 0.8): "))

		elif(task_selection == '3'):
			if(enter_params_task[0] == 'y' or enter_params_task[0] == 'Y'):
				params_task = []
				params_task.append(input("Number of trials (default: N = 20000): "))
				params_task.append(input("Probability of target (default: p_target = 0.5): "))
				params_task.append(input("Training percentage (default: tr_perc = 0.8): "))

		elif(task_selection == '4'):
			if(enter_params_task[0] == 'y' or enter_params_task[0] == 'Y'):
				params_task = []
				params_task.append(input("Number of trials (default: N = 20000): "))
				params_task.append(input("Training percentage (default: tr_perc = 0.8): "))

		elif(task_selection == '5'):
			if(enter_params_task[0] == 'y' or enter_params_task[0] == 'Y'):
				params_task = []
				params_task.append(input("Number of trials (default: N = 5000): "))
				params_task.append(input("Probability of target (default: p_target = 0.5): "))
				params_task.append(input("Training percentage (default: tr_perc = 0.8): "))

		elif(task_selection == '6'):
			if(enter_params_task[0] == 'y' or enter_params_task[0] == 'Y'):
				params_task = []
				params_task.append(input("Size of bit string (default: size = 8): "))
				params_task.append(input("Minimum length of sequence (default: min_length = 1): "))
				params_task.append(input("Maximum length of sequence (default: max_length = 20): "))
				params_task.append(input("Number of training iterations (default: training_iters = 200000): "))
				
		elif(task_selection == '7'):
			if(enter_params_task[0] == 'y' or enter_params_task[0] == 'Y'):
				params_task = []
				params_task.append(input("Size of bit string (default: size = 8): "))
				params_task.append(input("Minimum length of sequence (default: min_length = 1): "))
				params_task.append(input("Maximum length of sequence (default: max_length = 20): "))
				params_task.append(input("Minimum number of repeats (default: min_repeats = 2): "))
				params_task.append(input("Maximum number of repeats (default: max_repeats = 5): "))
				params_task.append(input("Number of training iterations (default: training_iters = 200000): "))
				

	print('\n\n')

	# run appropriate task depending on selected model
	if(model_selection == '0'):
		from AuGMEnT.AuGMEnT_tasks import run_task
		run_task(task_selection, params_bool_new, params_task)

	elif(model_selection == '1'):
		from HER.HER_tasks import run_task
		run_task(task_selection, params_bool_new, params_task)

	if(model_selection == '2'):
		from LSTM.LSTM_tasks import run_task
		run_task(task_selection, params_bool_new, params_task)


if __name__ == '__main__':
    main()