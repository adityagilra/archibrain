import numpy as np

model_dic = {
	'0': 'AuGMEnT',
	'1': 'HER',
	'2': 'LSTM'
}

model_selection = input("\nPlease select a model: \n\t 0: AuGMEnT \n\t 1: HER \n\t 2: LSTM\nEnter model number: ")
print("\nSelected model: ", model_dic[model_selection],'\n\n')


task_dic = {
	'0':'task 1-2', 
	'1':'task AX_CPT',
	'2':'task 12-AX_S',
	'3':'task 12-AX',
	'4':'saccade/anti-saccade task'
}

task_selection = input("\nPlease select a task: \n\t 0: task 1-2 \n\t 1: task AX_CPT \n\t 2: task 12-AX-S\n\t 3: task 12-AX\n\t 4: saccade/anti-saccade task\nEnter task number: ")
print("\nSelected task: ", task_dic[task_selection],'\n\n')

# boolean parameters for simulation steps
enter_params_bool = input("\nSkip steps for task simulation? (Yes/No, default No): ")

if(enter_params_bool[0] == 'y' or enter_params_bool[0] == 'Y'):
	params_bool = []
	params_bool_new = []
	params_bool.append((input("Do training? (Yes/No): ")))
	params_bool.append((input("Do test? (Yes/No): ")))
	params_bool.append((input("Do weight plots? (Yes/No): ")))
	params_bool.append((input("Do error plots? (Yes/No): ")))
	for p in params_bool:
		if(p[0] == 'y' or p[0] == 'Y'):
			params_bool_new.append(True)
		else:
			params_bool_new.append(False)

else:
	params_bool_new = None


# task specific parameters
enter_params_task = input("\nChange parameters for task building? (Yes/No, default No): ")

if(task_selection == '0'):
	if(enter_params_task[0] == 'y' or enter_params_task[0] == 'Y'):
		params_task = []
		params_task.append(int(input("\nNumber of trials (N): ")))
		params_task.append(float(input("Probability of '1' (p1): ")))
		params_task.append(float(input("Training percentage: ")))
	
	else:
		params_task = None

elif(task_selection == '1'):
	if(enter_params_task[0] == 'y' or enter_params_task[0] == 'Y'):
		params_task = []
		params.append(int(input("\nNumber of trials (N): ")))
		params.append(float(input("Probability of target (p1): ")))
		params.append(float(input("Training percentage: ")))
	
	else:
		params_task = None

elif(task_selection == '2'):
	if(enter_params_task[0] == 'y' or enter_params_task[0] == 'Y'):
		params_task = []
		params.append(int(input("\nNumber of trials (N): ")))
		params.append(float(input("Probability of digit (p_digit): ")))
		params.append(float(input("Probability of wrong response (p_wrong): ")))
		params.append(float(input("Probability of correct response (p_correct): ")))
		params.append(float(input("Training percentage: ")))
	
	else:
		params_task = None

elif(task_selection == '3'):
	if(enter_params_task[0] == 'y' or enter_params_task[0] == 'Y'):
		params_task = []
		params.append(int(input("\nNumber of trials (N): ")))
		params.append(float(input("Probability of target (p_target): ")))
		params.append(float(input("Training percentage: ")))
	
	else:
		params_task = None

elif(task_selection == '4'):
	if(enter_params_task[0] == 'y' or enter_params_task[0] == 'Y'):
		params_task = []
		params.append(int(input("\nNumber of trials (N): ")))
		params.append(float(input("Training percentage: ")))
	
	else:
		params_task = None


if(model_selection == '0'):
	from AuGMEnT.AuGMEnT_tasks import run_task
	run_task(task_selection, params_bool_new, params_task)

elif(model_selection == '1'):
	from HER.HER_tasks import run_task
	run_task(task_selection, params_bool_new, params_task)

if(model_selection == '2'):
	from LSTM.LSTM_tasks import run_task
	run_task(task_selection, params_bool_new, params_task)
