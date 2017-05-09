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


if(model_selection == '0'):
	from AuGMEnT.AuGMEnT_tasks import task_selector
	task_selector(task_selection)

elif(model_selection == '1'):
	from HER.HER_tasks import task_selector
	task_selector(task_selection)

if(model_selection == '2'):
	from LSTM.LSTM_tasks import task_selector
	task_selector(task_selection)
