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

enter_params = input("\nSet parameters? (0 - n, 1 - y): ")


if(model_selection == '0'):

	from AuGMEnT.AuGMEnT_tasks import task_selector
	
	if(enter_params == '1'):
		params = []
		params.append(int(input("\n\nDimension of regular units: ")))
		params.append(int(input("Dimension of memory units: ")))
		params.append(float(input("Synaptic tag decay (lambda): ")))
		params.append(float(input("Weight update coefficient (beta): ")))
		params.append(float(input("Discount rate for future rewards (gamma): ")))
		params.append(float(input("Percentage of softmax modality (epsilon): ")))
		params.append(float(input("Gain (g): ")))

		params.append(int(input("Reward System (0 - RL, 1 - PL, 2 - SRL: ")))
		params.append(int(input("Do training? (0 - n, 1 - y): ")))
		params.append(int(input("Do test? (0 - n, 1 - y): ")))
		params.append(int(input("Do weight plots? (0 - n, 1 - y): ")))
		params.append(int(input("Do error plots? (0 - n, 1 - y): ")))
	else:
		params = None

	task_selector(task_selection, params)

elif(model_selection == '1'):
	from HER.HER_tasks import task_selector
	task_selector(task_selection)

if(model_selection == '2'):
	from LSTM.LSTM_tasks import task_selector
	task_selector(task_selection)
