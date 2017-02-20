## IMPLEMENTATION OF SUPPORT VECTOR MACHINE (SVM) METHOD USING SCIKIT AND KERAS 
## AUTHOR: Marco Martinolli
## DATE: 20.02.2017

from sklearn import svm
from numpy import linalg as LA

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.regularizers import l2

import pandas as pd 
import numpy as np

# load data of the 2200 titanic passengers.
# Categories:
#   - social class      1st, 2nd, 3rd, crew
#   - age               adult/young
#   - sex               male/female
#   - survived          yes/no  <-- target

def load_data(string_file):
	
	df=np.genfromtxt(string_file,dtype=None)

	# shuffle the dataset
	np.random.shuffle(df)

	# convert string categorical variables to integers
	X=df[:,0:3]
	X[:,0]=np.where(X[:,0]==b"1st",1,X[:,0])
	X[:,0]=np.where(X[:,0]==b"2nd",2,X[:,0])
	X[:,0]=np.where(X[:,0]==b"3rd",3,X[:,0])
	X[:,0]=np.where(X[:,0]==b"crew",4,X[:,0])
	X[:,1]=np.where(X[:,1]==b"adult",1,0)
	X[:,2]=np.where(X[:,2]==b"male",1,0)

	X=X.astype(np.int)

	y=df[:,3]
	y=np.where(y==b"yes",1,0)
	
	return X,y

def subset_data(X,y,training_perc=0.8):

	sz=len(y)
	idx=int(np.around(sz*training_perc))	

	# distinction in training and test sets
	X_train=X[:idx,:]
	y_train=y[:idx]
	X_test=X[idx:,:]
	y_test=y[idx:]

	return X_train,y_train,X_test,y_test

def predict_scikit(X_train,y_train,X_test,verbose=False):
	
	classificator = svm.SVC()

	classificator.fit(X_train, y_train)

	p=classificator.predict(X_test)

	if (verbose):
		print('Test predictions \n',p)

	return p


def predict_keras(X_train,y_train,X_test,hid_num=30,epc_num=50,verb=False):
	
	y_tr=np.where(y_train==1,1,-1)

	# Construction of the neural network with keras
	# 3 layers, first 2 activated as rectifiers, output as sigmoidal
	
	model=Sequential()

	model.add(Dense(hid_num, input_dim=3))
	model.add(Activation('relu'))
	     
	model.add(Dropout(0.5))
	model.add(Dense(output_dim=1, W_regularizer=l2(0.01)))
	model.add(Activation('linear'))
	
	# binary classification
	model.compile(loss='hinge', 
		      optimizer='adadelta', 
		      metrics=['accuracy'])

	# fitting on the training dataset
	model.fit(X_train, y_tr, nb_epoch=epc_num,verbose=verb)
	
	# predict the output on the test set
	predictions = model.predict(X_test)
	pred_rounded = np.sign(predictions) 
	pred_rounded = np.where(pred_rounded==1,1,0)
	pred_rounded = np.squeeze(pred_rounded)	

	if(verb):
		print('Test predictions \n',pred_rounded)		

	return pred_rounded


def evaluation(pred,y_test):

	# Evaluation Table for binary classification
	##  True Pos    False Pos
	##  False Neg   True Neg
	Tab=np.zeros((2,2))	
	
	for p,t in zip(pred,y_test):
		if(p==1 and t==1):		
			Tab[0,0]+=1			
		elif(p==1 and t==0):		
			Tab[0,1]+=1
		elif(p==0 and t==1):		
			Tab[1,0]+=1
		else:		
			Tab[1,1]+=1

	print('\nT.P.=',Tab[0,0],'  F.P.=',Tab[0,1],'\nF.N.=',Tab[1,0],'  T.N.=',Tab[1,1])
	
	# Accuracy
	ACC=Tab[0,0]/(Tab[0,0]+Tab[0,1])

	# Recall	
	REC=Tab[0,0]/(Tab[0,0]+Tab[1,0])

	# F1-score
	F1_score=2*ACC*REC/(ACC+REC)	
	print('F1-score: ', F1_score)
	
	
	# Mean Squared Error
	ERR=(LA.norm(pred-y_test)**2)/len(y_test)
	print('Mean Squared Error: ', ERR)

	return 	 
 


[X,y]=load_data("titanic_data.csv")
[X_train,y_train,X_test,y_test]=subset_data(X,y,training_perc=0.8)

# Predictions using SCIKIT implementation of SVM
print('\n \n SVM PREDICTIONS (SCIKIT)')
pred_SVM=predict_scikit(X_train,y_train,X_test)
evaluation(pred_SVM,y_test)

# Predictions using keras implementation of SVM 
print('\n \n SVM PREDICTIONS (KERAS)\n')
pred_keras=predict_keras(X_train,y_train,X_test,hid_num=30,epc_num=100)
evaluation(pred_keras,y_test)


