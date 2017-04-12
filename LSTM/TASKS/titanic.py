## TITANIC TASK
## The dataset consists of 2200 titanic passengers classified by age, sex and social class.
## The classification task corresponds to learn which passengers survived depending on the input categories.
## AUTHOR: Marco Martinolli
## DATE: 07.03.2017

from numpy import linalg as LA

import pandas as pd 
import numpy as np

# load data of the 2200 titanic passengers.
# Categories:
#   - social class      1st, 2nd, 3rd, crew
#   - age               adult/young
#   - sex               male/female
#   - survived          yes/no          <-- target

def load_data(string_file):
	
	df=np.genfromtxt(string_file,dtype=None)

	# shuffle the dataset
	np.random.shuffle(df)

	# convert string categorical variables to integers
	X=df[:,0:3]
	X[:,0] = np.where(X[:,0]==b"1st",1,X[:,0])
	X[:,0] = np.where(X[:,0]==b"2nd",2,X[:,0])
	X[:,0] = np.where(X[:,0]==b"3rd",3,X[:,0])
	X[:,0] = np.where(X[:,0]==b"crew",4,X[:,0])
	X[:,1] = np.where(X[:,1]==b"adult",1,0)
	X[:,2] = np.where(X[:,2]==b"male",1,0)

	X=X.astype(np.int)

	y = df[:,3]
	y = np.transpose(np.array([y]))
	y=np.where(y==b"yes",[1,0,0,1],[0,1,1,0])
	
	return X,y

def subset_data(X,y,training_perc=0.8):

	sz=len(y)
	idx=int(np.around(sz*training_perc))	

	# Distintion in training and test sets
	X_train=X[:idx,:]
	y_train=y[:idx]
	X_test=X[idx:,:]
	y_test=y[idx:]

	return X_train,y_train,X_test,y_test


def data_construction(perc_training):
	
	[X,y] = load_data("DATA/titanic_data.csv")

	[X_train,y_train,X_test,y_test] = subset_data(X,y,perc_training)
		
	dic_resp =  {'array([[1, 0, 0, 1]])':'survived', 'array([[0, 1, 1, 0]])':'dead'}		

	return X_train, y_train, X_test, y_test, dic_resp

