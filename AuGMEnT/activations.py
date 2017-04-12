## LIBRARY FOR ACTIVATION FUNCTIONS
##
## AUTHOR: Marco Martinolli
## DATE: 30.03.2017


import numpy as np


def linear(inp,W):
	return np.dot(inp,W)

def rectifier(inp,W):
	return np.amax(np.array([0,linear(inp,W)]))

def sigmoid(inp,W):
	tot = np.dot(inp,W)
	f = 1/(1+np.exp(-tot))
	return f

def sigmoid_acc(inp,W,acc):
	tot = acc + np.dot(inp,W)
	f = 1/(1+np.exp(-tot))
	return f,tot

def tanh(inp,W):
	tot = np.dot(inp,W)
	return np.tanh(tot)  
