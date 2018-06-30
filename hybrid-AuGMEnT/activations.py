## LIBRARY FOR ACTIVATION FUNCTIONS
##
## AUTHOR: Marco Martinolli
## DATE: 30.03.2017


import numpy as np


def linear(inp,W):
	return np.dot(inp,W)

def rectifier(inp,W):
	return np.amax(np.array([0,linear(inp,W)]))

def sigmoidal(inp):
	f = 1/(1+np.exp(-inp))
	return f

def sigmoid(inp,W):
	tot = np.dot(inp,W)
	tot = np.clip(a=tot,a_min=-100,a_max=None) 
	f = 1/(1+np.exp(-tot))
	return f
	
def sigmoid_2(inp,W):
	tot = np.sum(np.multiply(inp,W),axis=0,keepdims=True)
	tot = np.clip(a=tot,a_min=-100,a_max=None) 
	f = 1/(1+np.exp(-tot))
	return f

def hard_sigmoid(inp,W,strength=1):
	tot1 = np.dot(inp,W)
	tot2 = np.clip(a=tot1,a_min=-100,a_max=None) 
	f = 1/(1+np.exp(-strength*tot2))
	return f

def sigmoid_acc(inp,W,acc,gate=1):
	tot1 = acc + gate*np.dot(inp,W)
	tot2 = np.clip(a=tot1,a_min=-100,a_max=None) 
	f = 1/(1+np.exp(-tot2))
	return f,tot


def sigmoid_acc_leaky(inp, W, acc, leak, gate=1):
	
	if isinstance(leak, list):
		num_groups=int(np.shape(acc)[1]/len(leak))
		leak_vec = np.repeat(leak, num_groups)
		tot1 = leak_vec*acc + gate*np.dot(inp,W)
	else:
		tot1 = leak*acc + gate*np.dot(inp,W)
	tot2 = np.clip(a=tot1,a_min=-100,a_max=None) 
	f = 1/(1+np.exp(-tot2))
	return f,tot1		

def softmax(inp):
	soft = np.exp(inp)/np.sum(np.exp(inp))
	return soft

def tanh(inp,W):
	tot = np.dot(inp,W)
	return np.tanh(tot)
