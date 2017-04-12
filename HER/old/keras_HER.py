## CUSTOMIZED KERAS LAYERS FOR HER LEVEL STRUCTURE
## AUTHOR: Marco Martinolli
## DATE: 27.02.2017

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.regularizers import l2
from keras.engine.topology import Layer

import numpy as np

class WMBlock(Layer):

	def __init__(self, output_dim,*args,**kwargs):
		print(output_dim)
		self.output_dim = output_dim
		super(WMBlock, self).__init__(*args, **kwargs)
	
	def build(self, input_shape):
		self.X = self.add_weight(shape = (input_shape[1], self.output_dim),
					 initializer = 'uniform',
					 trainable = True)
		super(WMBlock,self).build(input_shape)
		
	def call(self, s, mask=None):
			return np.dot(s, self.X)		
	 


class PredBlock(Layer):

	def __init__(self, output_dim, *args,**kwargs):
		self.output_dim = output_dim
		super(PredBlock,self).__init__(*args,**kwargs)
	

	def build(self, input_shape):
		self.W = self.add_weight(shape = (self.output_dim,input_shape[1]),
					 initializer = 'uniform',
					 trainable = True)
		super(PredBlock,self).build(input_shape[0])
		
	def call(self, r, p_err=None, mask=None):
		
		print(np.shape(self.W))		
		if p_err is not None:
			P_err = np.reshape(p_err,(np.shape(W)[0],-1))
			return np.dot(self.W + P_err, r)
		else:
			return np.dot(np.transpose(self.W), r)

