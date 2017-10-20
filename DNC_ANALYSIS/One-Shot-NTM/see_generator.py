import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'  #No logging TF

import tensorflow as tf
import numpy as np
import time
from scipy.ndimage import imread
from scipy.misc import imresize
from matplotlib import pyplot as plt
from matplotlib import gridspec
import matplotlib.image as mpimg
import pylab
import gc 

from MANN.Model import memory_augmented_neural_network
from MANN.Utils.Generator import OmniglotGenerator
from MANN.Utils.Metrics import accuracy_instance
from MANN.Utils.tf_utils import update_tensor

from scipy.ndimage import rotate,shift
from scipy.misc import imread,imresize

def data_augment():

	sess = tf.InteractiveSession()

	size = 20
	generator = OmniglotGenerator(data_folder='./data/omniglot', batch_size=1, nb_samples=5, nb_samples_per_class=10, max_rotation=30, max_shift=10, img_size=(size,size), max_iter=1)

	fig = plt.figure(figsize=(20,10))

	for i, (batch_input, batch_output) in generator:

		for j in np.arange(generator.nb_samples*generator.nb_samples_per_class):

			img = batch_input[0,j,:]
			lab = int(batch_output[0,j])
			img_reshaped = np.reshape(img,(size,size))
			plt.subplot(5,10,j+1)
			plt.imshow(1-img_reshaped, cmap='gray', interpolation="bicubic")
			plt.title(str(lab+1),fontweight='bold',fontsize=40,color='r')

		plt.show()

	str_save = 'episode_example.png'
	#str_save = 'ten_classes.png'
	fig.savefig(str_save)

def char_augment():
	
	num_samples = 2
	max_rot = 30
	max_sh = 10
	size = 20

	angles = np.random.uniform(-max_rot, max_rot, size=num_samples)
	shifts = np.random.uniform(-max_sh, max_sh, size=num_samples)

	print(angles)
	print(shifts)

	original = imread('char.png', flatten=True)
	original = original/255

	orig_size = np.shape(original)[0] 

	#plt.imshow(original,cmap='gray')
	#plt.show()

	for n in np.arange(num_samples):

		print('\nIMAGE ',n+1)

		rotated = np.maximum(np.minimum(rotate(original, angle=angles[n], cval=1.), 1.), 0.)
    		shifted = np.maximum(np.minimum(shift(rotated, shift=shifts[n],cval=1.), 1.), 0.)

    		resized = np.asarray(imresize(shifted, size=orig_size), dtype=np.float32)
    		final = (resized-np.min(resized))/(np.max(resized)-np.min(resized) )
		final = 1-final

		plt.subplot(2,2,2*n+1)
		plt.imshow(1-final, cmap='gray')
		#plt.title(str(n+1),fontweight='bold',fontsize=40)


    		resized = np.asarray(imresize(shifted, size=size), dtype=np.float32)
    		final = (resized-np.min(resized))/(np.max(resized)-np.min(resized) )
		final = 1-final


		plt.subplot(2,2,2*(n+1))
		plt.imshow(1-final, cmap='gray')

		#plt.title(str(n+1),fontweight='bold',fontsize=40)

	
	plt.show()

		
def main():
	#char_augment()
	data_augment()
	

main()
