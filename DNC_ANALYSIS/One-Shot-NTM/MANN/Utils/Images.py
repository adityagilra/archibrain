import tensorflow as tf
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
import os
import random

from scipy.ndimage import rotate,shift
from scipy.misc import imread,imresize


def get_shuffled_images(paths, labels, nb_samples=None):
    if nb_samples is not None:
        sampler = lambda x: random.sample(x, nb_samples)
    else:
        sampler = lambda x:x

    images = [(i, os.path.join(path, image)) for i,path in zip(labels,paths) for image in sampler(os.listdir(path)) ]
    random.shuffle(images)
    return images

def time_offset_label(labels_and_images):
    labels, images = zip(*labels_and_images)
    time_offset_labels = (None,) + labels[:-1]
    return zip(images, time_offset_labels)

def load_transform(image_path, angle=0., s=(0,0), size=(20,20)):
    #Load the image
    original = imread(image_path, flatten=True)/255

    #Rotate the image
    rotated = rotate(original, angle=angle, cval=1.)
    #rotated = (rotated - np.min(rotated))/(np.max(rotated)-np.min(rotated)) 

    #Shift the image
    shifted = shift(rotated, shift=s,cval=1.)
    #shifted = (shifted - np.min(shifted))/(np.max(shifted)-np.min(shifted)) 

    #Resize the image
    resized = np.asarray(imresize(shifted, size=size), dtype=np.float32) #Note here we coded manually as np.float32, it should be tf.float32
    #resized = np.asarray(shifted,dtype=np.float32)

    final = (resized-np.min(resized))/(np.max(resized)-np.min(resized))

    final = 1-final

    return final
