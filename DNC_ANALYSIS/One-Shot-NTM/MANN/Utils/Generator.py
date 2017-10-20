import tensorflow as tf
import numpy as np
import pytest

import os
import random
from Images import get_shuffled_images, time_offset_label, load_transform

class OmniglotGenerator(object):
    """Docstring for OmniglotGenerator"""
    def __init__(self, data_folder, batch_size=1, nb_samples=5, nb_samples_per_class=10, max_rotation=-np.pi/6, max_shift=10, img_size=(20,20), max_iter=None):
        super(OmniglotGenerator, self).__init__()
        self.data_folder = data_folder
        self.batch_size = batch_size
        self.nb_samples = nb_samples
        self.nb_samples_per_class = nb_samples_per_class
        self.max_rotation = max_rotation
        self.max_shift = max_shift
        self.img_size = img_size
        self.max_iter = max_iter
        self.num_iter = 0
        self.character_folders = [os.path.join(self.data_folder, family, character) \
                                  for family in os.listdir(self.data_folder) \
                                  if os.path.isdir(os.path.join(self.data_folder, family)) \
                                  for character in os.listdir(os.path.join(self.data_folder, family))]

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if (self.max_iter is None) or (self.num_iter < self.max_iter):
            self.num_iter += 1
            return (self.num_iter - 1), self.sample(self.nb_samples)
        else:
            raise StopIteration

    def sample(self, nb_samples):
        sampled_character_folders = random.sample(self.character_folders, nb_samples)
        random.shuffle(sampled_character_folders)

        example_inputs = np.zeros((self.batch_size, nb_samples * self.nb_samples_per_class, np.prod(self.img_size)), dtype=np.float32)
        example_outputs = np.zeros((self.batch_size, nb_samples * self.nb_samples_per_class), dtype=np.float32)     #notice hardcoded np.float32 here and above, change it to something else in tf

        for i in range(self.batch_size):
            labels_and_images = get_shuffled_images(sampled_character_folders, range(nb_samples), nb_samples=self.nb_samples_per_class)
	
            sequence_length = len(labels_and_images)
            labels, image_files = zip(*labels_and_images)
            angles = np.random.uniform(-self.max_rotation, self.max_rotation, size=sequence_length)
            shifts = np.random.uniform(-self.max_shift, self.max_shift, size=sequence_length)
	    #print(angles)
            #print(shifts)
            example_inputs[i] = np.asarray([load_transform(filename, angle=angle, s=shift, size=self.img_size).flatten() \
                                            for (filename, angle, shift) in zip(image_files, angles, shifts)], dtype=np.float32)
            example_outputs[i] = np.asarray(labels, dtype=np.int32)

        return example_inputs, example_outputs
