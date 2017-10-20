# MODEL: R-DNC Model. 
# The model is a simplifed version of the DNC model from https://github.com/deepmind/dnc, where:
#	- the addressing scheme is purely content-based, both for reading and writing (no allocation-based technique or temporal linkage)
#	- the addressing technique presents the additional option to select the LRUA module to memory for classification tasks (from A. Santoro, 'One-shot Learning with Memory-Augmented Neural Networks', 2016). The user can deactivate this option via the LRUA flag (default is True).
#	- the model scheme has the option to take as input either the new stimulus at each timestep (Vineet J. version) or its concatenation with previous read vectors (as in the original paper). The user can activate or deactivate concatenation using the conc_opt flag (default is True). 
#  
# TASK: image classification task on Omniglot test dataset
#
# N.B. Being a test dataset, the image characters are taken from a disjoint folder of alphabets not seen during training and images are not augmented (neither rotated nor shifted).
#
# Author: Marco Martinolli
# ==============================================================================
"""Example script to train the DNC on a repeated copy task."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import sonnet as snt
import numpy as np

import dnc

from MANN.Utils.Generator import OmniglotGenerator
from MANN.Utils.Metrics import accuracy_instance

from matplotlib import pyplot as plt
import pickle

np.set_printoptions(precision=3)

FLAGS = tf.flags.FLAGS

# Model parameters
tf.flags.DEFINE_integer("hidden_size", 200, "Size of LSTM hidden layer.")
tf.flags.DEFINE_integer("memory_size", 128, "The number of memory slots.")
tf.flags.DEFINE_integer("word_size", 40, "The width of each memory slot.")
tf.flags.DEFINE_integer("num_write_heads", 4, "Number of memory write heads.")
tf.flags.DEFINE_integer("num_read_heads", 4, "Number of memory read heads.")
tf.flags.DEFINE_boolean("LRUA",True,"Adoption of LRUA addressing scheme for one-shot learning")
tf.flags.DEFINE_boolean("conc_opt",True,"Concatenation of the input with previous read vectors")
tf.flags.DEFINE_integer("clip_value", 50, "Maximum absolute value of controller and dnc outputs.")

# Optimizer parameters.
tf.flags.DEFINE_float("max_grad_norm", 50, "Gradient clipping norm limit.")
tf.flags.DEFINE_float("learning_rate", 0, "Optimizer learning rate.")
tf.flags.DEFINE_float("optimizer_epsilon", 1e-10, "Epsilon used for RMSProp optimizer.")

# Task parameters
tf.flags.DEFINE_integer("batch_size", 16, "Batch size for training.")
tf.flags.DEFINE_integer("nb_class", 5, "Number of classes per episode")
tf.flags.DEFINE_integer("input_size", 20*20, "Sample image size")
tf.flags.DEFINE_integer("nb_samples_per_class", 10, "Number of samples per class")
tf.flags.DEFINE_float("rot_max", 0, "Maximum absolute rotation of sample images.")
tf.flags.DEFINE_float("shift_max", 0, "Maximum absolute shift of sample images.")

# Training options.
tf.flags.DEFINE_integer("num_episodes", 100, "Dimension of the training dataset")
tf.flags.DEFINE_integer("report_interval", 1,
                        "Iterations between reports (samples, valid loss).")
tf.flags.DEFINE_string("checkpoint_dir", "check_dir",
                       "Checkpointing directory.")
tf.flags.DEFINE_integer("checkpoint_interval", 0,
                        "Checkpointing step interval.")


def run_model(input_sequence, output_size):
  """Runs model on input sequence."""

  access_config = {
      "memory_size": FLAGS.memory_size,
      "word_size": FLAGS.word_size,
      "num_reads": FLAGS.num_read_heads,
      "num_writes": FLAGS.num_write_heads,
      "lrua_cond": FLAGS.LRUA
  }
  controller_config = {
      "hidden_size": FLAGS.hidden_size,
  }
  clip_value = FLAGS.clip_value

  dnc_core = dnc.DNC(access_config, controller_config, output_size, clip_value, FLAGS.conc_out)
  initial_state = dnc_core.initial_state(FLAGS.batch_size)
  output_sequence, state = tf.nn.dynamic_rnn(
      cell=dnc_core,
      inputs=input_sequence,
      time_major=True,
      initial_state=initial_state)

  return output_sequence, state


def test(num_trials, report_interval):
  """Trains the DNC and periodically reports the loss."""

  input_var = tf.placeholder(shape=(None, FLAGS.batch_size, FLAGS.input_size), dtype=tf.float32)
  target_var = tf.placeholder(shape=(None, FLAGS.batch_size), dtype=tf.int32)
  target_var_mod = tf.one_hot(indices=target_var, depth=FLAGS.nb_class)
  zero_first = tf.zeros(shape=[1,FLAGS.batch_size,FLAGS.nb_class], dtype=tf.float32)
  target_shift = tf.concat(values=[zero_first, target_var_mod[:-1,:,:]],axis=0)
  input_var_mod = tf.concat(values=[input_var, target_shift], axis=2)

  output_logits, _ = run_model(input_var_mod, FLAGS.nb_class)
  #output_var = tf.round(tf.sigmoid(output_logits))
  #print(state)

  generator = OmniglotGenerator(data_folder='./data/omniglot_evaluation', batch_size=FLAGS.batch_size, nb_samples=FLAGS.nb_class, nb_samples_per_class=FLAGS.nb_samples_per_class, max_rotation=FLAGS.rot_max, max_shift=FLAGS.shift_max, max_iter=FLAGS.num_episodes)  

  train_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_logits, labels=target_var_mod))

  # Set up optimizer with global norm clipping.
  trainable_variables = tf.trainable_variables()


  global_step = tf.get_variable(
      name="global_step",
      shape=[],
      dtype=tf.int64,
      initializer=tf.zeros_initializer(),
      trainable=False,
      collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.GLOBAL_STEP])

  accuracies = accuracy_instance(tf.argmax(output_logits, axis=2), target_var, nb_classes=FLAGS.nb_class, nb_samples_per_class=FLAGS.nb_samples_per_class, batch_size=generator.batch_size)
  sum_out = tf.reduce_sum(tf.reshape(tf.one_hot(tf.argmax(output_logits, axis=2), depth=generator.nb_samples), (-1, generator.nb_samples)), axis=0)
  #show_memory = state.access_state.memory

  saver = tf.train.Saver()

  if FLAGS.checkpoint_interval > 0:
    hooks = [
        tf.train.CheckpointSaverHook(
            checkpoint_dir=FLAGS.checkpoint_dir,
            save_steps=FLAGS.checkpoint_interval,
            saver=saver)
    ]
  else:
    hooks = []

  losses = []
  iters = []

  # Train.
  with tf.train.SingularMonitoredSession(hooks=hooks, checkpoint_dir=FLAGS.checkpoint_dir) as sess:
    
    print('Omniglot Task - R-DNC+LRUA - conc_input -',FLAGS.hidden_size,'hidden - ',FLAGS.learning_rate,'lr')
    start_iteration = sess.run(global_step)
    accs = np.zeros(generator.nb_samples_per_class)
    accs_final = np.zeros(generator.nb_samples_per_class)
    losses = []

    for i, (batch_input, batch_output) in generator:

      #print(batch_input[0,0,:])
      feed_dict = {input_var: batch_input, target_var: batch_output}

      loss, acc, lab_distr = sess.run([train_loss, accuracies, sum_out], feed_dict=feed_dict)

      accs += acc[0:FLAGS.nb_samples_per_class]
      accs_final += acc[0:FLAGS.nb_samples_per_class]
      losses.append(loss)
      if (i+1) % report_interval == 0:
         print('\nEpisode %05d: %.6f' % (i+1, np.mean(losses)) )
	 print('Labels Distribution: ', lab_distr)
         print('Accuracies:')
         print(accs/report_interval)
         losses, accs = [], np.zeros(generator.nb_samples_per_class)


    accs_final/=num_trials
    print('\n\nMEAN TEST PERFORMANCE: ')
    print(accs_final)
	
def main(unused_argv):
  tf.logging.set_verbosity(3)  # Print INFO log messages.
  test(FLAGS.num_episodes, FLAGS.report_interval)


if __name__ == "__main__":
  tf.app.run()
