# MODEL: DNC Model from https://github.com/deepmind/dnc.
# TASK: Saccade-AntiSaccade (SAS) 
#
# Author: Marco Martinolli
# ==============================================================================
"""Example script to train the DNC on SAS task."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import sonnet as snt
import numpy as np

import dnc
from task_SAS import construct_trial

from matplotlib import pyplot as plt
import pickle

FLAGS = tf.flags.FLAGS

# Model parameters
tf.flags.DEFINE_integer("hidden_size", 16, "Size of LSTM hidden layer.")
tf.flags.DEFINE_integer("memory_size", 16, "The number of memory slots.")
tf.flags.DEFINE_integer("word_size", 16, "The width of each memory slot.")
tf.flags.DEFINE_integer("num_write_heads", 1, "Number of memory write heads.")
tf.flags.DEFINE_integer("num_read_heads", 4, "Number of memory read heads.")
tf.flags.DEFINE_integer("clip_value", 20, "Maximum absolute value of controller and dnc outputs.")

# Optimizer parameters.
tf.flags.DEFINE_float("max_grad_norm", 50, "Gradient clipping norm limit.")
tf.flags.DEFINE_float("learning_rate", 0.05, "Optimizer learning rate.")
tf.flags.DEFINE_float("optimizer_epsilon", 1e-10, "Epsilon used for RMSProp optimizer.")

# Task parameters
tf.flags.DEFINE_integer("batch_size", 1, "Batch size for training.")

# Training options.
tf.flags.DEFINE_integer("num_trials", 25000, "Dimension of the training dataset")
tf.flags.DEFINE_integer("report_interval", 100,
                        "Iterations between reports (samples, valid loss).")
tf.flags.DEFINE_string("checkpoint_dir", "/tmp/tf/dnc",
                       "Checkpointing directory.")
tf.flags.DEFINE_integer("checkpoint_interval", -1,
                        "Checkpointing step interval.")

def run_model(input_sequence, output_size):
  """Runs model on input sequence."""

  access_config = {
      "memory_size": FLAGS.memory_size,
      "word_size": FLAGS.word_size,
      "num_reads": FLAGS.num_read_heads,
      "num_writes": FLAGS.num_write_heads,
  }
  controller_config = {
      "hidden_size": FLAGS.hidden_size,
  }
  clip_value = FLAGS.clip_value

  dnc_core = dnc.DNC(access_config, controller_config, output_size, clip_value)
  initial_state = dnc_core.initial_state(FLAGS.batch_size)
  output_sequence, _ = tf.nn.dynamic_rnn(
      cell=dnc_core,
      inputs=input_sequence,
      time_major=True,
      initial_state=initial_state)

  return output_sequence


def train(num_trials, report_interval):
  """Trains the DNC and periodically reports the loss."""

  input_var = tf.placeholder(shape=(None, FLAGS.batch_size, 5), dtype=tf.float32)
  target_var = tf.placeholder(shape=(None, FLAGS.batch_size, 3), dtype=tf.float32)

  output_logits = run_model(input_var, 3)
  output = tf.round(tf.sigmoid(output_logits))

  train_loss, feedback = softmax_cross_entropy(output_logits, target_var)

  # Set up optimizer with global norm clipping.
  trainable_variables = tf.trainable_variables()
  grads, _ = tf.clip_by_global_norm(
      tf.gradients(train_loss, trainable_variables), FLAGS.max_grad_norm)

  global_step = tf.get_variable(
      name="global_step",
      shape=[],
      dtype=tf.int64,
      initializer=tf.zeros_initializer(),
      trainable=False,
      collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.GLOBAL_STEP],use_resource=True)

  optimizer = tf.train.RMSPropOptimizer(
      FLAGS.learning_rate, epsilon=FLAGS.optimizer_epsilon)
  train_step = optimizer.apply_gradients(
      zip(grads, trainable_variables), global_step=global_step)

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

  convergence=False

  # Train.
  with tf.train.SingularMonitoredSession(hooks=hooks, checkpoint_dir=FLAGS.checkpoint_dir) as sess:

    start_iteration = sess.run(global_step)
   
    total_loss = 0

    num_PL=0
    num_AL=0
    num_PR=0
    num_AR=0
    trial_AL=np.zeros((50))
    trial_AR=np.zeros((50))
    trial_PL=np.zeros((50))
    trial_PR=np.zeros((50))
    prop_PL=0
    prop_PR=0
    prop_AL=0
    prop_AR=0  

    conv_iter=0 
    stop = True 

    E = np.zeros(num_trials)

    for train_iteration in xrange(start_iteration, num_trials):

      inp, tar, trial_type = construct_trial()     

      obs = np.expand_dims(inp,axis=1)
      target = np.expand_dims(tar,axis=1)
      feed_dict = {input_var: obs, target_var: target}

      _, loss, fb = sess.run([train_step, train_loss, feedback], feed_dict=feed_dict)
      total_loss += loss

      if fb==0:
           E[train_iteration] += 1	

      if trial_type==0:
           num_PL += 1
           if fb:
               trial_PL[(num_PL-1) % 50] = 1
           else:
               trial_PL[(num_PL-1) % 50] = 0
           prop_PL = np.mean(trial_PL)

      elif trial_type==1:
           num_PR += 1
           if fb:
               trial_PR[(num_PR-1) % 50] = 1
           else:
               trial_PR[(num_PR-1) % 50] = 0
           prop_PR = np.mean(trial_PR)
      
      elif trial_type==2:
           num_AL += 1
           if fb:
               trial_AL[(num_AL-1) % 50] = 1
           else:
               trial_AL[(num_AL-1) % 50] = 0
           prop_AL = np.mean(trial_AL)
      
      elif trial_type==3:
           num_AR += 1
           if fb:
               trial_AR[(num_AR-1) % 50] = 1
           else:
               trial_AR[(num_AR-1) % 50] = 0
           prop_AR = np.mean(trial_AR)
      
      if convergence==False and prop_PL>=0.9 and prop_PR>=0.9 and prop_AL>=0.9 and prop_AR>=0.9:
           convergence = True
           conv_iter = np.array([train_iteration])
	   print('SIMULATION CONVERGED AT TRIAL ', train_iteration)
	   if stop:
           	break  			

      if (train_iteration + 1) % report_interval == 0:

        print('Iteration '+str(train_iteration+1)+': Average Loss = '+str(total_loss / report_interval))
        print('\t  PL:',prop_PL,'  PR:',prop_PR,'  AL:',prop_AL,'  AR:',prop_AR)
        iters.append(train_iteration) 
        losses.append(total_loss / report_interval)
        total_loss = 0


    corr_test = 0
    N_test = 100
    for test_iteration in xrange(0, N_test):
       inp, tar, _ = construct_trial()     
       obs = np.expand_dims(inp,axis=1)
       target = np.expand_dims(tar,axis=1)
       feed_dict = {input_var: obs, target_var: target}

       fb = sess.run([feedback], feed_dict=feed_dict)
       if fb:
	  corr_test += 1
    perc = 100*float(corr_test)/float(N_test)
    print('TEST Percentage ',perc,'%')
 
    return conv_iter, E, perc

def sigmoid_cross_entropy(logits, target, log_prob_in_bits=False):

  xent = tf.nn.sigmoid_cross_entropy_with_logits(labels=target, logits=logits)
  loss_time_batch = tf.reduce_sum(xent, axis=2)
  loss_batch = tf.reduce_sum(loss_time_batch, axis=0)

  batch_size = tf.cast(tf.shape(logits)[1], dtype=loss_time_batch.dtype)

  loss = tf.reduce_sum(loss_batch) / batch_size
  if log_prob_in_bits:
    loss /= tf.log(2.)

  return loss

def softmax_cross_entropy(logits, target, log_prob_in_bits=False):

  soft_out = tf.nn.softmax(logits=logits)

  loss_time_batch = tf.nn.softmax_cross_entropy_with_logits(labels=target, logits=logits)
  loss_batch = tf.reduce_sum(loss_time_batch, axis=0)

  batch_size = tf.cast(tf.shape(logits)[1], dtype=loss_time_batch.dtype)

  loss = tf.reduce_sum(loss_batch) / batch_size
  if log_prob_in_bits:
    loss /= tf.log(2.)

  feedback =  tf.equal(tf.argmax(logits,axis=2),tf.argmax(target,axis=2))

  return loss, feedback[-1,:]


def main(unused_argv):
  tf.logging.set_verbosity(3)  # Print INFO log messages.

  conv_iter, E, perc = train(FLAGS.num_trials, FLAGS.report_interval) 

  str_err = 'results/DNC_long_saccade_error.txt'
  f = open(str_err,'ab')
  E_mean = np.mean(np.reshape(E,(-1,50)), axis=1)
  np.savetxt(f,E_mean) 
  f.close()

  str_perc = 'results/DNC_long_saccade_perc.txt'	
  f2 = open(str_perc,'a')
  f2.write(str(perc)+'\n') 
  f2.close()

  str_conv = 'results/DNC_long_saccade_conv.txt'	
  f3 = open(str_conv,'a')
  f3.write(str(conv_iter[0])+'\n') 
  f3.close()

if __name__ == "__main__":
  tf.app.run()
