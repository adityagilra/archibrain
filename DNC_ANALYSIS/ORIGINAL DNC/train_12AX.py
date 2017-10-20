# MODEL: DNC Model from https://github.com/deepmind/dnc.
# TASK: 12AX Task
#
# Author: Marco Martinolli
# ==============================================================================
"""Example script to train the DNC on 12AX task."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import sonnet as snt
import numpy as np

import dnc
from task_12AX import construct_trial

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
tf.flags.DEFINE_float("learning_rate", 0.02, "Optimizer learning rate.")
tf.flags.DEFINE_float("optimizer_epsilon", 1e-10, "Epsilon used for RMSProp optimizer.")

# Task parameters
tf.flags.DEFINE_integer("batch_size", 1, "Batch size for training.")
tf.flags.DEFINE_integer("perc_target", 0.5, "Probability to have a target pattern in a loop")
tf.flags.DEFINE_integer("min_loops", 1,"Lower limit on number of inner loops per trial")
tf.flags.DEFINE_integer("max_loops", 4,"Upper limit on number of inner loops per trial")

# Training options.
tf.flags.DEFINE_integer("num_trials", 10000, "Dimension of the training dataset")
tf.flags.DEFINE_integer("report_interval", 100,
                        "Iterations between reports (samples, valid loss).")
tf.flags.DEFINE_string("checkpoint_dir", "/tmp/tf/dnc",
                       "Checkpointing directory.")
tf.flags.DEFINE_integer("checkpoint_interval", -1,
                        "Checkpointing step interval.")
tf.flags.DEFINE_string("conv_criterion", 'strong',"Convergence criterion for 12AX task")

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

  input_var = tf.placeholder(shape=(None, FLAGS.batch_size, 8), dtype=tf.float32)
  target_var = tf.placeholder(shape=(None, FLAGS.batch_size, 2), dtype=tf.float32)

  output_logits = run_model(input_var, 2)
  output = tf.round(tf.sigmoid(output_logits))

  train_loss = softmax_cross_entropy(output_logits, target_var)

  # Set up optimizer with global norm clipping.
  trainable_variables = tf.trainable_variables()
  grads, _ = tf.clip_by_global_norm(
      tf.gradients(train_loss[0], trainable_variables), FLAGS.max_grad_norm)

  global_step = tf.get_variable(
      name="global_step",
      shape=[],
      dtype=tf.int64,
      initializer=tf.zeros_initializer(),
      trainable=False,
      collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.GLOBAL_STEP])

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

  CORR = 0
  EP_CORR = 0
  convergence=False
  conv_criterion = FLAGS.conv_criterion

  E = np.zeros(num_trials)

  stop = True

  # Train.
  with tf.train.SingularMonitoredSession(hooks=hooks, checkpoint_dir=FLAGS.checkpoint_dir) as sess:

    start_iteration = sess.run(global_step)
    total_loss = 0

    for train_iteration in xrange(start_iteration, num_trials):

      dataset_inputs, dataset_outputs = construct_trial(FLAGS.perc_target, FLAGS.min_loops, FLAGS.max_loops)     

      obs = np.expand_dims(dataset_inputs,axis=1)
      target = np.expand_dims(dataset_outputs,axis=1)
      feed_dict = {input_var: obs, target_var: target}

      _, loss = sess.run([train_step, train_loss], feed_dict=feed_dict)
      total_loss += loss[0]

      err_tr = np.abs(np.argmax(np.squeeze(target, axis=1), axis=1)-np.argmax(np.squeeze(loss[1],axis=1),axis=1))

      if conv_criterion=='strong':
      
         if (err_tr==0).all():
            CORR += len(err_tr)
         else:
	    CORR = 0
            E[train_iteration]+=np.sum(np.where(err_tr!=0,1,0))

         if CORR >=1000 and convergence==False:
            convergence=True
            conv_iter = train_iteration
            if stop:
                break

      elif conv_criterion=='lenient':
            
         if (err_tr==0).all():
            EP_CORR += 1
         else:
	    EP_CORR = 0           

         if EP_CORR >=50 and convergence==False:
            convergence=True
            conv_iter = train_iteration
            if stop:
                break

      if (train_iteration + 1) % report_interval == 0:

        print('Iteration '+str(train_iteration+1)+': Average Loss = '+str(total_loss / report_interval))
        iters.append(train_iteration) 
        losses.append(total_loss / report_interval)
        total_loss = 0

    print('Converged at ', conv_iter,' (',conv_criterion,')')

    corr_test = 0
    N_test = 1000
    perc = 0
    if N_test!=0:
       for test_iteration in xrange(0, N_test):
          inp, tar = construct_trial(FLAGS.perc_target, FLAGS.min_loops, FLAGS.max_loops)  
          obs = np.expand_dims(inp,axis=1)
          target = np.expand_dims(tar,axis=1)
          feed_dict = {input_var: obs, target_var: target}

          out = sess.run([train_loss], feed_dict=feed_dict)

          err_tr = np.abs(np.argmax(np.squeeze(target, axis=1), axis=1)-np.argmax(np.squeeze(out[0][1],axis=1),axis=1))
          if (err_tr==0).all():
             corr_test += 1

       perc = 100*float(corr_test)/float(N_test)
       print('TEST Percentage of correct trials: ',perc,'%')
 
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

  return loss, soft_out, loss_time_batch

def main(unused_argv):

  tf.logging.set_verbosity(3)  # Print INFO log messages.

  conv_iter, E, perc = train(FLAGS.num_trials, FLAGS.report_interval) 

  str_err = 'results/DNC_long_12AX_error.txt'
  f = open(str_err,'ab')
  E_mean = np.mean(np.reshape(E,(-1,50)), axis=1)
  np.savetxt(f,E_mean)
  f.close()

  str_perc = 'results/DNC_long_12AX_perc.txt'	
  f2 = open(str_perc,'a')
  f2.write(str(perc)+'\n') 
  f2.close()

  str_conv = 'results/DNC_long_12AX_conv.txt'	
  f3 = open(str_conv,'a')
  f3.write(str(conv_iter)+'\n') 
  f3.close()

if __name__ == "__main__":
  tf.app.run()
