#!/usr/bin/env python
# ==============================================================================
#          \file   read-records.py
#        \author   chenghuige  
#          \date   2016-07-19 17:09:07.466651
#   \Description   @TODO why not shuffle will be much slower then shuffle..
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os, time
import tensorflow as tf

from gezi import Timer
import melt 

import functools

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('batch_size', 5, 'Batch size.')
#flags.DEFINE_integer('num_epochs', 2, 'Number of epochs to run trainer.')
flags.DEFINE_integer('num_threads', 12, '')
flags.DEFINE_boolean('batch_join', True, '')
flags.DEFINE_boolean('shuffle', True, '')
flags.DEFINE_string('label_type', 'int', '')

flags.DEFINE_integer('num_test_steps', 10000, '')
flags.DEFINE_boolean('shuffle_then_decode', True, '')
flags.DEFINE_boolean('dynamic_pad', False, '')

def read_once(sess, step, ops):
  X, y = sess.run(ops)
  if not hasattr(read_once, "timer"):
    read_once.timer = Timer()
  if step % FLAGS.num_test_steps == 0:
    duration = read_once.timer.elapsed()
    print('steps:', step, ' duration:', duration, 'instance/s:',  FLAGS.batch_size * FLAGS.num_test_steps / duration)

    if step == 0:
      print(X)
      print(y)

from melt.flow import tf_flow

#-----test below will not work, sparse data should only use shuffle then decode
def decode_example(batch_serialized_examples, label_type=tf.int64, index_type=tf.int64, value_type=tf.float32):
  """
  decode batch_serialized_examples for use in parse libsvm fomrat sparse tf-record
  Returns:
  X,y
  """
  features = tf.parse_single_example(
      batch_serialized_examples,
      features={
          'label' : tf.FixedLenFeature([], label_type),
          'index' : tf.VarLenFeature(index_type),
          'value' : tf.VarLenFeature(value_type),
      })

  label = features['label']
  index = features['index']
  value = features['value']

  #return as X,y
  print(index, value)
  return (index, value), label

def read_records():
  # Tell TensorFlow that the model will be built into the default Graph.
  label_type = tf.int64 if FLAGS.label_type == 'int' else tf.float32
  if FLAGS.shuffle_then_decode:
    inputs = melt.shuffle_then_decode.inputs
    decode = functools.partial(melt.libsvm_decode.decode, label_type=label_type)
  else:
    inputs = melt.decode_then_shuffle.inputs
    decode = functools.partial(decode_example, label_type=label_type)

  with tf.Graph().as_default():
    X, y = inputs(
      sys.argv[1], 
      decode=decode,
      batch_size=FLAGS.batch_size,
      num_epochs=FLAGS.num_epochs, 
      num_threads=FLAGS.num_threads,
      batch_join=FLAGS.batch_join,
      shuffle=FLAGS.shuffle,
      dynamic_pad=FLAGS.dynamic_pad) #here is just test dynamic_pad not work for sparse if use decode_then_shuffle
    tf_flow(lambda sess, step: read_once(sess, step, [X, y]))
    

def main(_):
  read_records()


if __name__ == '__main__':
  tf.app.run()
