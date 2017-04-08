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

'''
python ./read-records-melt.py input
'''
import sys, os, time
import tensorflow as tf

from gezi import Timer
import melt 

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('batch_size', 5, 'Batch size.')
flags.DEFINE_integer('max_batch_length', 100, 'max batch length.')
#flags.DEFINE_integer('num_epochs', 1, 'Number of epochs to run trainer.')
flags.DEFINE_integer('num_threads', 1, '')
flags.DEFINE_boolean('batch_join', False, '')
flags.DEFINE_boolean('shuffle', False, '')

flags.DEFINE_boolean('shuffle_then_decode', True, '')

def read_once(sess, step, ops):
  label_, index_, index2_, index3_, value_ = sess.run(ops) 
  if not hasattr(read_once, "timer"):
    read_once.timer = Timer()
  if step < 5:
    print('step:', step)
    print(index2_)
    print(index2_.shape)
    print(index_)
    print(index3_)
    print(index3_.shape)


def decode_examples(batch_serialized_examples):
  features = tf.parse_example(
      batch_serialized_examples,
      features={
          'label' : tf.FixedLenFeature([], tf.int64),
          'index' : tf.VarLenFeature(tf.int64),
          'value' : tf.VarLenFeature(tf.float32),
      })

  label = features['label']
  index = features['index']
  index2 = tf.sparse_tensor_to_dense(index)
  index3 = tf.sparse_to_dense(index.indices, [index.dense_shape[0], FLAGS.max_batch_length], index.values)
  value = features['value']

  return label, index, index2, index3, value

def decode_example(serialized_example):
  features = tf.parse_single_example(
    serialized_example,
      features={
          'label' : tf.FixedLenFeature([], tf.int64),
          'index' : tf.VarLenFeature(tf.int64),
          'value' : tf.VarLenFeature(tf.float32),
      })

  label = features['label']
  index = features['index']
  index2 = tf.sparse_tensor_to_dense(index)
  index3 = tf.sparse_to_dense(index.indices, [index.dense_shape[0], FLAGS.max_batch_length], index.values)
  value = features['value']

  return label, index, index2, index3, value

from melt.flow import tf_flow
def read_records():
  # Tell TensorFlow that the model will be built into the default Graph.
  if FLAGS.shuffle_then_decode:
    inputs = melt.shuffle_then_decode.inputs
    decode_fn = decode_examples
  else:
    inputs = melt.decode_then_shuffle.inputs
    decode_fn = decode_example

  print('shuffle_then_decode?:', FLAGS.shuffle_then_decode)

  #looks like setting sparse==1 or 0 all ok, but sparse=1 is faster...
  #may be for large example and you only decode a small part features then sparse==0 will be
  #faster since decode before shuffle, shuffle less data
  #but sparse==0 for one flow can deal both sparse input and dense input

  #well, may be earlier tf version work with both shuffle_then_decode 1 or 0
  #now only work shuffle_then_decode for data with sparse ... if set 0
  #ValueError: All shapes must be fully defined: [TensorShape([]), TensorShape([Dimension(1)]), TensorShape([Dimension(None)]), TensorShape([Dimension(None), Dimension(100)]), TensorShape([Dimension(1)])]
  #since shuffle_then_decode is faster prefer to use it  ok!

  
  with tf.Graph().as_default():
    ops = inputs(
      sys.argv[1], 
      decode_fn=decode_fn,
      batch_size=FLAGS.batch_size,
      num_epochs=FLAGS.num_epochs, 
      num_threads=FLAGS.num_threads,
      batch_join=FLAGS.batch_join,
      shuffle_files=FLAGS.shuffle,
      fix_random=True)
    
    tf_flow(lambda sess, step: read_once(sess, step, ops))
    

def main(_):
  read_records()

if __name__ == '__main__':
  tf.app.run()
