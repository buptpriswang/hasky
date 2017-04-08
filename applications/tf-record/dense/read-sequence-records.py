#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   read-sequence-records.py
#        \author   chenghuige  
#          \date   2016-09-11 11:46:01.293654
#   \Description  
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

'''
python read-sequence-records.py /tmp/urate.train.seq
'''
import sys, os, time
import tensorflow as tf

from gezi import Timer
import melt 

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('batch_size', 3, 'Batch size.')
#flags.DEFINE_integer('num_epochs', 2, 'Number of epochs to run trainer.')
flags.DEFINE_integer('num_threads', 12, '')
flags.DEFINE_boolean('batch_join', True, '')
flags.DEFINE_boolean('shuffle_files', False, '')
flags.DEFINE_string('buckets', '', '')

#flags.DEFINE_boolean('decode_then_shuffle', True, '')

def read_once(sess, step, ops):
  id, X, y, length = sess.run(ops)
  if not hasattr(read_once, "timer"):
    read_once.timer = Timer()
  if step % 10000 == 0:
    print('duration:', read_once.timer.elapsed())
    #if step == 0:
    print('id', id)
    print(X)
    print(y)
    print(X.shape)
    print('length', length)

num_features = 488

def decode_example(serialized_example):
  features, sequence_features = tf.parse_single_sequence_example(
    serialized_example,
    context_features={
        'id' : tf.FixedLenFeature([], tf.int64),
        'label' : tf.FixedLenFeature([], tf.int64),
        'length' : tf.FixedLenFeature([], tf.int64),
    },
    sequence_features={
       'feature': tf.FixedLenSequenceFeature([], dtype=tf.float32)
       #'feature': tf.FixedLenSequenceFeature([num_features], dtype=tf.float32)
      }
    )

  id = features['id']
  label = features['label']
  length = features['length']

  #must resphae = 488! if not dynamic_pad=True
  #here may be can deal with like @TODO one image, multiple text(click query) we can pack the result here
  #like id,image, feature | id, image, feature1 ...
  
  print(sequence_features)
  #feature = tf.reshape(sequence_features['feature'], [num_features])
  feature = sequence_features['feature']
  print(tf.shape(feature))
  #feature = tf.reshape(sequence_features['feature'], [num_features])
  
  #feature =  tf.reshape(sequence_features['feature'], [-1,])
  #feature = tf.squeeze(sequence_features['feature'])
  #feature = sequence_features['feature']

  #context = tf.contrib.learn.run_n(features, n=1, feed_dict=None)
  #print(context[0])
  #sequence = tf.contrib.learn.run_n(sequence_features, n=1, feed_dict=None)
  #print(sequence[0])

  return id, feature, label, length

from melt.flow import tf_flow
def read_records():
  # Tell TensorFlow that the model will be built into the default Graph.
  #can only use decode_the_shuffle since right now only tf.parse_single_sequence_example,
  #@TODO verify sparse will work, for example seems decode_the_shuffle and shuffle_then_decode
  #all work for both dense and sparse
  inputs = melt.decode_then_shuffle.inputs
  #inputs = melt.shuffle_then_decode.inputs #since only parse_single_sequence_example could not use shuffle_then_decode
  decode = decode_example

  #looks like setting sparse==1 or 0 all ok, but sparse=1 is faster...
  #may be for large example and you only decode a small part features then sparse==0 will be
  #faster since decode before shuffle, shuffle less data
  #but sparse==0 for one flow can deal both sparse input and dense input
  with tf.Graph().as_default():
    id, X, y, length = inputs(
      sys.argv[1], 
      decode=decode,
      batch_size=FLAGS.batch_size,
      num_epochs=FLAGS.num_epochs, 
      num_threads=FLAGS.num_threads,
      dynamic_pad=True,
      batch_join=FLAGS.batch_join,
      bucket_boundaries=[int(x) for x in FLAGS.buckets.split(',') if x],
      length_index=-1)
    
    tf_flow(lambda sess, step: read_once(sess, step, [id, X, y, length]))
    

def main(_):
  read_records()

if __name__ == '__main__':
  tf.app.run()
  
