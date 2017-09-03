﻿#!/usr/bin/env python
# ==============================================================================
#          \file   read-records.py
#        \author   chenghuige  
#          \date   2016-07-19 17:09:07.466651
#   \Description  
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, time
import tensorflow as tf
from collections import defaultdict

import numpy as np

from gezi import Timer
import melt 

from deepiu.util import text2ids

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('vocab', '/home/gezi/new/temp/image-caption/makeup/tfrecord/seq-basic-inceptionResnetV2/vocab.txt', '')
#flags.DEFINE_integer('batch_size', 100, 'Batch size.')
#flags.DEFINE_integer('num_epochs', 1, 'Number of epochs to run trainer.')

#flags.DEFINE_integer('num_threads', 12, '')
#flags.DEFINE_boolean('batch_join', True, '')
#flags.DEFINE_boolean('shuffle_batch', True, '')
#flags.DEFINE_boolean('shuffle_files', True, '')


#TODO use input_flags.py
flags.DEFINE_string('input', '/home/gezi/new/temp/image-caption/makeup/tfrecord/seq-basic-inceptionResnetV2/valid/test*', '')
#flags.DEFINE_string('name', 'train', 'records name')
#flags.DEFINE_boolean('dynamic_batch_length', True, '')
#flags.DEFINE_boolean('shuffle_then_decode', True, '')
#flags.DEFINE_boolean('is_sequence_example', False, '')
#flags.DEFINE_string('decode_name', 'text', 'records name')
#flags.DEFINE_string('decode_str_name', 'text_str', 'records name')
#flags.DEFINE_string('image_feature_name', 'image_feature', 'records name')
#flags.DEFINE_boolean('pre_calc_image_feature', True, '')
#flags.DEFINE_integer('num_negs', 0, '')

max_index = 0
def read_once(sess, step, ops, neg_ops=None):
  global max_index
  if not hasattr(read_once, "timer"):
    read_once.timer = Timer()

  if neg_ops is None:
    image_name, image_feature, text, text_str = sess.run(ops)
  else:
    squeezed_neg_text_str = tf.squeeze(neg_ops[1])
    neg_ops += [squeezed_neg_text_str]
    ops = list(ops)
    ops.extend(neg_ops)
    
    image_name, image_feature, text, text_str, neg_text, neg_text_str, neg_text_str_squeeze = sess.run(ops)
  
  if step % 100 == 0:
    print('step:', step)
    print('duration:', read_once.timer.elapsed())
    print('image_name:', image_name[0])
    print('image_feature_len:', len(image_feature[0]))
    print('text:', text[0])
    print('len(text)', len(text[0]))
    print(text2ids.ids2text(text[0]))
    print('text_str:', text_str[0])
    print('len(text_str):', len(text_str[0]))

  cur_max_index = np.max(text)
  if cur_max_index > max_index:
    max_index = cur_max_index


from melt.flow import tf_flow
import input
def read_records():
  inputs, decode, decode_neg = input.get_decodes()
  #@TODO looks like single thread will be faster, but more threads for better randomness ?
  ops = inputs(
    FLAGS.input,
    decode_fn=decode,
    batch_size=FLAGS.batch_size,
    num_epochs=FLAGS.num_epochs, 
    num_threads=FLAGS.num_threads,
    #num_threads=1,
    batch_join=FLAGS.batch_join,
    shuffle_batch=FLAGS.shuffle_batch,
    shuffle_files=FLAGS.shuffle_files,
    #fix_random=True,
    #fix_sequence=True,
    #no_random=True,
    allow_smaller_final_batch=True,
    )
  print(ops) 
  
  FLAGS.num_negs = 0
  neg_ops = None  
  if FLAGS.num_negs:
    neg_ops = inputs(
      FLAGS.input,
      decode_fn=decode_neg,
      batch_size=FLAGS.batch_size * FLAGS.num_negs,
      num_epochs=FLAGS.num_epochs, 
      num_threads=FLAGS.num_threads,
      batch_join=FLAGS.batch_join,
      shuffle_files=FLAGS.shuffle_files)
    neg_ops = input.reshape_neg_tensors(neg_ops, FLAGS.batch_size, FLAGS.num_negs)

    neg_ops = list(neg_ops)

  timer = Timer()
  tf_flow(lambda sess, step: read_once(sess, step, ops, neg_ops))
  print('max_index:', max_index)
  print(timer.elapsed())
    

def main(_):
  text2ids.init()
  read_records()

if __name__ == '__main__':
  tf.app.run()
