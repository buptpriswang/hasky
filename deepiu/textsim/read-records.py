#!/usr/bin/env python
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

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('batch_size', 100, 'Batch size.')
#flags.DEFINE_integer('num_epochs', 1, 'Number of epochs to run trainer.')
flags.DEFINE_integer('num_threads', 12, '')
flags.DEFINE_boolean('batch_join', True, '')
flags.DEFINE_boolean('shuffle_batch', True, '')
flags.DEFINE_boolean('shuffle', True, '')

flags.DEFINE_string('input', '/home/gezi/new/temp/makeup/title2name/tfrecord/seq-basic/fixed_valid/test*', '')
flags.DEFINE_string('name', 'train', 'records name')
flags.DEFINE_boolean('dynamic_batch_length', True, '')
flags.DEFINE_boolean('shuffle_then_decode', True, '')

def read_once(sess, step, ops):
  global max_index
  if not hasattr(read_once, "timer"):
    read_once.timer = Timer()

  ltext_str, ltext, rtext, rtext_str = sess.run(ops)
  
  if step % 100 == 0:
    print('step:', step)
    print('duration:', read_once.timer.elapsed())
    print('name:', ltext_str[0])
    
    print('ltext:', ltext[0])
    print('ltext_str:', ltext_str[0])

    print('rtext:', rtext[0])
    print('rtext_str:', rtext_str[0])
    print('len(rtext_str):', len(rtext_str[0]))


from melt.flow import tf_flow
import input
def read_records():
  inputs, decode, _ = input.get_decodes()

  ops = inputs(
    FLAGS.input,
    decode_fn=decode,
    batch_size=FLAGS.batch_size,
    num_epochs=FLAGS.num_epochs, 
    num_threads=FLAGS.num_threads,
    #num_threads=1,
    batch_join=FLAGS.batch_join,
    shuffle_batch=FLAGS.shuffle_batch,
    shuffle_files=FLAGS.shuffle,
    #fix_random=True,
    #fix_sequence=True,
    #no_random=True,
    allow_smaller_final_batch=True,
    )
  print(ops) 
  
  timer = Timer()
  tf_flow(lambda sess, step: read_once(sess, step, ops))
  print('max_index:', max_index)
  print(timer.elapsed())
    

def main(_):
  read_records()

if __name__ == '__main__':
  tf.app.run()
