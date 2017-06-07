#!/usr/bin/env python
# ==============================================================================
#          \file   train.py
#        \author   chenghuige  
#          \date   2017-05-16 16:37:24.006002
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf 
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('batch_size', 1024, '')

import numpy as np

import melt 
logging = melt.logging
import gezi

from pointer import Pointer

from util import *

p = Pointer()

loss = p.build()

ops = [loss]

eval_ops = [loss]

train_feed_dict = {}
test_feed_dict = {}

def gen_feed_dict(inputs, actual_index_dists, segment_lengths, max_length=60, batch_size=1024):
  global train_feed_dict
  if train_feed_dict:
    return train_feed_dict 
  feed_dict = {}
  sequences = []
  first_indexes = []
  second_indexes = []
  input_length = max_length
  for batch_index in xrange(batch_size):
    data = generate_nested_sequence(max_length,
                                    segment_lengths[0],
                                    segment_lengths[1])
    sequences.append(data[0])                                           # J
    first_indexes.append(create_one_hot(input_length, data[1]))         # J
    second_indexes.append(create_one_hot(input_length, data[2]))        # J
    
    feed_dict[inputs] = np.stack(sequences)                                # B x J
    feed_dict[actual_index_dists] = np.stack([np.stack(first_indexes),     # I x B x J
                                               np.stack(second_indexes)])
  return feed_dict

def main(_):
  print('learning_rate', FLAGS.learning_rate)
  global train_feed_dict, test_feed_dict
  train_feed_dict = gen_feed_dict(p.inputs, p.actual_index_dists, (11, 20))
  test_feed_dict = gen_feed_dict(p.inputs, p.actual_index_dists, (6, 10))

  logging.set_logging_path(gezi.get_dir('/home/gezi/temp/pointer.model'))
  melt.apps.train_flow(ops, 
                       eval_ops=eval_ops,
                       gen_feed_dict_fn=lambda: train_feed_dict,
                       gen_eval_feed_dict_fn=lambda: test_feed_dict)

if __name__ == '__main__':
  tf.app.run()