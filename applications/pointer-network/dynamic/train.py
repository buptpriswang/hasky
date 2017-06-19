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

flags.DEFINE_integer('batch_size', 32, 'Batch size.  ')
flags.DEFINE_integer('max_steps', 10, 'Number of numbers to sort.  ')
flags.DEFINE_integer('rnn_size', 32, 'RNN size.  ')
flags.DEFINE_boolean('train_feed_prev', True, '')
flags.DEFINE_boolean('test_feed_prev', True, 'set to False just for experiment purpose')
flags.DEFINE_boolean('fully_attention', False, '')
flags.DEFINE_string('model_dir', '/home/gezi/temp/pointer.static.model', '')

import numpy as np

import melt 
logging = melt.logging
import gezi

from pointer_network import PointerNetwork

from dataset import DataGenerator

data_generator = DataGenerator()

eval_names = ['eval_loss', 'correct_predict_ratio']

with tf.variable_scope('main') as scope:
  pointer = PointerNetwork(FLAGS.max_steps, 
                           FLAGS.batch_size, 
                           FLAGS.rnn_size,
                           fully_attention=FLAGS.fully_attention)
  loss, _ , _, _ = pointer.build(feed_prev=FLAGS.train_feed_prev)
  scope.reuse_variables()
  #eval_loss, correct_predict_ratio, predicts, targets = pointer.build(feed_prev=True)
  eval_loss, correct_predict_ratio, predicts, targets = pointer.build(feed_prev=FLAGS.test_feed_prev)

ops = [loss]
eval_ops = [eval_loss, correct_predict_ratio, predicts, targets]

def deal_eval_results(results):
  melt.print_results(results, eval_names)
  correct_predict_ratio, predicts, targets = results[-3], results[-2], results[-1]
  num_show = 2
  for i, (predict, target) in enumerate(zip(predicts, targets)):
    if i < num_show:
      print('label--:', target)
      print('predict:', predict)

def gen_feed_dict():
   encoder_inputs, decoder_inputs, decoder_targets = data_generator.next_batch(
     FLAGS.batch_size, FLAGS.max_steps)
   return pointer.create_feed_dict(encoder_inputs, decoder_inputs, decoder_targets)


def main(_):
  print('learning_rate', FLAGS.learning_rate)

  logging.set_logging_path(gezi.get_dir(FLAGS.model_dir))

  melt.apps.train_flow(ops, 
                       eval_ops=eval_ops,
                       eval_names=eval_names,
                       gen_feed_dict_fn=gen_feed_dict,
                       gen_eval_feed_dict_fn=gen_feed_dict,
                       deal_eval_results_fn=deal_eval_results,
                       model_dir=FLAGS.model_dir)

if __name__ == '__main__':
  tf.app.run()
