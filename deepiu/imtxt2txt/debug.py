#!/usr/bin/env python
# ==============================================================================
#          \file   test-melt.py
#        \author   chenghuige  
#          \date   2016-08-17 15:34:45.456980
#   \Description  
# ==============================================================================
"""
train_input from input_app now te test_input 
valid_input set empty
python ./test.py --train_input test_input --valid_input ''
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('algo', 'imtxt2txt', '')
flags.DEFINE_string('model_dir', '/home/gezi/new/temp/imtxt_keyword/model/imtxt2txt.fixme/', '')  
flags.DEFINE_string('vocab', '/home/gezi/new/temp/imtxt_keyword/tfrecord/seq-basic/vocab.txt', '')

import sys

import melt

import input_app as InputApp
logging = melt.utils.logging

from deepiu.util import algos_factory

def read_once(sess, step):
  ops = [tf.get_collection('fixed_text')[-1], tf.get_collection('eval_text')[-1], 
         tf.get_collection('sequence')[-1], tf.get_collection('sequence_length')[-1], 
         tf.get_collection('outputs')[-1]]
  names = ['fixed_text', 'eval_text', 'sequence', 'sequence_length', 'outputs']
  results = sess.run(ops)
  
  for x, name in zip(results, names):
    print(x.shape, name)
    print(x)

  print([x.shape for x in results if not isinstance(x, (list, tuple))])
   
def test():
  trainer = algos_factory.gen_tranier(FLAGS.algo)
  input_app = InputApp.InputApp()
  sess = input_app.sess = tf.InteractiveSession()

  #init_op = tf.group(tf.global_variables_initializer(),
  #                  tf.local_variables_initializer())
  #sess.run(init_op)

  input_results = input_app.gen_input()
 
  #--------train
  image_name, image_feature, text, text_str, input_text, input_text_str = input_results[input_app.input_valid_name]

  #How to get ride of main scope, not use it ? use make_template to make auto share
  with tf.variable_scope(FLAGS.main_scope):
    loss = trainer.build_train_graph(image_feature, input_text, text)

  melt.flow.tf_flow(read_once, num_steps=1)

def main(_):
  logging.init(logtostderr=True, logtofile=False)
  global_scope = ''
  if FLAGS.add_global_scope:
    global_scope = FLAGS.global_scope if FLAGS.global_scope else FLAGS.algo
  with tf.variable_scope(global_scope):
    test()

if __name__ == '__main__':
  tf.app.run()

  
