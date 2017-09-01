#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   evaluate-sim-score.py
#        \author   chenghuige  
#          \date   2016-09-25 00:46:53.890615
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('valid_resource_dir_', '/home/gezi/new/temp/image-caption/makeup/tfrecord/seq-basic/valid/', '')
flags.DEFINE_string('model_dir', '/home/gezi/new/temp/image-caption/makeup/model/bow/', '')  
flags.DEFINE_string('vocab', '/home/gezi/new/temp/image-caption/makeup/tfrecord/seq-basic/vocab.txt', 'vocabulary binary file')
flags.DEFINE_boolean('print_predict', False, '')
flags.DEFINE_boolean('random', True, '')

import sys 
import numpy as np
import melt
logging = melt.logging
import gezi

from deepiu.util import evaluator

def evaluate_score():
  FLAGS.valid_resource_dir = FLAGS.valid_resource_dir_
  evaluator.init()
  predictor = melt.SimPredictor(FLAGS.model_dir)
  evaluator.evaluate_scores(predictor, random=FLAGS.random)

def main(_):
  logging.init(logtostderr=True, logtofile=False)
  evaluate_score()

if __name__ == '__main__':
  tf.app.run()
