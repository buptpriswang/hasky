#!/usr/bin/env python
# ==============================================================================
#          \file   test.py
#        \author   chenghuige  
#          \date   2017-09-03 08:44:17.214012
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os

import tensorflow as tf 
flags = tf.app.flags 
FLAGS = flags.FLAGS

flags.DEFINE_string('info_dir', None, '')

def main(_):
  print(FLAGS.info_dir)
  
if __name__ == '__main__':
  tf.app.run()
