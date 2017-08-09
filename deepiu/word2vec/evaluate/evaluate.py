#!/usr/bin/env python
# ==============================================================================
#          \file   evaluate.py
#        \author   chenghuige  
#          \date   2017-08-09 15:05:28.979959
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
#from __future__ import print_function

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('word_embedding_file', None, '')

import sys, os
import numpy as np  

embedding = np.load(FLAGS.word_embedding_file)

def sum_embedding()
