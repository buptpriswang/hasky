#!/usr/bin/env python
# ==============================================================================
#          \file   dump-embedding-word.py
#        \author   chenghuige  
#          \date   2017-08-08 20:38:49.910743
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
#from __future__ import print_function

import sys, os

import numpy as np
import tensorflow as tf

import melt 

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string("model_dir", "./", "")

predictor = melt.Predictor(FLAGS.model_dir)

embedding = predictor.inference('word_embedding')

np.save(os.path.join(FLAGS.model_dir, 'word_embedding.npy'), embedding)
