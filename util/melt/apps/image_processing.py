#!/usr/bin/env python
# ==============================================================================
#          \file   image_processing.py
#        \author   chenghuige  
#          \date   2017-04-07 08:49:43.118136
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('image_model_name', 'InceptionV3', '')

image_processing_fn = None

import melt
def init():
  global image_processing_fn 
  image_processing_fn = melt.image.create_image2feature_fn(FLAGS.image_model_name)
  return image_processing_fn
