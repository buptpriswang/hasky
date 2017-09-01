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

image_processing_fn = None

import melt
def init(image_model_name='InceptionV3', slim_preprocessing=True):
  global image_processing_fn 
  if not slim_preprocessing:
  	image_processing_fn = melt.image.create_image2feature_fn(image_model_name)
  else:
		image_processing_fn = melt.image.create_image2feature_slim_fn(image_model_name)
  return image_processing_fn
