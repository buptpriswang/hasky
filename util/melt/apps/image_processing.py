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
from melt import logging

def init(image_model_name='InceptionResnetV2', im2text_prcocessing=False):
  global image_processing_fn 
  if image_processing_fn is None:
    if im2text_prcocessing:
  	  image_processing_fn = melt.image.image_processing.create_image2feature_fn(image_model_name)
    else:
		  image_processing_fn = melt.image.image_processing.create_image2feature_slim_fn(image_model_name)
  return image_processing_fn
