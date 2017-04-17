#!/usr/bin/env python
# ==============================================================================
#          \file   image_model.py
#        \author   chenghuige  
#          \date   2017-04-10 19:58:46.031602
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import melt
  
class ImageModel(object):
  def __init__(self, 
               image_checkpoint_file,
               model_name='InceptionV3', 
               height=299, 
               width=299,
               image_format='jpeg'):
    self.sess = tf.Session()
    self.images_feed =  tf.placeholder(tf.string, [None,], name='images')
    self.img2feautres_op = self._build_graph(model_name, height, width, image_format=image_format)

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    self.sess.run(init_op)
    #---load inception model check point file
    init_fn = melt.image.create_image_model_init_fn(model_name, image_checkpoint_file)
    init_fn(self.sess)


  def _build_graph(self, model_name, height, width, image_format='jpeg'):
    melt.apps.image_processing.init(model_name)
    return melt.apps.image_processing.image_processing_fn(self.images_feed,  
                                                          height=height, 
                                                          width=width,
                                                          image_format=image_format)

  def process(self, images):
    return self.sess.run(self.img2feautres_op, feed_dict={self.images_feed: images})
