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

import os, math

import tensorflow as tf

import numpy as np
  
import melt

class ImageModel(object):
  def __init__(self, 
               image_checkpoint_file,
               model_name='InceptionResnetV2', 
               height=299, 
               width=299,
               feature_name=None,
               image_format='jpeg',
               sess=None,
               graph=None):
    self.graph = tf.Graph() if graph is None else graph
    self.sess = melt.gen_session(graph=self.graph) if sess is None else sess
    self.feature_name = feature_name
    self.model_name = model_name
    with self.sess.graph.as_default():
      self.images_feed =  tf.placeholder(tf.string, [None,], name='images')
      self.img2feautres_op = self._build_graph(model_name, height, width, image_format=image_format)
      self.img2feautres_op2 = self._build_graph2(model_name, height, width, image_format=image_format)

      init_op = tf.group(tf.global_variables_initializer(),
                         tf.local_variables_initializer())
      self.sess.run(init_op)
      #---load inception model check point file
      init_fn = melt.image.image_processing.create_image_model_init_fn(model_name, image_checkpoint_file)
      init_fn(self.sess)

  def _build_graph(self, model_name, height, width, image_format=None):
    melt.apps.image_processing.init(model_name, self.feature_name)
    return melt.apps.image_processing.image_processing_fn(self.images_feed,  
                                                          height=height, 
                                                          width=width,
                                                          image_format=image_format,
                                                          feature_name=self.feature_name)

  def _build_graph2(self, model_name, height, width, image_format=None):
    feature_name = melt.get_features_name(self.model_name)
    melt.apps.image_processing.init(model_name, feature_name)
    return melt.apps.image_processing.image_processing_fn(self.images_feed,  
                                                          height=height, 
                                                          width=width,
                                                          image_format=image_format,
                                                          feature_name=feature_name)

  def process(self, images):
    if not isinstance(images, (list, tuple, np.ndarray)):
      images = [images]

    if isinstance(images[0], (str, np.string_)) and len(images[0]) < 1000:
      images = [melt.image.read_image(image) for image in images]

    return self.sess.run(self.img2feautres_op, feed_dict={self.images_feed: images})

  def process2(self, images):
    if not isinstance(images, (list, tuple, np.ndarray)):
      images = [images]

    if isinstance(images[0], (str, np.string_)) and len(images[0]) < 1000:
      images = [melt.image.read_image(image) for image in images]

    return self.sess.run(self.img2feautres_op2, feed_dict={self.images_feed: images})

  def gen_feature(self, images):
    return self.process(images)

  def gen_features(self, images):
    return self.process2(images)
