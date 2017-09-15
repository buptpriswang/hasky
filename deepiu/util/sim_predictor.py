#!/usr/bin/env python
# ==============================================================================
#          \file   TextPredictor.py
#        \author   chenghuige  
#          \date   2017-09-14 17:35:07.765273
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf 

import sys, os
import numpy as np 
import melt   

class SimPredictor(object):
  def __init__(self, model_dir,  
               image_checkpoint_path=None, 
               image_model_name='InceptionResnetV2', 
                key='score', index=0,
                sess=None):
    self.image_model = None
    if image_checkpoint_path is not None:
      self.image_model = melt.image.ImageModel(image_checkpoint_path, image_model_name, sess=sess)
    self.predictor = melt.SimPredictor(model_dir, key=key, index=index, sess=sess)
    self.sess = self.predictor._sess
    self.index = index

  def predict(self, ltext, rtext):
    if self.image_model is not None:
      ltext = self.image_model.gen_feature(ltext)
    return self.predictor.predict(ltext, rtext)

  def elementwise_predict(self, ltexts, rtexts):
    if self.image_model is not None:
      ltexts = self.image_model.gen_feature(ltexts)
    return self.predictor.elementwise_predict(ltexts, rtexts)

  def top_k(self, ltext, rtext, k=1):
    if self.image_model is not None:
      ltext = self.image_model.gen_feature(ltext)
      rtext = self.image_model.gen_feature(rtext)
    return self.predictor.top_k(ltext, rtext, k=k)

  def words_importance(self, inputs):  
    return self.sess.run(tf.get_collection('words_importance')[self.index], feed_dict={self.predictor._rfeed: inputs})

  def words_score(self, image):
    if self.image_model is not None:
      image = self.image_model.gen_feature(image)
    return self.sess.run(tf.get_collection('image_words_score')[self.index], feed_dict={self.predictor._lfeed: image}) 

  def top_words(self, image, k=1): 
    if self.image_model is not None:
      image = self.image_model.gen_feature(image)
    scores = tf.get_collection('image_words_score')[self.index]
    vals, indexes = tf.nn.top_k(scores, k)
    return self.sess.run([vals, indexes], feed_dict={self.predictor._lfeed: image}) 