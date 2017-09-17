#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   rnn.py
#        \author   chenghuige  
#          \date   2016-09-22 19:13:30.275142
#   \Description  
# ==============================================================================
"""
notice here use last and first refer to the orignal sequence
see for 
1, 2, 3, 4
last means for 4, first means for 1

for backward 
will first rerverse
4, 3, 2, 1
then we get output last means still for 4, and first for 1

Notice for tf implemented bidrectional, will reverse for backward ouput first
always be carefull to use the same position vector(both forward and backward) to add
"""
  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS

import melt

from deepiu.image_caption.algos.discriminant_trainer import DiscriminantTrainer
from deepiu.image_caption.algos.discriminant_predictor import DiscriminantPredictor

from seq2seq.encoders import PoolingEncoder

logging = melt.logging

class Pooling(object):
  def __init__(self, is_training=True, is_predict=False):
    super(Pooling, self).__init__()
    self.is_training = is_training
    self.is_predict = is_predict

    logging.info('rnn_method:{}'.format(FLAGS.rnn_method))
    logging.info('rnn_output_method:{}'.format(FLAGS.rnn_output_method))
    logging.info('num_layers:{}'.format(FLAGS.num_layers))
    logging.info('keep_prob:{}'.format(FLAGS.keep_prob))
    logging.info('emb_dim:{} notice for bidrectional need to * 2, emb_dim means final emb_dim for embedding'.format(FLAGS.emb_dim))
    
    mode = ''
    if is_training:
      self.trainer = DiscriminantTrainer(is_training=True)
      mode = tf.contrib.learn.ModeKeys.TRAIN
      
    self.encoder = PoolingEncoder(params=PoolingEncoder.default_params(), mode=mode)
  
  def gen_text_feature(self, text, emb):
    inputs = tf.nn.embedding_lookup(emb, text)
    text_feature = self.encoder.encode(inputs, melt.length(text)).final_state
    #print('---------------------', text_feature)
    return text_feature

  def build_train_graph(self, image_feature, text, neg_text):
    self.trainer.gen_text_feature = self.gen_text_feature
    loss = self.trainer.build_graph(image_feature, text, neg_text)
    
    #--for debug
    self.gradients = tf.gradients(loss, tf.trainable_variables())
    
    return loss
   
class PoolingPredictor(DiscriminantPredictor, melt.PredictorBase):
  def __init__(self):
    #super(RnnPredictor, self).__init__()
    melt.PredictorBase.__init__(self)
    DiscriminantPredictor.__init__(self)  

    self.gen_text_feature = Pooling(is_training=False, is_predict=True).gen_text_feature
  
