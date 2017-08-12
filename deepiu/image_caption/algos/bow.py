#!/usr/bin/env python
# ==============================================================================
#          \file   bow.py
#        \author   chenghuige  
#          \date   2016-08-18 00:56:57.071798
#   \Description  
# ==============================================================================

#TODO remove bow, cn.. ,just use discrinant_trainer with encoder as class member like DualSim
  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS

import melt

from deepiu.image_caption.algos.discriminant_trainer import DiscriminantTrainer
from deepiu.image_caption.algos.discriminant_predictor import DiscriminantPredictor

from deepiu.seq2seq import bow_encoder

class Bow(object):
  def __init__(self, is_training=True, is_predict=False):
    super(Bow, self).__init__()
    self.is_training = is_training
    self.is_predict = is_predict

    if is_training:
      self.trainer = DiscriminantTrainer(is_training=True)

  def gen_text_feature(self, text, emb):
    """
    this common interface, ie may be lstm will use other way to gnerate text feature
    """
    return bow_encoder.encode(text, emb)

  # def gen_text_importance(self, text, emb):
  #   #return bow_encoder.importance_encode(text, emb=emb)
  #   #text batch_size must be 1 currently [1, seq_len] -> [seq_len, 1]
  #   sequence = tf.transpose(sequence, [1, 0])
  #   word_feature = self.trainer.forward_text(word_index)
  #   return word_feature

  def build_train_graph(self, image_feature, text, neg_text, neg_image=None):
    #self.trainer = DiscriminantTrainer(is_training=self.is_training)
    self.trainer.gen_text_feature = self.gen_text_feature
    loss = self.trainer.build_graph(image_feature, text, neg_text, neg_image, lookup_negs_once=True)
    return loss


class BowPredictor(DiscriminantPredictor, melt.PredictorBase):
  def __init__(self):
    #super(BowPredictor, self).__init__()
    melt.PredictorBase.__init__(self)
    DiscriminantPredictor.__init__(self)  

    predictor = Bow(is_training=False, is_predict=True)
    self.gen_text_feature = predictor.gen_text_feature

  #--------- only used during training evaluaion, image_feature and text all small
  def build_train_graph(self, image_feature, text, neg_text, neg_image=None):
    """
    Only used for train and evaluation, hack!
    """
    return super(DiscriminantPredictor, self).build_graph(
      image_feature, text, neg_text, neg_image, lookup_negs_once=True)
