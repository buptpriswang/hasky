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

import sys, os
import numpy as np 
import melt   
from deepiu.util import ids2text

class TextPredictor(object):
  def __init__(self, model_dir, 
               vocab_path=None, 
               image_checkpoint_path=None, 
               image_model_name='InceptionResnetV2', 
               feature_name=None,
               image_model=None,
               index=0, 
               sess=None):
    self.image_model = image_model
    self.feature_name = feature_name
    if image_model is None and image_checkpoint_path:
      self.image_model = melt.image.ImageModel(image_checkpoint_path, 
                                               image_model_name, 
                                               feature_name=feature_name,
                                               sess=sess)

    if not isinstance(model_dir, (list, tuple)):
      self.predictor = melt.TextPredictor(model_dir, index=index, sess=sess)
    else:
      self.predictor = melt.EnsembleTextPredictor(model_dir, index=index, sess=sess)

    if vocab_path:
      ids2text.init(vocab_path)

  def _predict(self, image, length_normalization_fator=None, beam_size=None):
    if self.image_model is not None:
      if self.feature_name is None:
        image = self.image_model.gen_feature(image)
      else:
        image = self.image_model.gen_features(image)
    return self.predictor.predict_text(image, length_normalization_fator, beam_size)
  
  def predict_text(self, image, length_normalization_fator=None, beam_size=None):
    return self._predict(image, length_normalization_fator, beam_size)

  def predict(self, ltext, rtext):
    if self.image_model is not None:
      if self.feature_name is None:
        ltext = self.image_model.gen_feature(ltext)
      else:
        ltext = self.image_model.gen_features(ltext)
    return self.predictor.predict(ltext, rtext)

  def elementwise_predict(self, ltexts, rtexts):
    if self.image_model is not None:
      if self.feature_name is None:
        ltexts = self.image_model.gen_feature(ltexts)
      else:
        ltexts = self.image_model.gen_features(ltexts)
    return self.predictor.elementwise_predict(ltexts, rtexts)

  def word_ids(self, image, length_normalization_fator=None, beam_size=None):
    return self._predict(image, length_normalization_fator, beam_size)

  def translate(self, image, length_normalization_fator=None, beam_size=None):
    texts, scores = self._predict(image, length_normalization_fator, beam_size)
    return [ids2text.translate(text[0]) for text in texts]

  def predict_best(self, image, length_normalization_fator=None, beam_size=None):
    return self.translate(image, length_normalization_fator, beam_size)