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
  def __init__(self, model_dir, vocab_path, image_checkpoint_path=None, image_model_name='InceptionResnetV2', index=0, sess=None):
    self.image_model = None
    if image_checkpoint_path is not None:
      self.image_model = melt.image.ImageModel(image_checkpoint_path, image_model_name, sess=sess)

    if not isinstance(model_dir, (list, tuple)):
      self.predictor = melt.TextPredictor(model_dir, index=index, sess=sess)
    else:
      self.predictor = melt.EnsembleTextPredictor(model_dir, index=index, sess=sess)

    ids2text.init(vocab_path)

  def _predict(self, image):
    if self.image_model is not None:
      image = self.image_model.gen_feature(image)
    return self.predictor.predict(image)
  
  def predict_text(self, image):
    return self._predict(image)

  def predict(self, image):
    texts, scores = self._predict(image)
    texts = texts[0]
    scores = scores[0]
    return np.array([ids2text.translate(text) for text in texts]), scores

  def word_ids(self, image):
    return self._predict(image)

  def translate(self, image):
    texts, scores = self._predict(image)
    return [ids2text.translate(text[0]) for text in texts]

  def predict_best(self, image):
    return self.translate(image)