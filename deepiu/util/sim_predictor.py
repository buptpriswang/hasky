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

class SimPredictor(object):
  def __init__(self, model_dir,  key='score', index=0, 
               image_checkpoint_path=None, image_model_name='InceptionResnetV2', sess=None):
    self.image_model = None
    if image_checkpoint_path is not None:
      self.image_model = melt.image.ImageModel(image_checkpoint_path, image_model_name, sess=sess)
    self.predictor = melt.SimPredictor(model_dir, key=key, index=index)

  def predict(self, ltext, rtext):
    if self.image_model is not None:
      ltext = self.image_model.gen_feature(ltext)
    return self.predictor.predict(ltext, rtext)

  def elementwise_predict(self, ltexts, rtexts):
    if self.image_model is not None:
      ltexts = self.image_model.gen_feature(ltexts)
    return self.predictor.elementwise_predict(ltexts, rtexts)
