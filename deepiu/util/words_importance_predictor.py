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

from deepiu.util import text2ids

class WordsImportancePredictor(object):
  def __init__(self, model_dir, vocab_path, key=None, index=0, sess=None):
    self.predictor = melt.WordsImportancePredictor(model_dir, key=key, index=index)
    text2ids.init(vocab_path)

  def predict(self, inputs, seg_method='basic', feed_single=True, max_words=None):
  	if not isinstance(inputs, (list, tuple, np.ndarray)):
  		inputs = [inputs]
  	if isinstance(inputs[0][0], str):
  		word_ids = [text2ids.text2ids(input, seg_method=seg_method, feed_single=feed_single, max_words=max_words) for input in inputs]
  	else:
  		word_ids = inputs
  		
  	return self.predictor.predict(word_ids), word_ids
