#!/usr/bin/env python
#encoding=utf-8
# ==============================================================================
#          \file   inference.py
#        \author   chenghuige  
#          \date   2017-09-14 07:45:47.415075
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os
from deepiu.util.words_importance_predictor import WordsImportancePredictor 

model_dir = '/home/gezi/new/temp/image-caption/ai-challenger/model/bow/'
vocab_path = '/home/gezi/new/temp/image-caption/ai-challenger/tfrecord/seq-basic/vocab.txt'

predictor = WordsImportancePredictor(model_dir, vocab_path)

scores = predictor.predict(['绿油油的田地里有一个双手拿着工具的男人在干活'])

print(scores)
