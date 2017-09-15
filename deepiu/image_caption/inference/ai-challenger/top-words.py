#!/usr/bin/env python
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
from deepiu.util.sim_predictor import SimPredictor 
from deepiu.util import vocabulary

import melt

image_dir = '/home/gezi/data2/data/ai_challenger/image_caption/pic/'

image_file = '6275b5349168ac3fab6a493c509301d023cf39d3.jpg'
if len(sys.argv) > 1:
  image_file = sys.argv[1]

image_path = os.path.join(image_dir, image_file)
image_model_checkpoint_path = '/home/gezi/data/image_model_check_point/inception_resnet_v2_2016_08_30.ckpt'
model_dir = '/home/gezi/new/temp/image-caption/ai-challenger/model/bow/'
vocab_path = '/home/gezi/new/temp/image-caption/ai-challenger/tfrecord/seq-basic/vocab.txt'

vocabulary.init(vocab_path)
vocab  = vocabulary.vocab

predictor = SimPredictor(model_dir, image_model_checkpoint_path, image_model_name='InceptionResnetV2')

scores, word_ids = predictor.top_words([melt.read_image(image_path)], 50)
scores = scores[0]
word_ids = word_ids[0]

for word_id, score in zip(word_ids, scores):
  print(vocab.key(int(word_id)), score)
