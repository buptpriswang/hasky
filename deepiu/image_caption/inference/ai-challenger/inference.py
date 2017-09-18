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
import melt   
from deepiu.util import ids2text

image_dir = image_dir = '/home/gezi/data2/data/ai_challenger/image_caption/pic/'
image_file = '6275b5349168ac3fab6a493c509301d023cf39d3.jpg'

image_model_checkpoint_path = '/home/gezi/data/image_model_check_point/inception_resnet_v2_2016_08_30.ckpt'
model_name='InceptionResnetV2'
feature_name = melt.image.get_features_name(model_name)
image_model = melt.image.ImageModel(image_model_checkpoint_path, model_name=model_name, feature_name=feature_name)

model_dir = '/home/gezi/new/temp/image-caption/ai-challenger/model/showattentell/'
predictor = melt.TextPredictor(model_dir)

vocab_path = '/home/gezi/new/temp/image-caption/ai-challenger/tfrecord/seq-basic/vocab.txt'
ids2text.init(vocab_path)

def predict(image_name):
  image_path = os.path.join(image_dir, image_name)
  feature = image_model.gen_feature(image_path) 
  
  texts, scores = predictor.predict(feature)
  #print(texts, scores)

  texts = texts[0]
  scores = scores[0]
  for text, score in zip(texts, scores):
    print(ids2text.ids2text(text), score)
    print(ids2text.translate(text), score)

while True:
  image_name = raw_input('image_name like 6275b5349168ac3fab6a493c509301d023cf39d3.jpg:')
  if image_name == 'q':
    break
  if not image_name.endswith('.jpg'):
    image_name += '.jpg'
  predict(image_name)
