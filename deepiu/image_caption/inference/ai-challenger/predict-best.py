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
from deepiu.util.text_predictor import TextPredictor

image_dir = '/home/gezi/data2/data/ai_challenger/image_caption/pic/'
image_file = '6275b5349168ac3fab6a493c509301d023cf39d3.jpg'
image_path = os.path.join(image_dir, image_file)
image_model_checkpoint_path = '/home/gezi/data/image_model_check_point/inception_resnet_v2_2016_08_30.ckpt'
model_dir = '/home/gezi/new/temp/image-caption/ai-challenger/model/showandtell/'
vocab_path = '/home/gezi/new/temp/image-caption/ai-challenger/tfrecord/seq-basic/vocab.txt'

predictor = TextPredictor(model_dir, vocab_path, image_model_checkpoint_path, image_model_name='InceptionResnetV2')
text = predictor.predict_best(image_path)

print(text)
