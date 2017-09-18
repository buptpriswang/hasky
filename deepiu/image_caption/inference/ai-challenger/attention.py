#!/usr/bin/env python
# ==============================================================================
#          \file   attention.py
#        \author   chenghuige  
#          \date   2017-09-18 17:28:43.514149
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os, math
import gezi, melt
import numpy as np

TEXT_MAX_WORDS = 100    
decode_max_words = 10

vocab_path = '/home/gezi/new/temp/image-caption/ai-challenger/tfrecord/seq-basic/train/vocab.txt'

image_dir = image_dir = '/home/gezi/data2/data/ai_challenger/image_caption/pic/'
image_file = '6275b5349168ac3fab6a493c509301d023cf39d3.jpg'
image_path = os.path.join(image_dir, image_file)

image_model_checkpoint_path = '/home/gezi/data/image_model_check_point/inception_resnet_v2_2016_08_30.ckpt'
model_name='InceptionResnetV2'
image_model = melt.image.ImageModel(image_model_checkpoint_path, 
                                    model_name=model_name,
                                    #feature_name=melt.image.image_processing.get_features_name(model_name))
                                    feature_name='Conv2d_7b_1x1')
