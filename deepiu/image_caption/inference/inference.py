#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   inference.py
#        \author   chenghuige  
#          \date   2016-10-06 19:48:19.923359
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS
 
flags.DEFINE_string('vocab', '/home/gezi/new/temp/image-caption/keyword/tfrecord/seq-basic/vocab.txt', 'vocabulary file')
flags.DEFINE_string('image_path', '/home/gezi/data/imgs/im2txt/usa-campus.jpg', '')
flags.DEFINE_string('image_model_path', '/home/gezi/data/inceptionv3/inception_v3.ckpt', '')
flags.DEFINE_string('model_dir', '/home/gezi/new/temp/image-caption/keyword/model/showandtell/', '')

import gezi
import melt 
from deepiu.util import text2ids

image_model = melt.image.ImageModel(FLAGS.image_model_path)

text2ids.init(FLAGS.vocab)


def predict(predictor, image_path):
  timer = gezi.Timer()
  image_feature = image_model.process_one_image(image_path)
  text, score = predictor.inference(['text', 'text_score'], 
                                    feed_dict= {
                                      'show_and_tell/model_init_1/image_feature:0': image_feature
                                      })
  
  for result in text:
    print(result, text2ids.ids2text(result), 'decode time(ms):', timer.elapsed_ms())
  
  timer = gezi.Timer()
  texts, scores = predictor.inference(['beam_text', 'beam_text_score'], 
                                    feed_dict= {
                                      'show_and_tell/model_init_1/image_feature:0': image_feature
                                      })

  texts = texts[0]
  scores = scores[0]
  for text, score in zip(texts, scores):
    print(text, text2ids.ids2text(text), score)

  print('beam_search using time(ms):', timer.elapsed_ms())


predictor = melt.Predictor(FLAGS.model_dir)

predict(predictor, FLAGS.image_path)

