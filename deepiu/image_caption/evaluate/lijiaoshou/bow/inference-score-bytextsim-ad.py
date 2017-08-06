#!/usr/bin/env python
# -*- coding: gbk -*-
# ==============================================================================
#          \file   predict.py
#        \author   chenghuige  
#          \date   2016-10-19 06:54:26.594835
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('model_dir', '/home/gezi/new/temp/image-caption/lijiaoshou/model/bow/', '')
#flags.DEFINE_string('model_dir', '/home/gezi/new/temp/image-caption/keyword/model/bow.lijiaoshou/', '')
#flags.DEFINE_string('model_dir', '/home/gezi/new/temp/image-caption/keyword/model/bow', '')

flags.DEFINE_string('vocab', '/home/gezi/new/temp/image-caption/lijiaoshou/tfrecord/bow/vocab.txt', 'vocabulary file')
#flags.DEFINE_string('vocab', '/home/gezi/new/temp/image-caption/keyword/tfrecord/bow/vocab.txt', 'vocabulary file')

flags.DEFINE_string('image_feature_name_', 'bow/main/image_feature:0', 'model_init_1 because predictor after trainer init')
flags.DEFINE_string('text_name', 'bow/main/text:0', '')
flags.DEFINE_string('text2_name', 'bow/main/text2:0', '')

flags.DEFINE_string('text_file', '/home/gezi/data/lijiaoshou/wenan.special.txt', '')
#flags.DEFINE_string('image_feature_file_', '/home/gezi/data/lijiaoshou/train/shoubai_feature.txt_0', 'train data')
#flags.DEFINE_string('image_feature_file_', '/home/gezi/data/lijiaoshou/toutiao_feature.txt', 'train data')
flags.DEFINE_string('image_feature_file_', '/home/gezi/data/lijiaoshou/candidate_feature.txt', 'train data')

flags.DEFINE_integer('batch_size_', 10000, '')

flags.DEFINE_string('seg_method_', 'full', '')
flags.DEFINE_bool('feed_single_', False, '')


import sys, os, math
import gezi, melt
import numpy as np

from deepiu.util import text2ids

import conf
from conf import TEXT_MAX_WORDS, NUM_RESERVED_IDS, ENCODE_UNK

predictor = None 

img_html = '<p><a href={0} target=_blank><img src={0} height=200></a></p>\n pos:{1} score:{2}, text:{3}, itext:{4}'

def _text2ids(text, max_words):
  word_ids = text2ids.text2ids(text, 
                               seg_method=FLAGS.seg_method_, 
                               feed_single=FLAGS.feed_single_, 
                               allow_all_zero=True, 
                               pad=False)
  word_ids = word_ids[:max_words]
  word_ids = gezi.pad(word_ids, max_words, 0)

  return word_ids

def predicts(image_features, input_texts, text):
  #TODO may be N texts to speed up as bow support this
  word_ids_list = [_text2ids(text, TEXT_MAX_WORDS)] 
  input_word_ids_list = [_text2ids(input_text, TEXT_MAX_WORDS) for input_text in input_texts]

  score = predictor.inference('textsim_score', 
                              feed_dict= {
                                      FLAGS.text_name: input_word_ids_list,
                                      FLAGS.text2_name: word_ids_list
                                      })
  
  score = score.squeeze()

  return score.tolist()
  
def top_images(text):
  image_set = set()
  images = []
  itexts = []
  image_features = []
  input_texts = []
  scores = []
  for line in open(FLAGS.image_feature_file_):
    l = line.strip().split('\t')
    image = l[0].strip()
    if image in image_set:
      continue
    else:
      image_set.add(image)
    input_text = l[1].split('\x01')[0].strip()
    image_feature = l[-1].split('\x01')
    image_feature = [float(x) for x in image_feature]

    image_features.append(image_feature)
    images.append(image)
    input_texts.append(input_text)
    itexts.append(input_text)

    
    if len(image_features) == FLAGS.batch_size_:
      scores += predicts(image_features, input_texts, text)
      image_features = []
      input_texts = []

  if image_features:
    scores += predicts(image_features, input_texts, text)

  image_scores = zip(scores, images, itexts)
  image_scores.sort(reverse=True)

  for i, (score, image, input_text) in enumerate(image_scores[:10]):
    print(img_html.format(image, i, score, text, input_text))

  #print(scores)


def run():
  for i, line in enumerate(open(FLAGS.text_file)):
    text = line.split('\t')[0].strip()
    timer = gezi.Timer()
    top_images(text)
    print(i, text, timer.elapsed(), file=sys.stderr)


def main(_):
  text2ids.init()
  global predictor, image_model
  predictor = melt.Predictor(FLAGS.model_dir)

  run()

if __name__ == '__main__':
  tf.app.run()
