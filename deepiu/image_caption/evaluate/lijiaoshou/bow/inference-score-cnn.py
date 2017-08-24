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

flags.DEFINE_string('model_dir', '/home/gezi/new/temp/image-caption/lijiaoshou/model/cnn/', '')
#flags.DEFINE_string('model_dir', '/home/gezi/new/temp/image-caption/keyword/model/bow.lijiaoshou/', '')
#flags.DEFINE_string('model_dir', '/home/gezi/new/temp/image-caption/keyword/model/bow', '')
#flags.DEFINE_string('vocab', '/home/gezi/new/temp/image-caption/keyword/tfrecord/bow/vocab.txt', 'vocabulary file')
flags.DEFINE_string('vocab', '/home/gezi/new/temp/image-caption/lijiaoshou/tfrecord/seq-basic/vocab.txt', 'vocabulary file')

flags.DEFINE_string('image_feature_name_', 'cnn/main/image_feature:0', 'model_init_1 because predictor after trainer init')
flags.DEFINE_string('text_name', 'cnn/main/text:0', 'model_init_1 because predictor after trainer init')


flags.DEFINE_string('text_file', '/home/gezi/data/lijiaoshou/wenan.special.txt', '')
#flags.DEFINE_string('image_feature_dir', '/home/gezi/data/lijiaoshou/train/', '')
#flags.DEFINE_string('image_feature_file_', '/home/gezi/new/data/keyword/valid/part-00000', 'valid data')
#flags.DEFINE_string('image_feature_pattern', '/home/gezi/new/data/keyword/valid/part*', 'valid data')
flags.DEFINE_string('image_feature_pattern', '/home/gezi/data/lijiaoshou/candidate_feature.txt', 'train data')
flags.DEFINE_integer('num_files', 2, '')
flags.DEFINE_integer('batch_size_', 100000, '')
flags.DEFINE_integer('text_max_words', 50, '')

flags.DEFINE_string('seg_method_', 'basic', '')
flags.DEFINE_bool('feed_single_', True, '')

#flags.DEFINE_string('seg_method_', 'full', '')
#flags.DEFINE_bool('feed_single_', False, '')

import sys, os, math
import gezi, melt
import numpy as np

from deepiu.util import text2ids

import glob 
import conf
from conf import TEXT_MAX_WORDS, NUM_RESERVED_IDS, ENCODE_UNK

predictor = None 

img_html = '<p> <td><a href={0} target=_blank><img src={0} height=250 width=250></a></p> {1} {2}, <br /> {3}<td>'

def _text2ids(text, max_words):
  word_ids = text2ids.text2ids(text, 
                               seg_method=FLAGS.seg_method_, 
                               feed_single=FLAGS.feed_single_, 
                               allow_all_zero=True, 
                               pad=False)
  word_ids = word_ids[:max_words]
  word_ids = gezi.pad(word_ids, max_words, 0)

  return word_ids

def predicts(image_features, text):
  #TODO may be N texts to speed up as bow support this
  word_ids_list = [_text2ids(text, FLAGS.text_max_words or TEXT_MAX_WORDS)] 

  score = predictor.inference('score', 
                              feed_dict= {
                                      FLAGS.image_feature_name_: image_features,
                                      FLAGS.text_name: word_ids_list
                                      })
  
  score = score.squeeze()

  return score.tolist()
  
def top_images(text):
  image_set = set()
  images = []
  image_features = []
  scores = []
  itexts = []
  num = 0
  for file in glob.glob(FLAGS.image_feature_pattern):
    print(file, file=sys.stderr)
    for line in open(file):
      l = line.strip().split('\t')
      image = l[0].strip()
      itext = l[1].strip()
      if image in image_set:
        continue
      else:
        image_set.add(image)
      image_feature = l[-1].split('\x01')
      image_feature = [float(x) for x in image_feature]

      image_features.append(image_feature)

      images.append(image)

      itexts.append(itext)
    
      if len(image_features) == FLAGS.batch_size_:
        scores += predicts(image_features, text)
        image_features = []
    num += 1
    if num == FLAGS.num_files:
      break

  if image_features:
    scores += predicts(image_features, text)

  image_scores = zip(scores, images, itexts)
  image_scores.sort(reverse=True)

  print('<p><font size="5" color="red"><B>%s</B></font></p>'%text)
  for i, (score, image, itext) in enumerate(image_scores[:50]):
    if i % 5 == 0:
      print('<table><tr>')
    #itext = ''
    print(img_html.format(image, i, score, itext))
    if (i + 1) % 5 == 0:
      print('</tr></table>')

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
