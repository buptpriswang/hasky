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

flags.DEFINE_string('model_dir', '/home/gezi/new/temp/image-caption/lijiaoshou/model/bow.full/', '')
#flags.DEFINE_string('model_dir', '/home/gezi/new/temp/image-caption/keyword/model/bow.lijiaoshou/', '')
#flags.DEFINE_string('model_dir', '/home/gezi/new/temp/image-caption/keyword/model/bow', '')
#flags.DEFINE_string('vocab', '/home/gezi/new/temp/image-caption/keyword/tfrecord/bow/vocab.txt', 'vocabulary file')
flags.DEFINE_string('vocab', '/home/gezi/new/temp/image-caption/lijiaoshou/tfrecord/bow/vocab.txt', 'vocabulary file')

flags.DEFINE_string('image_feature_name_', 'bow/main/image_feature:0', 'model_init_1 because predictor after trainer init')
flags.DEFINE_string('text_name', 'bow/main/text:0', 'model_init_1 because predictor after trainer init')


flags.DEFINE_string('image_file', '/home/gezi/data/lijiaoshou/test_image.txt', '')
#flags.DEFINE_string('image_feature_dir', '/home/gezi/data/lijiaoshou/train/', '')
#flags.DEFINE_string('image_feature_file_', '/home/gezi/new/data/keyword/valid/part-00000', 'valid data')
#flags.DEFINE_string('image_feature_pattern', '/home/gezi/new/data/keyword/valid/part*', 'valid data')
flags.DEFINE_string('image_feature_pattern', '/home/gezi/data/lijiaoshou/candidate_feature.txt', 'train data')

flags.DEFINE_string('all_text_strs', '/home/gezi/new/temp/image-caption/lijiaoshou/tfrecord/seq-basic/valid/distinct_text_strs.npy', '')
flags.DEFINE_string('all_texts', '/home/gezi/new/temp/image-caption/lijiaoshou/tfrecord/seq-basic/valid/distinct_texts.npy', '')

flags.DEFINE_integer('num_files', 2, '')
flags.DEFINE_integer('batch_size_', 100000, '')
flags.DEFINE_integer('text_max_words', 100, '')

flags.DEFINE_string('seg_method_', 'full', '')
flags.DEFINE_bool('feed_single_', False, '')

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

ENCODE_UNK = False

img_html = '<p> <td><a href={0} target=_blank><img src={0} height=250 width=250></a></p> <td>'

def predicts(image_features, word_ids_list):
  score = predictor.inference('image_words_score', 
                              feed_dict= {
                                      FLAGS.image_feature_name_: image_features
                                      })
  
  score = score.squeeze()

  #return score.tolist()
  return score
  


def run():
  m = {}
  files = glob.glob(FLAGS.image_feature_pattern)
  for file in files:
    for line in open(file):
      l = line.strip().split('\t')
      m[l[0]] = l[-1]

  for i, line in enumerate(open(FLAGS.image_file)):
    image = line.strip()
    if image not in m:
      continue
    image_feature = m[image].split('\x01')
    image_feature = [float(x) for x in image_feature] 
    timer = gezi.Timer()
    word_ids_list = np.load(FLAGS.all_texts)
    all_text_strs = np.load(FLAGS.all_text_strs)
    scores = predicts([image_feature], word_ids_list)
    print(img_html.format(image))
    vocab = text2ids.vocab
    topn = 50
    indexes = (-scores).argsort()[:topn]
    j = 0
    for i, index in enumerate(indexes):
      if index > 20000:
        continue 
      if vocab.key(int(index)) == '丙烯酸':
        continue
      print(j, vocab.key(int(index)), scores[index])
      print('<br>')
      j += 1

    print(i, image, timer.elapsed(), file=sys.stderr)


def main(_):
  text2ids.init()
  global predictor, image_model
  predictor = melt.Predictor(FLAGS.model_dir)

  run()

if __name__ == '__main__':
  tf.app.run()
