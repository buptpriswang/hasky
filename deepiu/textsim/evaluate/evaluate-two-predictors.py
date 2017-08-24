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

flags.DEFINE_string('model_dir', '/home/gezi/new/temp/makeup/title2name/model/bow', '')

flags.DEFINE_string('vocab', '/home/gezi/new/temp/makeup/title2name/tfrecord/seq-basic/vocab.txt', 'vocabulary file')

flags.DEFINE_string('ltext', 'dual_bow/main/ltext:0', '')
flags.DEFINE_string('rtext', 'dual_bow/main/rtext:0', '')

flags.DEFINE_integer('batch_size_', 10000, '')

flags.DEFINE_string('seg_method_', 'basic', '')
flags.DEFINE_bool('feed_single_', True, '')


import sys, os, math, glob
import gezi, melt
import numpy as np

from deepiu.util import text2ids

import conf
from conf import TEXT_MAX_WORDS

predictor = None 

def _text2ids(text, max_words):
  word_ids = text2ids.text2ids(text, 
                               seg_method=FLAGS.seg_method_, 
                               feed_single=FLAGS.feed_single_, 
                               append_start=False,
                               append_end=False,
                               allow_all_zero=True, 
                               pad=True,
                               max_words=max_words)
  return word_ids

def sim(ltexts, rtexts, tag='nearby'):
  #TODO may be N texts to speed up as bow support this
  lword_ids_list = [_text2ids(ltext, TEXT_MAX_WORDS) for ltext in ltexts]  
  rword_ids_list = [_text2ids(rtext, TEXT_MAX_WORDS) for rtext in rtexts]

  return predictor.inference(['%s_values'%tag, '%s_indices'%tag], 
                              feed_dict= {
                                      FLAGS.ltext: lword_ids_list,
                                      FLAGS.rtext: rword_ids_list
                                      })
  

def run():
  corpus_pattern = os.path.join('/home/gezi/data/product/makeup/tb/title2name/valid/*')

  max_words = 50
  ltexts = [
      '��ʫ���죨Estee Lauder����͸�޻��۲��ܼ�����¶ 15ml����˪ ANR �������� ����ϸ�ƣ�',
      '���������С���ʫ���죨Estee Lauder����͸�޻��۲�����˪ 15ml����˪ ANR �������� ����ϸ�� ����Ȧ��'
      ]

  rtexts = []
  for file in glob.glob(corpus_pattern):
    rtexts += [x.strip() for x in open(file).readlines()]
  rtexts = [x.split('\t')[1] for x in rtexts]
  rtexts = list(set(rtexts))
  

def main(_):
  text2ids.init()
  global predictor
  predictor = melt.Predictor(FLAGS.model_dir)
  predictor2 = melt.Predictor('/home/gezi/new/temp/makeup/title2name/model/cnn.hic/')
  print(predictor, predictor2)
  print(tf.get_default_graph().get_all_collection_keys())
  print(tf.get_collection('score'))

if __name__ == '__main__':
  tf.app.run()
