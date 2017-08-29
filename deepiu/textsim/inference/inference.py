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
  corpus_pattern = os.path.join('./lankou.dasou.format.txt')

  ltexts = []
  for line in open('./lancou.yanshuang.txt'):
    ltexts.append(line.strip().split('\t')[1])

  ltexts2 = ['【专柜直供京东配送】lancome兰蔻菁纯卓颜眼霜20ml', 'lancome兰蔻金纯卓颜眼霜20ml(品质保证)', 'lancome兰蔻菁纯臻颜眼霜5ml*1',
             'lancome兰蔻菁纯臻颜眼霜20ml(品质保证)', '【满199-100】兰蔻(lancome)菁纯臻颜眼霜3ml*3小样', '兰蔻(lancome)箐纯臻颜润养眼霜3ml']
 
  ltexts2 = [
             '兰蔻 Lancome 小黑瓶 眼霜精华肌底眼部凝霜眼膜霜 小黑瓶眼霜15ml',
             'Lancome兰蔻小黑瓶 眼霜精华肌底眼部凝霜/眼膜霜15ml',
             '兰蔻 Lancome 小黑瓶 眼霜精华肌底眼部凝霜 小黑瓶眼霜15ml',
            ]

  ltexts2 = ['lancome兰蔻眼霜新立体塑颜紧致眼部精华乳15ml', 'lancome兰蔻眼霜新立体塑颜紧致眼霜15ml', '兰蔻(Lancome)箐纯臻颜润养眼霜3ml', 
            '【专柜行货】兰蔻小黑瓶眼霜肌底眼部凝霜眼霜15ml正装', '兰蔻(lancome)小黑瓶精华肌底眼部凝霜眼霜15ml']
  print(len(ltexts))
  #ltexts = ltexts[:10]

  max_words = 50
  
  rtexts = []
  for file in glob.glob(corpus_pattern):
    rtexts += [x.strip() for x in open(file).readlines()]
  #rtexts = [x.split('\t')[1] for x in rtexts if '兰蔻' in x]
  rtexts = list(set(rtexts))

  print(len(rtexts))
  #rtexts = rtexts[:40000]
  values, indices = sim(ltexts2, rtexts)

  for i in xrange(len(ltexts2)):
    print('-----------[input]:', ltexts2[i])
    for index, value in zip(indices[i], values[i]):
      print(rtexts[index], value)

  rtexts = ltexts 

  ##rtexts = rtexts[:40000]
  #values, indices = sim(ltexts2, rtexts, tag='lsim_nearby')

  #for i in xrange(len(ltexts2)):
  #  print('-----------[input]:', ltexts2[i])
  #  for index, value in zip(indices[i], values[i]):
  #    print(rtexts[index], value)
  # 

def main(_):
  text2ids.init()
  global predictor
  predictor = melt.Predictor(FLAGS.model_dir)

  run()

if __name__ == '__main__':
  tf.app.run()
