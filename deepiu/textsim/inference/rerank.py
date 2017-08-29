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
flags.DEFINE_string('exact_model_dir', '/home/gezi/new/temp/makeup/title2name/model/seq2seq.gencopy.switch/', '')
flags.DEFINE_float('exact_ratio', 1., '')

flags.DEFINE_string('vocab', '/home/gezi/new/temp/makeup/title2name/tfrecord/seq-basic/vocab.txt', 'vocabulary file')

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

def sim(ltexts, rtexts):
  #TODO may be N texts to speed up as bow support this
  lword_ids_list = [_text2ids(ltext, TEXT_MAX_WORDS) for ltext in ltexts]  
  rword_ids_list = [_text2ids(rtext, TEXT_MAX_WORDS) for rtext in rtexts]

  return predictor.top_k(lword_ids_list, rword_ids_list, 50, ratio=FLAGS.exact_ratio)

def run():
  corpus_pattern = os.path.join('./lankou.dasou.format.txt')
  corpus_pattern = os.path.join('/home/gezi/data/product/makeup/tb/title2name/name.filtered.txt')

  ltexts = []
  for line in open('./lancou.yanshuang.txt'):
    ltexts.append(line.strip().split('\t')[1])

  ltexts2 = ['��ר��ֱ���������͡�lancome��ޢݼ��׿����˪20ml', 'lancome��ޢ��׿����˪20ml(Ʒ�ʱ�֤)', 'lancome��ޢݼ��������˪5ml*1',
             'lancome��ޢݼ��������˪20ml(Ʒ�ʱ�֤)', '����199-100����ޢ(lancome)ݼ��������˪3ml*3С��', '��ޢ(lancome)�䴿����������˪3ml']
 
  ltexts2 = [
             '��ޢ Lancome С��ƿ ��˪���������۲���˪��Ĥ˪ С��ƿ��˪15ml',
             'Lancome��ޢС��ƿ ��˪���������۲���˪/��Ĥ˪15ml',
             '��ޢ Lancome С��ƿ ��˪���������۲���˪ С��ƿ��˪15ml',
            ]

  ltexts2 = ['lancome��ޢ��˪���������ս����۲�������15ml', 'lancome��ޢ��˪���������ս�����˪15ml', '��ޢ(Lancome)�䴿����������˪3ml', 
            '��ר���л�����ޢС��ƿ��˪�����۲���˪��˪15ml��װ', '��ޢ(lancome)С��ƿ���������۲���˪��˪15ml']
  print(len(ltexts))
  #ltexts = ltexts[:10]

  max_words = 50
  
  rtexts = []
  for file in glob.glob(corpus_pattern):
    rtexts += [x.strip() for x in open(file).readlines()]
  rtexts = [x.split('\t')[1] for x in rtexts if '��ޢ' in x]
  rtexts = list(set(rtexts))

  print(len(rtexts))
  values, indices = sim(ltexts2, rtexts)

  for i in xrange(len(ltexts2)):
    print('-----------[input]:', ltexts2[i])
    for index, value in zip(indices[i], values[i]):
      print(rtexts[index], value)


def main(_):
  text2ids.init()
  global predictor
  predictor = melt.RerankSimPredictor(FLAGS.model_dir, FLAGS.exact_model_dir)

  run()

if __name__ == '__main__':
  tf.app.run()
