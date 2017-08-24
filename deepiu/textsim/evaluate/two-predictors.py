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
  #ltexts = ['��ʫ����ˮ��˪', '��ʫ����С��ƿ', '��ʫ�����ʯ��', '��������Ȫ��ˮ����', 'Adidas���ϴ�˹��ʿ��ˮ��ʿ��������ˮ ��������100ml���������С�']
  ltexts = ['��������Ȫ��ˮ����', 
            '�ֻ��¹�balea����������������������ͱ�ʪϴ���̽����150ml',
            '��ʫ����anr�޻���͸����¶',
            #'�ձ�����POLA����ɺ����ʯ������Ǿ������Ǹ�Ƭ3������', 
            #'Adidas���ϴ�˹��ʿ��ˮ��ʿ��������ˮ ��������100ml���������С�',
            #'lancome/��ޢMiracle�氮�漣Ůʿ��ˮEDP~30/50ML�����Ʒ����'
           ]

  #ltexts = ['������bb˪/cc˪ BB˪ ����BB �۱� �������滻װ*2+�������', 
  #          '���������С���������MAYBELLINE����Ȼ������˪01����ɫ 14g������� �ᱡ��ױ ����ʪ ���룩']
  #ltexts = ['ȥ��ͷ���������鲴����������ϴ����ѧ����Ůʿ�����࿹��']

  #ltexts = ['���������С���羿�����Urtekram�������� Դ��ˮ������Ľ˹150ml']

  #ltexts = ['��羿��������������Ľ˿150ml ������ë�ײ�ˮ��ʪ���ư�����ĭӨ��ϴ����Ůʿ����']

  ltexts = ['ŷ����������Ժ���߾���Һ',
            'ŷ����������Ժϸ���޻�����Һ']

  ltexts = [
            'ѩ��������/����ҹ��˯���޸���Ĥ',
            'ѩ��������˺��/��貾�����Ĥ'
           ]

  ltexts = [
      '�ձ�ԭװ����soft stoneҸ�¸�ˬֹ��ζȥ������ʯ����ʯ',
      '�ձ�deona tulleҸ��ֹ����soft stone����������Ҹ��¶',
      '�ձ�deona tulle soft stone����ʯ����ʯҸ��ֹ��¶',
      ]
  ltexts = [
           'Ұ����魻�������Ĥ',
           '����ռ�Ұ����魻�������Ĥ',
           '���������С���ʫ���죨Estee Lauder�����Խ�ʵ�����˪15ml(�������Խ�ʵ��˪)����˪ �������� ����ϸ�ƣ�',
           ]
  ltexts = [
      'Adidas���ϴ�˹��ʿ��ˮ��ʿ��������ˮ ��������100ml���������С�',
      '���ϴ�˹��ʿŮʿ��������Һ50ml ������ˮ/������� ���㼤����������Ծ������ ���������С���ʿ ����+����+�츳 3ƿ',
      '���ϴ�˹��adidas����ʿ��ˮ-��սŷ��50ml',
      '���ϴ�˹adidas��ʿ��ˮ ����50ml'
      ]

  ltexts = [
      '��ʫ���죨Estee Lauder����͸�޻��۲��ܼ�����¶ 15ml����˪ ANR �������� ����ϸ�ƣ�',
      '���������С���ʫ���죨Estee Lauder����͸�޻��۲�����˪ 15ml����˪ ANR �������� ����ϸ�� ����Ȧ��'
      ]

  rtexts = []
  for file in glob.glob(corpus_pattern):
    rtexts += [x.strip() for x in open(file).readlines()]
  rtexts = [x.split('\t')[1] for x in rtexts]
  rtexts = list(set(rtexts))
  
  #rtexts = rtexts[:40000]
  values, indices = sim(ltexts, rtexts)

  for i in xrange(len(ltexts)):
    print('-----------[input]:', ltexts[i])
    for index, value in zip(indices[i], values[i]):
      print(rtexts[index], value)

  rtexts = []
  for file in glob.glob(corpus_pattern):
    rtexts += open(file).readlines()
  rtexts = [x.split('\t')[0] for x in rtexts]
  rtexts = list(set(rtexts))

  #rtexts = rtexts[:40000]
  values, indices = sim(ltexts, rtexts, tag='lsim_nearby')

  for i in xrange(len(ltexts)):
    print('-----------[input]:', ltexts[i])
    for index, value in zip(indices[i], values[i]):
      print(rtexts[index], value)
   

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
