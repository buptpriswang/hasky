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
  #ltexts = ['雅诗兰黛水润霜', '雅诗兰黛小棕瓶', '雅诗兰黛红石榴', '婷美矿物泉补水精华', 'Adidas阿迪达斯男士香水男士古龙淡香水 冰点男香100ml【京东超市】']
  ltexts = ['婷美矿物泉补水精华', 
            '现货德国balea芭乐雅莲花青竹深层清洁控油保湿洗面奶洁面膏150ml',
            '雅诗兰黛anr修护肌透精华露',
            #'日本代购POLA宝丽珊瑚化石鲨鱼软骨精华护骨钙片3个月量', 
            #'Adidas阿迪达斯男士香水男士古龙淡香水 冰点男香100ml【京东超市】',
            #'lancome/兰蔻Miracle真爱奇迹女士香水EDP~30/50ML香港正品代购'
           ]

  #ltexts = ['美宝莲bb霜/cc霜 BB霜 气垫BB 粉饼 象牙白替换装*2+气垫外盒', 
  #          '【京东超市】美宝莲（MAYBELLINE）超然无瑕轻垫霜01亮肤色 14g（巨遮瑕 轻薄裸妆 滋润保湿 隔离）']
  #ltexts = ['去黑头祛螨洁面乳泊舒控油祛痘除螨洗面奶学生男女士深层清洁抗痘']

  #ltexts = ['【京东超市】亚缇克兰（Urtekram）洁面乳 源生水漾洁面慕斯150ml']

  #ltexts = ['亚缇克兰柔采凝肌洁面慕丝150ml 深层清洁毛孔补水保湿改善暗黄泡沫莹润洗面奶女士洁面']

  ltexts = ['欧莲丽妃美容院休眠精华液',
            '欧莲丽妃美容院细胞修护精华液']

  ltexts = [
            '雪花秀与润/雨润夜间睡眠修复面膜',
            '雪花秀玉容撕拉/玉璨净颜面膜'
           ]

  ltexts = [
      '日本原装进口soft stone腋下干爽止汗味去狐臭软石消臭石',
      '日本deona tulle腋下止汗膏soft stone消臭狐臭净腋汗露',
      '日本deona tulle soft stone消臭石除臭石腋下止汗露',
      ]
  ltexts = [
           '野生橄榄焕彩面贴膜',
           '爱沐空间野生橄榄焕采面贴膜',
           '【京东超市】雅诗兰黛（Estee Lauder）弹性紧实柔肤眼霜15ml(又名弹性紧实眼霜)（眼霜 提拉紧致 淡化细纹）',
           ]
  ltexts = [
      'Adidas阿迪达斯男士香水男士古龙淡香水 冰点男香100ml【京东超市】',
      '阿迪达斯男士女士走珠香体液50ml 走珠香水/香体滚珠 冰点激情征服清新跃动净怡 【京东超市】男士 激情+征服+天赋 3瓶',
      '阿迪达斯（adidas）男士香水-决战欧冠50ml',
      '阿迪达斯adidas男士香水 征服50ml'
      ]

  ltexts = [
      '雅诗兰黛（Estee Lauder）肌透修护眼部密集精华露 15ml（眼霜 ANR 滋润眼周 淡化细纹）',
      '【京东超市】雅诗兰黛（Estee Lauder）肌透修护眼部精华霜 15ml（眼霜 ANR 提拉紧致 淡化细纹 黑眼圈）'
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
