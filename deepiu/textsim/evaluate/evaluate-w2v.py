#!/usr/bin/env python
#encoding=gbk
# ==============================================================================
#          \file   test_embsim.py
#        \author   chenghuige  
#          \date   2017-08-09 16:25:02.455936
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os,glob
import tensorflow as tf
import melt 


from libword_counter import Vocabulary

from deepiu.util import text2ids

dir = '/home/gezi/new/temp/makeup/title2name/tfrecord/seq-basic/'

text2ids.init(os.path.join(dir, 'vocab.txt'))
vocab = text2ids.vocab

embsim = melt.EmbeddingSim(os.path.join(dir, 'word2vec'), name='w_in')

corpus_pattern = os.path.join('/home/gezi/data/product/makeup/tb/title2name/valid/*')

max_words = 50
#itexts = ['雅诗兰黛水润霜', '雅诗兰黛小棕瓶', '雅诗兰黛红石榴', '婷美矿物泉补水精华', 'Adidas阿迪达斯男士香水男士古龙淡香水 冰点男香100ml【京东超市】']
itexts = ['美宝莲bb霜/cc霜 BB霜 气垫BB 粉饼 象牙白替换装*2+气垫外盒', 
            '【京东超市】美宝莲（MAYBELLINE）超然无瑕轻垫霜01亮肤色 14g（巨遮瑕 轻薄裸妆 滋润保湿 隔离）']

itexts = ['亚缇克兰水莹润肌洁面慕丝洗面奶深层清洁控油洁面乳补水滋润正品']

left_ids = [text2ids.text2ids(x, seg_method='basic', feed_single=True, max_words=max_words) for x in itexts]


lids_ = tf.placeholder(dtype=tf.int32, shape=[None, max_words]) 
rids_ = tf.placeholder(dtype=tf.int32, shape=[None, max_words]) 
nids_ = embsim.top_sim(lids_, rids_)
sess = embsim._sess 

corpus_text = []
for file in glob.glob(corpus_pattern):
  corpus_text += open(file).readlines()
corpus_text = [x.strip() for x in corpus_text]

r_text = [x.split('\t')[1] for x in corpus_text]
r_text = list(set(r_text))
right_ids = [text2ids.text2ids(x, seg_method='basic', feed_single=True, max_words=max_words) for x in r_text] 
print(len(corpus_text))

values, indices = sess.run(nids_, {lids_: left_ids, rids_ : right_ids})

for i in xrange(len(itexts)):
  print('-----------[input]:', itexts[i])
  for index, value in zip(indices[i], values[i]):
    print(r_text[index], value)

r_text = [x.split('\t')[0] for x in corpus_text]
r_text = list(set(r_text))
right_ids = [text2ids.text2ids(x, seg_method='basic', feed_single=True, max_words=max_words) for x in r_text] 
print(len(corpus_text))

values, indices = sess.run(nids_, {lids_: left_ids, rids_ : right_ids})

for i in xrange(len(itexts)):
  print('-----------[input]:', itexts[i])
  for index, value in zip(indices[i], values[i]):
    print(r_text[index], value)
