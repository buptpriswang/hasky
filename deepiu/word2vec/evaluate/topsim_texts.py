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

import sys, os
import tensorflow as tf
import melt 

from libword_counter import Vocabulary

from deepiu.util import text2ids

dir = '/home/gezi/new/temp/makeup/title2name/tfrecord/seq-basic/'

text2ids.init(os.path.join(dir, 'vocab.txt'))
vocab = text2ids.vocab

embsim = melt.EmbeddingSim(os.path.join(dir, 'word2vec'), name='w_in')

corpus_file = os.path.join('/home/gezi/data/product/makeup/tb/title2name/valid/name.filtered.rand.valid.txt_0')

max_words = 50
itexts = ['雅诗兰黛水润霜', '雅诗兰黛小棕瓶', '雅诗兰黛红石榴', '婷美矿物补水精华', 'Adidas阿迪达斯男士香水男士古龙淡香水 冰点男香100ml【京东超市】']

left_ids = [text2ids.text2ids(x, seg_method='basic', feed_single=True, max_words=max_words) for x in itexts]

corpus_text = open(corpus_file).readlines()

corpus_text = [x.split()[0] for x in corpus_text]

right_ids = [text2ids.text2ids(x, seg_method='basic', feed_single=True, max_words=max_words) for x in corpus_text[:10000]]

lids_ = tf.placeholder(dtype=tf.int32, shape=[None, max_words]) 
rids_ = tf.placeholder(dtype=tf.int32, shape=[None, max_words]) 
nids_ = embsim.top_sim(lids_, rids_)
sess = embsim._sess 
values, indices = sess.run(nids_, {lids_: left_ids, rids_ : right_ids})

for i in xrange(len(itexts)):
  print('-----------[input]:', itexts[i])
  for index, value in zip(indices[i], values[i]):
    print(corpus_text[index], value)
