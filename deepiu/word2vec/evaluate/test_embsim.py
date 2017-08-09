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

dir = '/home/gezi/new/temp/makeup/title2name/tfrecord/seq-basic/'

vocab = Vocabulary(os.path.join(dir, 'vocab.txt'), 1)

embsim = melt.EmbeddingSim(os.path.join(dir, 'word2vec'), name='w_in')

wid = vocab.id('曼秀雷敦') 

wid_ = tf.placeholder(dtype=tf.int32, shape=[None,1]) 
nids_ = embsim.nearby(wid_)

sess = embsim._sess 

#nids = sess.run(nids_, {wid_: wid})
values, indices = sess.run(nids_, {wid_: [[wid]]})

for index, value in zip(indices[0], values[0]):
  print(vocab.key(int(index)), value)
