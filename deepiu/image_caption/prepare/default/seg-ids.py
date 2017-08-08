#!/usr/bin/env python
# ==============================================================================
#          \file   to_flickr_caption.py
#        \author   chenghuige  
#          \date   2016-07-11 16:29:27.084402
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""
@TODO could do segment parallel @TODO
now single thread... slow
"""

import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('vocab', None, 'vocabulary file')
flags.DEFINE_string('seg_method_', 'basic', '')
flags.DEFINE_bool('feed_single_', True, '')

import gezi.nowarning

import sys,os
import numpy as np
import melt

import conf 
from conf import IMAGE_FEATURE_LEN

from deepiu.util import text2ids

text2ids.init()

START_WORD = '<S>'
END_WORD = '</S>'
NUM_WORD = '<NUM>'

print('seg_method:', FLAGS.seg_method_, file=sys.stderr)

num = 0
for line in sys.stdin:
  if num % 10000 == 0:
    print(num, file=sys.stderr)
  l = line.rstrip().split('\t')
  
  texts = l[1].split('\x01')
  
  for text in texts:
    ids = text2ids.text2ids(text, 
        seg_method=FLAGS.seg_method_,
        feed_single=FLAGS.feed_single_, 
        allow_all_zero=True, 
        pad=False, 
        append_start=True,
        append_end=True,
        to_lower=True,
        norm_digit=True)
    if num % 10000 == 0:
      print(ids, file=sys.stderr)
      print(text2ids.ids2text(ids), file=sys.stderr)
    ids = map(str, ids)
    if ids:
      print('\t'.join(ids))
  num += 1
