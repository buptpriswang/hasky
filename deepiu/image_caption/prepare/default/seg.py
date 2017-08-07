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

flags.DEFINE_string('seg_method', 'basic', '')

assert FLAGS.seg_method is 'basic'

import nowarning

import sys,os
import numpy as np
import melt

import conf 
from conf import IMAGE_FEATURE_LEN

from gezi import Segmentor
segmentor = Segmentor()

START_WORD = '<S>'
END_WORD = '</S>'
NUM_WORD = '<NUM>'

print('seg_method:', FLAGS.seg_method, file=sys.stderr)

num = 0
for line in sys.stdin:
  if num % 10000 == 0:
    print(num, file=sys.stderr)
  l = line.rstrip().split('\t')
  
  texts = l[1].split('\x01')
  
  for text in texts:
    text = text.lower()
    words = segmentor.Segment(text, FLAGS.seg_method)
    if num % 10000 == 0:
      print(text, '|'.join(words), len(words), file=sys.stderr)
    wlist = [START_WORD]
    for word in words:
      if word.isdigit():
        wlist.append(NUM_WORD)
      else:
        wlist.append(word)
    wlist.append(END_WORD)
    print('\t'.join(wlist))
  num += 1
