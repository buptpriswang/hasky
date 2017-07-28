#!/usr/bin/env python
# ==============================================================================
#          \file   input.py
#        \author   chenghuige  
#          \date   2016-08-17 23:50:47.335840
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS


import functools

import melt

def _decode(example, parse):
  features_dict = {
     'article': tf.FixedLenFeature([], tf.string),
     'abstract': tf.FixedLenFeature([], tf.string),
    }

  features = parse(example, features=features_dict)

  article = features['article']
  abstract = features['abstract']

  return article, abstract

def decode_examples(examples):
  return _decode(examples, tf.parse_example)

def decode_example(example):
  return _decode(example, tf.parse_single_example)

#-----------utils
def get_decodes():
  if FLAGS.shuffle_then_decode:
    inputs = melt.shuffle_then_decode.inputs
    decode = lambda x: decode_examples(x)
  else:
    assert False, 'since have sparse data must use shuffle_then_decode'
    inputs = melt.decode_then_shuffle.inputs
    decode = lambda x: decode_example(x)

  return inputs, decode
