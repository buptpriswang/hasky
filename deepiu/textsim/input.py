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

import melt

import conf
from conf import TEXT_MAX_WORDS

def _decode(example, parse):
  features = parse(
      example,
      features={
          'ltext_str': tf.FixedLenFeature([], tf.string),
          'ltext': tf.VarLenFeature(tf.int64),
          'rtext_str': tf.FixedLenFeature([], tf.string),
          'rtext': tf.VarLenFeature(tf.int64),
      })

  ltext = features['ltext']
  rtext = features['rtext']

  #for attention to be numeric stabel and since encoding not affect speed, dynamic rnn encode just pack zeros at last
  #input_maxlen = 0 if dynamic_batch_length else INPUT_TEXT_MAX_WORDS
  #lmaxlen = TEXT_MAX_WORDS
  #TODO... check affect..  for decomposable_nli as use masked softmax must use dynamic batch length for all!
  lmaxlen = 0 if FLAGS.dynamic_batch_length else TEXT_MAX_WORDS
  ltext = melt.sparse_tensor_to_dense(ltext, lmaxlen)
  
  rmaxlen = 0 if FLAGS.dynamic_batch_length else TEXT_MAX_WORDS
  rtext = melt.sparse_tensor_to_dense(rtext, rmaxlen)

  ltext_str = features['ltext_str']
  rtext_str = features['rtext_str']
  
  #--HACK TODO just to make sequence smae as image_caption image_name, image_feature, text, text_str
  return ltext_str, ltext, rtext, rtext_str

def decode_examples(serialized_examples,):
  return _decode(serialized_examples, tf.parse_example)

def decode_example(serialized_example):
  return _decode(serialized_example, tf.parse_single_example)


#-----------utils
def get_decodes(use_neg=True):
  if FLAGS.shuffle_then_decode:
    inputs = melt.shuffle_then_decode.inputs
    decode = lambda x: decode_examples(x)
    decode_neg = decode if use_neg else None
  else:
    assert False, 'since have sparse data must use shuffle_then_decode'
    inputs = melt.decode_then_shuffle.inputs
    decode = lambda x: decode_example(x)
    decode_neg = decode if use_neg else None

  return inputs, decode, decode_neg

def reshape_neg_tensors(neg_ops, batch_size, num_negs):
  neg_ops = list(neg_ops)
  for i in xrange(len(neg_ops)):
    #notice for strs will get [batch_size, num_negs, 1], will squeeze later
    neg_ops[i] = tf.reshape(neg_ops[i], [batch_size, num_negs,-1])
  return neg_ops
