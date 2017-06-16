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
import melt

import conf
from conf import TEXT_MAX_WORDS, INPUT_TEXT_MAX_WORDS

def _decode(example, parse, dynamic_batch_length):
  features = parse(
      example,
      features={
          'text_str': tf.FixedLenFeature([], tf.string),
          'text': tf.VarLenFeature(tf.int64),
          'input_text_str': tf.FixedLenFeature([], tf.string),
          'input_text': tf.VarLenFeature(tf.int64),
      })

  text = features['text']
  input_text = features['input_text']

  maxlen = 0 if dynamic_batch_length else TEXT_MAX_WORDS
  text = melt.sparse_tensor_to_dense(text, maxlen)
  
  #for attention to be numeric stabel and since encoding not affect speed, dynamic rnn encode just pack zeros at last
  #input_maxlen = 0 if dynamic_batch_length else INPUT_TEXT_MAX_WORDS
  input_maxlen = INPUT_TEXT_MAX_WORDS
  input_text = melt.sparse_tensor_to_dense(input_text, input_maxlen)

  text_str = features['text_str']
  input_text_str = features['input_text_str']
  
  try:
    image_name = features['image_name']
  except Exception:
    image_name = text_str

  return image_name, text, text_str, input_text, input_text_str

def decode_examples(serialized_examples, dynamic_batch_length):
  return _decode(serialized_examples, tf.parse_example, dynamic_batch_length)

def decode_example(serialized_example, dynamic_batch_length):
  return _decode(serialized_example, tf.parse_single_example, dynamic_batch_length)


#-----------utils
def get_decodes(shuffle_then_decode, dynamic_batch_length):
  if shuffle_then_decode:
    inputs = melt.shuffle_then_decode.inputs
    decode = lambda x: decode_examples(x, dynamic_batch_length)
  else:
    inputs = melt.decode_then_shuffle.inputs
    decode = lambda x: decode_example(x, dynamic_batch_length)
  return inputs, decode

