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

try:
  import conf
  from conf import IMAGE_FEATURE_LEN,TEXT_MAX_WORDS
except Exception:
  from deepiu.image_caption.conf import IMAGE_FEATURE_LEN, TEXT_MAX_WORDS


def _decode(example, parse):
  features_dict = {
     'image_name': tf.FixedLenFeature([], tf.string),
     FLAGS.decode_name: tf.VarLenFeature(tf.int64),
     FLAGS.decode_str_name: tf.FixedLenFeature([], tf.string),
    }

  if FLAGS.pre_calc_image_feature:
    features_dict[FLAGS.image_feature_name] = tf.FixedLenFeature([IMAGE_FEATURE_LEN], tf.float32)
  else:
    features_dict['image_data'] = tf.FixedLenFeature([], dtype=tf.string)

  features = parse(example, features=features_dict)

  image_name = features['image_name']
  if FLAGS.pre_calc_image_feature:
    image_feature = features[FLAGS.image_feature_name]
  else:
    image_feature = features['image_data']

  text_str = features[FLAGS.decode_str_name]
  
  text = features[FLAGS.decode_name]
  maxlen = 0 if FLAGS.dynamic_batch_length else TEXT_MAX_WORDS
  text = melt.sparse_tensor_to_dense(text, maxlen)

  return image_name, image_feature, text, text_str

def decode_examples(examples):
  return _decode(examples, tf.parse_example)

def decode_example(example):
  return _decode(example, tf.parse_single_example)

def decode_sequence_example(example):
  context_features_dict = {
     'image_name': tf.FixedLenFeature([], tf.string),
     FLAGS.decode_str_name: tf.FixedLenFeature([], tf.string),
    }

  if FLAGS.pre_calc_image_feature:
    context_features_dict[FLAGS.image_feature_name] = tf.FixedLenFeature([IMAGE_FEATURE_LEN], tf.float32)
  else:
    context_features_dict['image_data'] = tf.FixedLenFeature([], dtype=tf.string)

  features, sequence_features = tf.parse_single_sequence_example(example, 
                                              context_features=context_features_dict,
                                              sequence_features={
                                                 FLAGS.decode_name: tf.FixedLenSequenceFeature([], dtype=tf.int64)
                                                })

  image_name = features['image_name']
  if FLAGS.pre_calc_image_feature:
    image_feature = features[FLAGS.image_feature_name]
  else:
    image_feature = features['image_data']
  text_str = features[FLAGS.decode_str_name]

  text = sequence_features[FLAGS.decode_name]
  
  return image_name, image_feature, text, text_str

#---------------for negative sampling using tfrecords
def _decode_neg(example, parse):
  features = parse(
      example,
      features={
          FLAGS.decode_name: tf.VarLenFeature(tf.int64),
          FLAGS.decode_str_name: tf.FixedLenFeature([], tf.string),
      })

  text = features[FLAGS.decode_name]
  maxlen = 0 if FLAGS.dynamic_batch_length else TEXT_MAX_WORDS
  text = melt.sparse_tensor_to_dense(text, maxlen)
  text_str = features[FLAGS.decode_str_name]
  
  return text, text_str

def decode_neg_examples(examples):
  return _decode_neg(examples, tf.parse_example)

def decode_neg_example(example):
  return _decode_neg(example, tf.parse_single_example)

def decode_neg_sequence_example(example):
  features, sequence_features = tf.parse_single_sequence_example(
      example,
      context_features={
          FLAGS.decode_str_name: tf.FixedLenFeature([], tf.string),
      },
      sequence_features={
          FLAGS.decode_name: tf.FixedLenSequenceFeature([], dtype=tf.int64),
        })

  text_str = features[FLAGS.decode_str_name]
  text = sequence_features[FLAGS.decode_name]
  
  return text, text_str

#-----------utils
def get_decodes(use_neg=True):
  if FLAGS.is_sequence_example:
    assert FLAGS.dynamic_batch_length, 'sequence example must be dyanmic batch length for fixed input'
    inputs = functools.partial(melt.decode_then_shuffle.inputs,
                               dynamic_pad=True,
                               bucket_boundaries=FLAGS.buckets,
                               length_fn=lambda x: tf.shape(x[-2])[-1])
    decode = lambda x: decode_sequence_example(x)
    assert not (use_neg and FLAGS.buckets), 'if use neg, discriminant method do not use buckets'
    decode_neg = (lambda x: decode_neg_sequence_example(x)) if use_neg else None
  else:
    if FLAGS.shuffle_then_decode:
      inputs = melt.shuffle_then_decode.inputs
      decode = lambda x: decode_examples(x)
      decode_neg = (lambda x: decode_neg_examples(x)) if use_neg else None
      print('decode_neg', decode_neg)
    else:
      assert False, 'since have sparse data must use shuffle_then_decode'
      inputs = melt.decode_then_shuffle.inputs
      decode = lambda x: decode_example(x)
      decode_neg = (lambda x: decode_neg_example(x)) if use_neg else None

  return inputs, decode, decode_neg

def reshape_neg_tensors(neg_ops, batch_size, num_negs):
  neg_ops = list(neg_ops)
  for i in xrange(len(neg_ops)):
    #notice for strs will get [batch_size, num_negs, 1], will squeeze later
    neg_ops[i] = tf.reshape(neg_ops[i], [batch_size, num_negs,-1])
  return neg_ops
