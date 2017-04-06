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
from conf import IMAGE_FEATURE_LEN,TEXT_MAX_WORDS

def _decode(example, parse, dynamic_batch_length, is_training=False, reuse=None):
  if FLAGS.pre_calc_image_feature:
    features = parse(
        example,
        features={
            'image_name': tf.FixedLenFeature([], tf.string),
            'image_feature': tf.FixedLenFeature([IMAGE_FEATURE_LEN], tf.float32),
            'text': tf.VarLenFeature(tf.int64),
            'text_str': tf.FixedLenFeature([], tf.string),
        })
  else:
    features = parse(
        example,
        features={
            'image_name': tf.FixedLenFeature([], tf.string),
            'image_data': tf.FixedLenFeature([], dtype=tf.string),
            'text': tf.VarLenFeature(tf.int64),
            'text_str': tf.FixedLenFeature([], tf.string),
        })

  image_name = features['image_name']
  if FLAGS.pre_calc_image_feature:
    image_feature = features['image_feature']
  else:
    image_feature = features['image_data']
    #print('---------------', image_feature)
    image_feature = tf.map_fn(lambda img: melt.image.process_image(img,
                                                                   is_training,
                                                                   height=FLAGS.image_height, 
                                                                   width=FLAGS.image_width,
                                                                   distort=FLAGS.distort_image), 
                              image_feature,
                              dtype=tf.float32)

    image_feature = melt.image.inception_v3(
        image_feature,
        trainable=False,
        #trainable=True,
        is_training=is_training,
        reuse=reuse)

  text = features['text']
  maxlen = 0 if dynamic_batch_length else TEXT_MAX_WORDS
  text = melt.sparse_tensor_to_dense(text, maxlen)
  text_str = features['text_str']
  
  return image_name, image_feature, text, text_str

def decode_examples(serialized_examples, dynamic_batch_length, is_training=False, reuse=None):
  return _decode(serialized_examples, tf.parse_example, dynamic_batch_length,
                 is_training=is_training, reuse=reuse)

def decode_example(serialized_example, dynamic_batch_length, is_training=False, reuse=None):
  return _decode(serialized_example, tf.parse_single_example, dynamic_batch_length, 
                 is_training=is_training, reuse=reuse)

#---------------for negative sampling using tfrecords
def _decode_neg(example, parse, dynamic_batch_length):
  features = parse(
      example,
      features={
          'text': tf.VarLenFeature(tf.int64),
          'text_str': tf.FixedLenFeature([], tf.string),
      })

  text = features['text']
  maxlen = 0 if dynamic_batch_length else TEXT_MAX_WORDS
  text = melt.sparse_tensor_to_dense(text, maxlen)
  text_str = features['text_str']
  
  return text, text_str

def decode_neg_examples(serialized_examples, dynamic_batch_length):
  return _decode_neg(serialized_examples, tf.parse_example, dynamic_batch_length)

def decode_neg_example(serialized_example):
  return _decode_neg(serialized_example, tf.parse_single_example, dynamic_batch_length)

#-----------utils
def get_decodes(shuffle_then_decode, dynamic_batch_length, use_neg=True):
  if shuffle_then_decode:
    inputs = melt.shuffle_then_decode.inputs
    #TODO inception model first used in evaluator.py init... so here all reuse=True  TODO may not depend on building sequence ?
    decode_train = lambda x: decode_examples(x, dynamic_batch_length, is_training=True, reuse=True)
    decode = lambda x: decode_examples(x, dynamic_batch_length, reuse=True)
    decode_neg = (lambda x: decode_neg_examples(x, dynamic_batch_length)) if use_neg else None
  else:
    inputs = melt.decode_then_shuffle.inputs
    decode_train = lambda x: decode_example(x, dynamic_batch_length, is_training=True, reuse=True)
    decode = lambda x: decode_example(x, dynamic_batch_length, reuse=True)
    decode_neg = (lambda x: decode_neg_example(x, dynamic_batch_length)) if use_neg else None
  return inputs, decode, decode_neg, decode_train

def reshape_neg_tensors(neg_ops, batch_size, num_negs):
  neg_ops = list(neg_ops)
  for i in xrange(len(neg_ops)):
    #notice for strs will get [batch_size, num_negs, 1], will squeeze later
    neg_ops[i] = tf.reshape(neg_ops[i], [batch_size, num_negs,-1])
  return neg_ops
