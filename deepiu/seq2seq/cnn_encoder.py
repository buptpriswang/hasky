#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   cnn_encoder.py
#        \author   chenghuige  
#          \date   2016-12-24 00:00:43.524179
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS
  
N_FILTERS = 10 #filters is as output
WINDOW_SIZE = 3
#FILTER_SHAPE1 = [WINDOW_SIZE, EMBEDDING_SIZE]
FILTER_SHAPE2 = [WINDOW_SIZE, N_FILTERS]
POOLING_WINDOW = 4
POOLING_STRIDE = 2

#https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/learn/text_classification_cnn.py
#may be change the interfance to encode(inputs) ?
def encode(sequence, emb=None):
  """2 layer ConvNet to predict from sequence of words to a class."""
  #[batch_size, length, emb_dim]
  word_vectors = tf.nn.embedding_lookup(emb, sequence) if emb else sequence
  word_vectors = tf.expand_dims(word_vectors, -1)
  emb_dim = word_vectors.get_shape()[-1]
  FILTER_SHAPE1 = [WINDOW_SIZE, emb_dim]
  with tf.variable_scope('CNN_Layer1'):
    # Apply Convolution filtering on input sequence.
    conv1 = tf.layers.conv2d(
        word_vectors,
        filters=N_FILTERS,
        kernel_size=FILTER_SHAPE1,
        padding='VALID',
        # Add a ReLU for non linearity.
        activation=tf.nn.relu)
    print('-----conv1', conv1)
    # Max pooling across output of Convolution+Relu.
    pool1 = tf.layers.max_pooling2d(
        conv1,
        pool_size=POOLING_WINDOW,
        strides=POOLING_STRIDE,
        padding='SAME')
    print('-----pool1', pool1)
    # Transpose matrix so that n_filters from convolution becomes width.
    pool1 = tf.transpose(pool1, [0, 1, 3, 2])
    print('-----pool1', pool1)
  with tf.variable_scope('CNN_Layer2'):
    # Second level of convolution filtering.
    conv2 = tf.layers.conv2d(
        pool1,
        filters=N_FILTERS,
        kernel_size=FILTER_SHAPE2,
        padding='VALID')
    print('-----conv2', conv2)
    # Max across each filter to get useful features for classification.
    pool2 = tf.squeeze(tf.reduce_max(conv2, 1), squeeze_dims=[1])

    print('-----pool2', pool2)
    #return pool2
    return tf.layers.dense(pool2, FLAGS.emb_dim)