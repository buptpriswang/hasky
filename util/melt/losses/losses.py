#!/usr/bin/env python
# ==============================================================================
#          \file   losses.py
#        \author   chenghuige  
#          \date   2017-08-07 13:11:10.672206
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
#from __future__ import print_function

import sys, os

import tensorflow as tf 

def reduce_loss(loss_matrix, combiner='mean'):
  if combiner == 'mean':
    return tf.reduce_mean(loss_matrix)
  else:
    return tf.reduce_sum(loss_matrix)

def hinge(pos_score, neg_score, margin=0.1, combiner='mean', name=None):
  with tf.name_scope(name, 'hinge_loss', [pos_score, neg_score]):
    loss_matrix = tf.maximum(0., margin - (pos_score - neg_score))
    loss = reduce_loss(loss_matrix, combiner)
    return loss

def cross_entropy(scores, num_negs=1, combiner='mean', name=None):
  with tf.name_scope(name, 'cross_entropy_loss', [scores]):
    batch_size = scores.get_shape()[0]
    targets = tf.concat([tf.ones([batch_size, 1], tf.float32), tf.zeros([batch_size, num_negs], tf.float32)], 1)
    #http://www.wildml.com/2016/07/deep-learning-for-chatbots-2-retrieval-based-model-tensorflow/ 
    #I think for binary is same for sigmoid or softmax
    logits = tf.sigmoid(scores)
    loss_matrix = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=targets)
    #loss_matrix = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=targets)
    loss = reduce_loss(loss_matrix, combiner)
    return loss

def hinge_cross(pos_score, neg_score, combiner='mean', name=None):
  with tf.name_scope(name, 'hinge_cross_loss', [pos_score, neg_score]):
    logits = pos_score - neg_score
    logits = tf.sigmoid(logits)
    targets = tf.ones_like(neg_score, tf.float32)
    loss_matrix = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=targets)
    loss = reduce_loss(loss_matrix, combiner)
    return loss
