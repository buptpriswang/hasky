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

#https://stackoverflow.com/questions/37479119/doing-pairwise-distance-computation-with-tensorflow
#https://hackernoon.com/one-shot-learning-with-siamese-networks-in-pytorch-8ddaab10340e
#https://hackernoon.com/facial-similarity-with-siamese-networks-in-pytorch-9642aa9db2f7
#http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf

#there are different versions, one is not to use sqrt.. just square
#input score not l2 distance withou sqrt
def contrastive(pos_score, neg_scores, margin=1.0, use_square=True, combiner='mean', name=None,):
  #relu same like hinge.. tf.max..
  #neg_scores = tf.nn.relu(margin - neg_scores)
  neg_scores = tf.nn.relu(margin - tf.sqrt(neg_scores))
  if use_square:  
    neg_scores = tf.square(neg_scores)
  else:
    pos_score = tf.sqrt(pos_score)
  
  scores = tf.concat([pos_score, neg_scores], 1)
  loss = reduce_loss(scores, combiner) * 0.5
  return loss

def triplet(pos_score, neg_scores, margin=1.0, combiner='mean', name=None,):
  #margin small then loss turn to zero quick, margin big better diff hard images but hard to converge
  #if pos_score(dist) is smaller then neg_score more then margin then loss is zero
  scores = tf.nn.relu(margin - (neg_scores - pos_score))
  return reduce_loss(scores, combiner)

#this is cross entorpy for cosine same... scores must be -1 <-> 1 TODO
def cross_entropy(scores, combiner='mean', name=None):
  with tf.name_scope(name, 'cross_entropy_loss', [scores]):
    batch_size = scores.get_shape()[0]
    num_negs = scores.get_shape()[1] - 1
    targets = tf.concat([tf.ones([batch_size, 1], tf.float32), tf.zeros([batch_size, num_negs], tf.float32)], 1)
    #http://www.wildml.com/2016/07/deep-learning-for-chatbots-2-retrieval-based-model-tensorflow/ 
    #I think for binary is same for sigmoid or softmax
    
    #logits = tf.sigmoid(scores)
    
    #logits = (1. + scores) / 2.

    logits = scores

    loss_matrix = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=targets)
    #loss_matrix = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=targets)
    loss = reduce_loss(loss_matrix, combiner)
    return loss

#---------below pairwise
def hinge(pos_score, neg_score, margin=0.1, combiner='mean', name=None):
  with tf.name_scope(name, 'hinge_loss', [pos_score, neg_score]):
    ##so if set 0.5 margin , will begin loss from 0.5 since pos and neg sore most the same at begin
    #loss_matrix = tf.maximum(0., margin - (pos_score - neg_score))
    loss_matrix = tf.nn.relu(margin - (pos_score - neg_score))
    loss = reduce_loss(loss_matrix, combiner)
    return loss

def pairwise_cross(pos_score, neg_score, combiner='mean', name=None):
  with tf.name_scope(name, 'hinge_cross_loss', [pos_score, neg_score]):
    score = pos_score - neg_score
    #logits = tf.sigmoid(score)
    logits = score
    targets = tf.ones_like(neg_score, tf.float32)
    loss_matrix = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=targets)
    loss = reduce_loss(loss_matrix, combiner)
    return loss

def pairwise_exp(pos_score, neg_score, theta=1.,  combiner='mean', name=None):
  with tf.name_scope(name, 'hinge_exp_loss', [pos_score, neg_score]):
    score = pos_score - neg_score
    loss = tf.log(1. + tf.exp(-theta * score))
    loss = reduce_loss(loss, combiner)
    return loss
