#!/usr/bin/env python
# ==============================================================================
#          \file   image_encoder.py
#        \author   chenghuige  
#          \date   2017-09-17 21:57:04.147177
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS

import sys, os

class ImageEncoder(object):
  def __init__(self, is_training=False, is_predict=False, 
               emb_dim=None, initializer=None):
    """
    emb_dim means word emb dim
    """
    self.emb_dim = emb_dim
    self.initializer = initializer or tf.random_uniform_initializer(minval=-FLAGS.initializer_scale, maxval=FLAGS.initializer_scale)

class ShowAndTellEncoder(ImageEncoder):
  def __init__(self, is_training=False, is_predict=False,
               emb_dim=None, initializer=None):
    ImageEncoder.__init__(self, is_training, is_predict, emb_dim, initializer)

  def build_image_embeddings(self, image_feature):
    with tf.variable_scope("image_embedding") as scope:
      image_embeddings = tf.contrib.layers.fully_connected(
          inputs=image_feature,
          num_outputs=self.emb_dim,  #turn image to word embdding space, same feature length
          activation_fn=None,
          weights_initializer=self.initializer,
          biases_initializer=None,
          scope=scope)
    return image_embeddings

  def encode(self, image_feature):
    encoder_output = None
    state = None
    image_emb = self.build_image_embeddings(image_feature)
    return encoder_output, state, image_emb

class SimpleMemoryEncoder(ShowAndTellEncoder):
  def __init__(self, is_training=False, is_predict=False,
               emb_dim=None, initializer=None):
    ShowAndTellEncoder.__init__(self, is_training, is_predict, emb_dim, initializer)

  def encode(self, image_features):
    image_embs = self.build_image_embeddings(image_features)
    encoder_output = image_emb
    state = None
    image_emb = tf.reduce_mean(image_embs, 1)
    return encoder_output, state, image_emb
  
class MemoryEncoder(ShowAndTellEncoder):
  """
  my first show attend and tell implementation, this is baseline
  """
  def __init__(self, is_training=False, is_predict=False,
               emb_dim=None, initializer=None):
    ShowAndTellEncoder.__init__(self, is_training, is_predict, emb_dim, initializer)

  def encode(self, image_features):
    image_embs = self.build_image_embeddings(image_features)
    #to make it like rnn encoder outputs
    with tf.variable_scope("attention_embedding") as scope:
      encoder_output = tf.contrib.layers.fully_connected(
          inputs=image_embs,
          num_outputs=FLAGS.rnn_hidden_size,
          activation_fn=None,
          weights_initializer=self.initializer,
          biases_initializer=None,
          scope=scope)
    state = None
    image_emb = tf.reduce_mean(image_embs, 1)
    return encoder_output, state, image_emb

class ShowAttendAndTellEncoder(ImageEncoder):
  """
  strictly follow paper and copy from  @yunjey 's implementation of show-attend-and-tell
  https://github.com/yunjey/show-attend-and-tell
  this is like im2txt must add_text_start pad <S> before original text
  """
  def __init__(self, is_training=False, is_predict=False):
    ImageEncoder.__init__(self, is_training=False, is_predict=False)

  def get_initial_lstm(self, features):
    with tf.variable_scope('initial_lstm') as scope:
      features_mean = tf.reduce_mean(features, 1)
      with tf.variable_scope("h_embedding") as scope:
        h = tf.contrib.layers.fully_connected(
              inputs=features_mean,
              num_outputs=FLAGS.rnn_hidden_size,
              activation_fn=tf.nn.tanh,
              weights_initializer=self.initializer,
              scope=scope)
      with tf.variable_scope("c_embedding") as scope:
        c = tf.contrib.layers.fully_connected(
              inputs=features_mean,
              num_outputs=FLAGS.rnn_hidden_size,
              activation_fn=tf.nn.tanh,
              weights_initializer=self.initializer,
              scope=scope)
    return c, h

  def encode(self, image_features):
    image_emb = None
    encoder_output = image_features
    state = self.get_initial_lstm(image_features)
    return encoder_output, state, image_emb


class RnnEncoder(ShowAndTellEncoder):
  """
  using rnn to encode image features
  """
  def __init__(self, is_training=False, is_predict=False,
               emb_dim=None, initializer=None):
    ShowAndTellEncoder.__init__(self, is_training, is_predict, emb_dim, initializer)

  def encode(self, image_features):
    image_embs = self.build_image_embeddings(image_features)
    #to make it like rnn encoder outputs
    with tf.variable_scope("attention_embedding") as scope:
      encoder_output = tf.contrib.layers.fully_connected(
          inputs=image_embs,
          num_outputs=FLAGS.rnn_hidden_size,
          activation_fn=None,
          weights_initializer=self.initializer,
          biases_initializer=None,
          scope=scope)
    state = None
    image_emb = tf.reduce_mean(image_embs, 1)
    return encoder_output, state, image_emb

class RnnControllerEncoder(ShowAndTellEncoder):
  """
  using rnn/lstm controller for some steps to encode image features
  """
  def __init__(self, is_training=False, is_predict=False,
               emb_dim=None, initializer=None):
    ShowAndTellEncoder.__init__(self, is_training, is_predict, emb_dim, initializer)

  def encode(self, image_features):
    image_embs = self.build_image_embeddings(image_features)
    #to make it like rnn encoder outputs
    with tf.variable_scope("attention_embedding") as scope:
      encoder_output = tf.contrib.layers.fully_connected(
          inputs=image_embs,
          num_outputs=FLAGS.rnn_hidden_size,
          activation_fn=None,
          weights_initializer=self.initializer,
          biases_initializer=None,
          scope=scope)
    state = None
    image_emb = tf.reduce_mean(image_embs, 1)
    return encoder_output, state, image_emb

Encoders = {
  'ShowAndTell': ShowAndTellEncoder,
  'SimpleMemory': SimpleMemoryEncoder,
  'Memory': MemoryEncoder,
  'ShowAttendAndTell': ShowAttendAndTellEncoder,
  'Rnn': RnnEncoder,
  'RnnController': RnnControllerEncoder
}
