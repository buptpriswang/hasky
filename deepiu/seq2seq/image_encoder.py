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

import nets #slim nets
import tensorflow.contrib.slim as slim
from deepiu.seq2seq import embedding
from deepiu.seq2seq import rnn_encoder

import melt

class ImageEncoder(object):
  def __init__(self, is_training=False, is_predict=False, 
               emb_dim=None, initializer=None):
    """
    emb_dim means word emb dim
    """
    self.is_training = is_training
    self.is_predict = is_predict
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

def inception_resnet_v2(inputs, is_training=True,
                        dropout_keep_prob=0.8,
                        reuse=None,
                        scope='InceptionResnetV2Help'):
  """Creates the Inception Resnet V2 model.

  Args:
    inputs: a 4-D tensor of size [batch_size, height, width, 3].
    num_classes: number of predicted classes.
    is_training: whether is training or not.
    dropout_keep_prob: float, the fraction to keep before final layer.
    reuse: whether or not the network and its variables should be reused. To be
      able to reuse 'scope' must be given.
    scope: Optional variable_scope.
    create_aux_logits: Whether to include the auxilliary logits.

  Returns:
    logits: the logits outputs of the model.
    end_points: the set of end_points from the inception model.
  """

  with tf.variable_scope(scope, 'InceptionResnetV2Help', [inputs],
                         reuse=reuse) as scope:
    with slim.arg_scope([slim.batch_norm, slim.dropout],
                        is_training=is_training):
    
      net = inputs

      with tf.variable_scope('Logits'):
        net = slim.avg_pool2d(net, net.get_shape()[1:3], padding='VALID',
                              scope='AvgPool_1a_8x8')
        net = slim.flatten(net)

        net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                           scope='Dropout')

    return net
  
class MemoryEncoder(ShowAndTellEncoder):
  """
  """
  def __init__(self, is_training=False, is_predict=False,
               emb_dim=None, initializer=None):
    ShowAndTellEncoder.__init__(self, is_training, is_predict, emb_dim, initializer)

  def encode(self, image_features):
    image_emb = inception_resnet_v2(tf.reshape(image_features, [-1, 8, 8, 1536]), is_training=False)
 
    image_features = tf.concat([image_features, tf.expand_dims(image_emb, 1)], 1)    
    image_embs = self.build_image_embeddings(image_features)
    image_emb = image_embs[:,-1]
    image_embs = image_embs[:,:-1]
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
    #image_emb = tf.reduce_mean(image_embs, 1)

    return encoder_output, state, image_emb

class MemoryWithPosConcatEncoder(ShowAndTellEncoder):
  """
  """
  def __init__(self, is_training=False, is_predict=False,
               emb_dim=None, initializer=None):
    ShowAndTellEncoder.__init__(self, is_training, is_predict, emb_dim, initializer)
    self.pos_emb = embedding.get_embedding_cpu(name='pos_emb', height=FLAGS.image_attention_size)

  def encode(self, image_features):
    fe_dim = int(IMAGE_FEATURE_LEN / FLAGS.image_attention_size)
    attn_dim = int(math.sqrt(FLAGS.image_attention_size))
    image_emb = inception_resnet_v2(tf.reshape(image_features, [-1, attn_dim, attn_dim, fe_dim]), is_training=False)
 
    image_features = tf.concat([image_features, tf.expand_dims(image_emb, 1)], 1)    
    image_embs = self.build_image_embeddings(image_features)
    image_emb = image_embs[:,-1]
    image_embs = image_embs[:,:-1]
    #128,64,512 64,512
    image_embs = tf.concat([image_embs, tf.tile(tf.expand_dims(self.pos_emb, 0), [melt.get_batch_size(image_embs), 1, 1])], 1)
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
    #image_emb = tf.reduce_mean(image_embs, 1)

    return encoder_output, state, image_emb

class MemoryWithPosSumEncoder(ShowAndTellEncoder):
  """
  """
  def __init__(self, is_training=False, is_predict=False,
               emb_dim=None, initializer=None):
    ShowAndTellEncoder.__init__(self, is_training, is_predict, emb_dim, initializer)
    self.pos_emb = embedding.get_embedding_cpu(name='pos_emb', height=FLAGS.image_attention_size)

  def encode(self, image_features):
    fe_dim = int(IMAGE_FEATURE_LEN / FLAGS.image_attention_size)
    attn_dim = int(math.sqrt(FLAGS.image_attention_size))
    image_emb = inception_resnet_v2(tf.reshape(image_features, [-1, attn_dim, attn_dim, fe_dim]), is_training=False)
    
    image_features = tf.concat([image_features, tf.expand_dims(image_emb, 1)], 1)    
    image_embs = self.build_image_embeddings(image_features)
    image_emb = image_embs[:,-1]
    image_embs = image_embs[:,:-1]
    image_embs += self.pos_emb
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
    #image_emb = tf.reduce_mean(image_embs, 1)

    return encoder_output, state, image_emb

class BaselineMemoryEncoder(ShowAndTellEncoder):
  """
  my first show attend and tell implementation, this is baseline
  it works pointing to correct part, and convergent faster 
  but final result is not better then no attention
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
    self.encoder = rnn_encoder.RnnEncoder(is_training, is_predict)

  def encode(self, image_features):
    #TODO 8 8 1536
    fe_dim = int(IMAGE_FEATURE_LEN / FLAGS.image_attention_size)
    attn_dim = int(math.sqrt(FLAGS.image_attention_size))
    image_emb = inception_resnet_v2(tf.reshape(image_features, [-1, attn_dim, attn_dim, fe_dim]), is_training=False)
 
    image_features = tf.concat([image_features, tf.expand_dims(image_emb, 1)], 1)    
    image_embs = self.build_image_embeddings(image_features)
    image_emb = image_embs[:,-1]
    image_embs = image_embs[:,:-1]
    #to make it like rnn encoder outputs
    encoder_output, state = self.encoder.encode(image_embs, 
                                                embedding_lookup=False, 
                                                output_method=melt.rnn.OutputMethod.all)

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
  'BaselineMemory': BaselineMemoryEncoder,
  'Memory': MemoryEncoder,
  'MemoryWithPosSum': MemoryWithPosSumEncoder,
  'MemoryWithPosConcat': MemoryWithPosConcatEncoder,
  'ShowAttendAndTell': ShowAttendAndTellEncoder,
  'Rnn': RnnEncoder,
  'RnnController': RnnControllerEncoder
}
