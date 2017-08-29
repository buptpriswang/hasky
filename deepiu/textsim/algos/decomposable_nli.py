#!/usr/bin/env python
# ==============================================================================
#          \file   decomposable_nli.py
#        \author   chenghuige  
#          \date   2017-08-25 20:58:08.434879
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_boolean('nli_cosine', False, '')

import melt
logging = melt.logging

import melt.slim

import deepiu

from deepiu.textsim import conf
from deepiu.textsim.conf import TEXT_MAX_WORDS

from deepiu.util import vocabulary
from deepiu.seq2seq import embedding, encoder_factory

from deepiu.util.rank_loss import dot, compute_sim, pairwise_loss, normalize, PairwiseGraph

import numpy as np
import glob

#modified version from https://gist.github.com/marekgalovic/a1a4073b917ae1b18dc7413436794dca
#original paper from https://arxiv.org/pdf/1606.01933v1.pdf 
#A Decomposable Attention Model for Natural Language Inference

class DecomposableNLI(PairwiseGraph):
  def __init__(self, is_training=True, is_predict=False):
    super(DecomposableNLI, self).__init__()

    self.is_training = is_training
    self.is_predict = is_predict

    #TODO move to melt.EmbeddingTrainerBase
    emb_dim = FLAGS.emb_dim
    init_width = 0.5 / emb_dim
    vocabulary.init()
    vocab_size = vocabulary.get_vocab_size()
    self.vocab_size = vocab_size

    # cpu for adgrad optimizer
    self.emb = embedding.get_or_restore_embedding_cpu()

    melt.visualize_embedding(self.emb, FLAGS.vocab)
    if is_training and FLAGS.monitor_level > 0:
      melt.monitor_embedding(self.emb, vocabulary.vocab, vocab_size)

    self._attention_output_size = 256
    self._comparison_output_size = 128

    self.scope = 'decomposable_nli'
    self.build_train_graph = self.build_graph

  #----------below is core algo
  #for train u, v is slim already! assume input.py did this
  def compute_sim(self, u, v):
    u_len = melt.length(u)
    v_len = melt.length(v)
    u, v = self.embeddings_lookup(u, v)
    alpha, beta = self.attention_layer(u, v, u_len, v_len)
    u, v = self.comparison_layer(u, v, alpha, beta, u_len, v_len)
    u, v = self.aggregation_layer(u, v)
    
    if not FLAGS.nli_cosine:
    	f = tf.concat([u, v], 1)
    	#f = tf.concat((u, v, tf.abs(u-v), u*v), 1)
    	score = self.fc_layer(f)
    	#score = tf.sigmoid(score)
    else:
    	score = melt.element_wise_cosine(u, v)

    return score

  def fc_layer(self, f):
    return melt.slim.mlp(f,
                 [100, 1],
                 #activation_fn=tf.nn.tanh,
                 #activation_fn=None,
                 activation_fn=tf.nn.relu,
                 dropout=FLAGS.dropout,
                 training=self.is_training,
                 biases_initializer=None,
                 scope='fc')

  def embeddings_lookup(self, u, v):
    with tf.name_scope('embeddings_lookup'):
      return tf.nn.embedding_lookup(self.emb, u), tf.nn.embedding_lookup(self.emb, v)

  def attention_layer(self, u, v, u_len, v_len):
    with tf.name_scope('attention_layer'):
      e_u = tf.layers.dense(u, self._attention_output_size, activation=tf.nn.relu, name='attention_nn')
      e_v = tf.layers.dense(v, self._attention_output_size, activation=tf.nn.relu, name='attention_nn', reuse=True)

      e = tf.matmul(e_u, e_v, transpose_b=True, name='e')
        
      alpha = tf.matmul(self.masked_softmax(tf.transpose(e, [0,2,1]), u_len), u, name='alpha')
      beta = tf.matmul(self.masked_softmax(e, v_len), v, name='beta')
      
      return alpha, beta

  def masked_softmax(self, values, lengths):
    with tf.name_scope('masked_softmax'): 
      mask = tf.expand_dims(tf.sequence_mask(lengths, dtype=tf.float32), -2)    
      inf_mask = (1 - mask) * -np.inf
      inf_mask = tf.where(tf.is_nan(inf_mask), tf.zeros_like(inf_mask), inf_mask)
      return tf.nn.softmax(tf.multiply(values, mask) + inf_mask)

  def comparison_layer(self, u, v, alpha, beta, u_len, v_len):
    with tf.name_scope('comparison_layer'):
      u_comp = tf.layers.dense(
          tf.concat([u, beta], 2),
          self._comparison_output_size,
          activation=tf.nn.relu,
          name='comparison_nn'
      )
      u_comp = tf.multiply(
          u_comp,
          tf.expand_dims(tf.sequence_mask(u_len, dtype=tf.float32), -1)
      )

      v_comp = tf.layers.dense(
          tf.concat([v, alpha], 2),
          self._comparison_output_size,
          activation=tf.nn.relu,
          name='comparison_nn',
          reuse=True
      )
      v_comp = tf.multiply(
          v_comp,
          tf.expand_dims(tf.sequence_mask(v_len, dtype=tf.float32), -1)
      )

      return u_comp, v_comp
            
  def aggregation_layer(self, u, v):
    with tf.name_scope('aggregation_layer'):
      u_agg = tf.reduce_sum(u, 1)
      v_agg = tf.reduce_sum(v, 1)

      return u_agg, v_agg

#TODO move to melt.ElemetwiseTextSimPrecitor
class DecomposableNLIPredictor(DecomposableNLI, melt.PredictorBase):
  def __init__(self):
    melt.PredictorBase.__init__(self)
    DecomposableNLI.__init__(self, is_training=False, is_predict=True)

    self.ltext_feed = None
    self.rtext_feed = None

  #TODO move to TextSimPredictorBase
  #not in init for dont want to put under name space 'model_init'
  def get_ltext_feed(self, text_max_words=TEXT_MAX_WORDS):
    if self.ltext_feed is None:
      self.ltext_feed = tf.placeholder(tf.int32, [None, text_max_words], name='ltext')
    return self.ltext_feed

  def get_rtext_feed(self, text_max_words=TEXT_MAX_WORDS):
    if self.rtext_feed is None:
      self.rtext_feed = tf.placeholder(tf.int32, [None, text_max_words], name='rtext')
    return self.rtext_feed

  def init_predict(self, text_max_words=TEXT_MAX_WORDS):
    if self.ltext_feed is None:
      ltext_feed = self.get_ltext_feed(text_max_words)
      rtext_feed = self.get_rtext_feed(text_max_words)
      score = self.score = self.build_graph(ltext_feed, rtext_feed)
      tf.add_to_collection('score', score)
      tf.get_variable_scope().reuse_variables()
    return score

  def predict(self, ltexts, rtexts):
    feed_dict = {
      self.ltext_feed: ltexts,
      self.rtext_feed: rtexts
      }
    
    score = self.sess.run(self.score, feed_dict)
    return score

  #TODO may be speedup by cocant [ltex, rtext] as one batch and co forward then split 
  def build_graph(self, ltext, rtext):
    with tf.variable_scope(self.scope):
      ltext = melt.slim_batch(ltext)
      rtext = melt.slim_batch(rtext)
      score = self.compute_sim(ltext, rtext)
      return score
