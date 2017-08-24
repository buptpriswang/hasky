#!/usr/bin/env python
# ==============================================================================
#          \file   dual_textsim.py
#        \author   chenghuige
#          \date   2017-08-10 06:07:20.870443
#   \Description
# ==============================================================================


from __future__ import absolute_import
from __future__ import division
# from __future__ import print_function

import sys
import os, glob
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('mlp_dims', '256', '')
flags.DEFINE_string('rtext_bow', False, 'rtext alwyas use bow')

import melt
logging = melt.logging

import melt.slim

import deepiu

from deepiu.textsim import conf
from deepiu.textsim.conf import TEXT_MAX_WORDS

from deepiu.util import vocabulary
from deepiu.seq2seq import embedding, encoder_factory

from deepiu.util.rank_loss import dot, compute_sim, pairwise_loss, normalize

import numpy as np


#from melt import dot

class DualTextsim(object):
  """
  similar to http://www.wildml.com/2016/07/deep-learning-for-chatbots-2-retrieval-based-model-tensorflow/
  """

  def __init__(self, encoder_type='bow', is_training=True, is_predict=False):
    super(DualTextsim, self).__init__()

    self.is_training = is_training
    self.is_predict = is_predict

    self.encoder = encoder_factory.get_encoder(encoder_type, is_training, is_predict)
    self.encoder_type = encoder_type

    emb_dim = FLAGS.emb_dim
    init_width = 0.5 / emb_dim
    vocabulary.init()
    vocab_size = vocabulary.get_vocab_size()
    self.vocab_size = vocab_size

    # cpu for adgrad optimizer
    if (not FLAGS.word_embedding_file) or glob.glob(FLAGS.model_dir + '/model.ckpt*'):
      logging.info('Word embedding random init or from model_dir :{} and finetune=:{}'.format(
          FLAGS.model_dir, FLAGS.finetune_word_embedding))
      self.emb = embedding.get_embedding_cpu(
          name='emb', trainable=FLAGS.finetune_word_embedding)
    else:
      # https://github.com/tensorflow/tensorflow/issues/1570
      # still adgrad must cpu..
      # if not fintue emb this will be ok if fintune restart will ok ? must not use word embedding file? os.path.exists(FLAGS.model_dir) ? judge?
      # or will still try to load from check point ? TODO for safe you could re run by setting word_embedding_file as None or ''
      logging.info('Loading word embedding from :{} and finetune=:{}'.format(
          FLAGS.word_embedding_file, FLAGS.finetune_word_embedding))
      self.emb = melt.load_constant_cpu(
          FLAGS.word_embedding_file, name='emb', trainable=FLAGS.finetune_word_embedding)

    if FLAGS.position_embedding:
      logging.info('Using position embedding')
      self.pos_emb = embedding.get_embedding_cpu(name='pos_emb', height=TEXT_MAX_WORDS)
    else:
      self.pos_emb = None

    melt.visualize_embedding(self.emb, FLAGS.vocab)
    if is_training and FLAGS.monitor_level > 0:
      melt.monitor_embedding(self.emb, vocabulary.vocab, vocab_size)

    self.activation = melt.activations[FLAGS.activation]

    # TODO can consider global initiallizer like
    # with tf.variable_scope("Model", reuse=None, initializer=initializer)
    # https://github.com/tensorflow/models/blob/master/tutorials/rnn/ptb/ptb_word_lm.py
    self.weights_initializer = tf.random_uniform_initializer(
        -FLAGS.initializer_scale, FLAGS.initializer_scale)
    self.biases_initialzier = melt.slim.init_ops.zeros_initializer if FLAGS.bias else None

    self.mlp_dims = [int(x) for x in FLAGS.mlp_dims.split(',')] if FLAGS.mlp_dims is not '0' else None
    
    self.scope = 'dual_textsim'

    self.build_train_graph = self.build_graph

  def mlp_layers(self, text_feature):
    dims = self.mlp_dims
    if not dims:
      return text_feature

    return melt.slim.mlp(text_feature,
                         dims,
                         self.activation,
                         weights_initializer=self.weights_initializer,
                         biases_initializer=self.biases_initialzier,
                         scope='text_mlp')

  def encode(self, text):
    if self.pos_emb is None:
      text_feature = self.encoder.encode(text, self.emb)
    else:
      text_feature = self.encoder.encode(text, self.emb, self.pos_emb)
    #rnn will return encode_feature, state
    if isinstance(text_feature, tuple):
      text_feature = text_feature[0]
    return text_feature

  def lforward(self, text):
    """
    Args:
    text: batch text [batch_size, max_text_len]
    """
    text_feature = self.encode(text)
    text_feature = self.mlp_layers(text_feature)
    ##--well if not normalize will get big values.. then sigmod like 72 -> 1
    #if not FLAGS.loss == 'cross':
    ## contrastive loss work both norm or not norm, for simplicity here not norm
    ##https://www.quora.com/When-training-siamese-networks-how-does-one-determine-the-margin-for-contrastive-loss-How-do-you-convert-this-loss-to-accuracy
    ##You can just normalize features using L2 before using Contrastive Loss. 
    ##Then the margin can be constant while training because the distance between features will be normalized. 
    #if not FLAGS.loss == 'contrastive':  
    text_feature = normalize(text_feature)
    return text_feature

  def rforward(self, text):
    """
    Args:
    text: batch text [batch_size, max_text_len]
    """
    text_feature = self.encode(text)
    text_feature = normalize(text_feature)
    return text_feature

  #not work since ltext and rtext can have different length..
  def forword(self, ltext, rtext):
    assert not FLAGS.rtext_bow
    text = tf.concat([ltext, rtext], 0)
    text_feature = self.encode(text)
    ltext_feature, rtext_feature = tf.split(text_feature, 2, 0)
    ltext_feature = self.mlp_layers(ltext_feature)
    ltext_feature = normalize(ltext_feature)
    rtext_feature = normalize(rtext_feature)
    return ltext_feature, rtext_feature
 
  def build_graph(self, ltext, rtext, neg_ltext, neg_rtext):
    assert (neg_ltext is not None) or (neg_rtext is not None)
    with tf.variable_scope(self.scope) as scope:
      ltext_feature = self.lforward(ltext)
      #scope.reuse_variables() #rfword share same rnn or cnn..
      rtext_feature = self.rforward(rtext)
      pos_score = compute_sim(ltext_feature, rtext_feature)

      scope.reuse_variables()

      neg_scores_list = []
      if neg_rtext is not None:
        num_negs = neg_rtext.get_shape()[1]
        for i in xrange(num_negs):
          neg_rtext_feature_i = self.rforward(neg_rtext[:, i, :])
          neg_scores_i = compute_sim(ltext_feature, neg_rtext_feature_i)
          neg_scores_list.append(neg_scores_i)
      if neg_ltext is not None:
        num_negs = neg_ltext.get_shape()[1]
        for i in xrange(num_negs):
          neg_ltext_feature_i = self.lforward(neg_ltext[:, i, :])
          neg_scores_i = compute_sim(neg_ltext_feature_i, rtext_feature)
          neg_scores_list.append(neg_scores_i)
    
      #[batch_size, num_negs]
      neg_scores = tf.concat(neg_scores_list, 1)
      #---------------rank loss
      #[batch_size, 1 + num_negs]
      scores = tf.concat([pos_score, neg_scores], 1)
      tf.add_to_collection('scores', scores)

      loss = pairwise_loss(pos_score, neg_scores)
    return loss
    
class DualTextsimPredictor(DualTextsim, melt.PredictorBase):
  def __init__(self, encoder_type='bow'):
    melt.PredictorBase.__init__(self)
    DualTextsim.__init__(self, encoder_type=encoder_type, is_training=False, is_predict=True)

    self.ltext_feed = None
    self.rtext_feed = None

    self.text = None

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
      tf.get_variable_scope().reuse_variables()
      if not FLAGS.elementwise_predict:
        nearby = self.build_nearby_graph(ltext_feed, rtext_feed)
        lsim_nearby = self.build_lsim_nearby_graph(ltext_feed, rtext_feed)
        rsim_nearby = self.build_rsim_nearby_graph(ltext_feed, rtext_feed)
      lscore = self.build_lsim_graph(ltext_feed, rtext_feed)
      rscore = self.build_rsim_graph(ltext_feed, rtext_feed)
      emb_score = self.build_embsim_graph(ltext_feed, rtext_feed)
      lwords_importance = self.build_lwords_importance_graph(ltext_feed)
      rwords_importance = self.build_rwords_importance_graph(rtext_feed)
      
      tf.add_to_collection('score', score)
      if not FLAGS.elementwise_predict:
        tf.add_to_collection('nearby_values', nearby.values)
        tf.add_to_collection('nearby_indices', nearby.indices)
        tf.add_to_collection('lsim_nearby_values', lsim_nearby.values)
        tf.add_to_collection('lsim_nearby_indices', lsim_nearby.indices)
        tf.add_to_collection('rsim_nearby_values', rsim_nearby.values)
        tf.add_to_collection('rsim_nearby_indices', rsim_nearby.indices)
      tf.add_to_collection('lscore', lscore)
      tf.add_to_collection('rscore', rscore)
      tf.add_to_collection('emb_score', emb_score)
      tf.add_to_collection('lwords_importance', lwords_importance)
      tf.add_to_collection('rwords_importance', rwords_importance)
    return score

  def predict(self, ltexts, rtexts):
    feed_dict = {
      self.ltext_feed: ltexts,
      self.rtext_feed: rtexts
      }
    
    score = self.sess.run(self.score, feed_dict)
    if FLAGS.elementwise_predict and self.is_predict:
      ##--TODO
      #score = self.sess.run(self.elemetwise_score, feed_dict)
      score = score.reshape((len(rtexts),))

    return score

  def elementwise_bulk_predict(self, ltexts, rtexts):
    scores = []
    if len(rtexts) >= len(ltexts):
      for ltext in ltexts:
        stacked_ltexts = np.array([ltext] * len(rtexts))
        score = self.predict(stacked_ltexts, rtexts)
        scores.append(score)
    else:
      for rtext in rtexts:
        stacked_rtexts = np.array([rtext] * len(ltexts))
        score = self.predict(ltexts, stacked_rtexts)
        scores.append(score)
    return np.array(scores)  

  def bulk_predict(self, ltexts, rtexts):
    """
    input images features [m, ] , texts feature [n,] will
    outut [m, n], for each image output n scores for each text  
    """
    if FLAGS.elementwise_predict:
      return self.elementwise_bulk_predict(ltexts, rtexts)
    return self.predict(ltexts, rtexts)
    
  
  #TODO may be speedup by cocant [ltex, rtext] as one batch and co forward then split 
  def build_graph(self, ltext, rtext):
    with tf.variable_scope(self.scope):
      ltext_feature = self.lforward(ltext)
      #make to cpu ? for mem issue of cnn? if not perf hurt much?
      #reidctor.bulk_predict duration: 125.557517052
      #cpu is slow evaluate_scores duration: 135.078355074
      #if self.encoder_type != 'cnn':
      rtext_feature = self.rforward(rtext)
      #else:
      #with tf.device('/cpu:0'):
      #rtext_feature = self.rforward(rtext)

      score = dot(ltext_feature, rtext_feature)
      
      #if FLAGS.elementwise_predict and self.is_predict:
      #  self.elemetwise_score = melt.element_wise_dot(ltext_feature, rtext_feature)
      return score

  def build_lsim_graph(self, ltext, rtext):
    with tf.variable_scope(self.scope):
      ltext_feature = self.lforward(ltext)
      rtext_feature = self.lforward(rtext)
      score = dot(ltext_feature, rtext_feature)
      return score

  def build_rsim_graph(self, ltext, rtext):
    with tf.variable_scope(self.scope):
      ltext_feature = self.rforward(ltext)
      rtext_feature = self.rforward(rtext)
      score = dot(ltext_feature, rtext_feature)
      return score

  def build_embsim_graph(self, ltext, rtext):
    """
    for bow is emb sim
    for rnn is sim after rnn encoding
    """
    #TODO imporve speed by concat pos and all negs to one batch
    with tf.variable_scope(self.scope):
      ltext_feature = self.encode(ltext)
      rtext_feature = self.encode(rtext)
      score = melt.cosine(ltext_feature, rtext_feature)
      return score

  def build_nearby_graph(self, ltext, rtext, topn=50, sorted=True):
  	score = self.build_graph(ltext, rtext)
  	return tf.nn.top_k(score, topn, sorted=sorted)

  def build_lsim_nearby_graph(self, ltext, rtext, topn=50, sorted=True):
  	score = self.build_lsim_graph(ltext, rtext)
  	return tf.nn.top_k(score, topn, sorted=sorted)

  def build_rsim_nearby_graph(self, ltext, rtext, topn=50, sorted=True):
  	score = self.build_rsim_graph(ltext, rtext)
  	return tf.nn.top_k(score, topn, sorted=sorted)

  def _build_words_importance_graph(self, text, forward_fn):
    with tf.variable_scope(self.scope):
      text2 = text

      # TODO hack here for rnn with start pad <S> as 2!
      if FLAGS.encode_start_mark:
        start_pad = tf.zeros([1, 1], dtype=text.dtype) + 2
        text2 = tf.concat([start_pad, text], 1)
      if FLAGS.encode_end_mark:
        end_pad = tf.zeros([1, 1], dtype=text.dtype) + 1
        text2 = tf.concat([text2, end_pad], 1)

      # text batch_size must be 1! currently [1, seq_len] -> [seq_len, 1]
      words = tf.transpose(text2, [1, 0])
      #[seq_len, 1] -> [seq_len, emb_dim]
      word_feature = forward_fn(words)

      #[batch_size, seq_len] -> [batch_size, emb_dim]  [1, emb_dim]
      text_feature = forward_fn(text)
      
      #[1, seq_len]
      score = dot(text_feature, word_feature)
      return score

  def build_lwords_importance_graph(self, text):
    return self._build_words_importance_graph(text, self.lforward)

  def build_rwords_importance_graph(self, text):
    return self._build_words_importance_graph(text, self.rforward)

  #-----------for evaluate
  def init_evaluate_constant_text(self, text_npy):
    #self.text = tf.constant(text_npy)
    if self.text is None:
      #if self.encoder_type == 'cnn':
      #  import numpy as np
      #  #text_npy = np.load(text_npy)
      #  text_npy = text_npy[:5000]
      #  #self.text = melt.load_constant_cpu(text_npy, self.sess)
      ##else:
      self.text = melt.load_constant(text_npy, self.sess)

  def build_evaluate_fixed_text_graph(self, image_feature, step=None): 
    """
    text features directly load to graph,
    used in evaluate.py for both fixed text and fixed words
    """
    if not step:
      score = self.build_graph(image_feature, self.text)
    else:
      num_texts = self.text.get_shape()[0]
      start = 0
      scores = []
      while start < num_texts:
        end = start + step 
        scores.append(self.build_graph(image_feature, self.text[start:end, :]))
        start = end 
      score = tf.concat(scores, 1)
    return score

  def build_evaluate_image_word_graph(self, image_feature):
    with tf.variable_scope(self.scope):
      image_feature = self.lforward(image_feature)
      #no need for embedding lookup
      word_feature = self.forward_word_feature()
      score = dot(image_feature, word_feature)
      return score

  def forward_word_feature(self):
    #@TODO may need melt.first_nrows so as to avoid caclc to many words
    # du -h comment_feature_final.npy 
    #3.6G comment_feature_final.npy  so 100w 4G, 800w 32G, 1500w word will exceed cpu 
    MAX_EMB_WORDS = 20000
    if self.encoder_type == 'cnn':
      MAX_EMB_WORDS = 500
    num_words = min(self.vocab_size - 1, MAX_EMB_WORDS)
    word_index = tf.reshape(tf.range(num_words), [num_words, 1])
    #for cnn might be need to conv so 1 might be less then window size
    if self.encoder_type == 'cnn':
      word_index = tf.concat([word_index, tf.zeros([num_words, TEXT_MAX_WORDS - 1], tf.int32)], 1)
    word_feature = self.rforward(word_index)
    return word_feature