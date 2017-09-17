#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   discriminant_self.py
#        \author   chenghuige  
#          \date   2016-09-22 22:39:41.260080
#   \Description  
# ==============================================================================

"""
TODO
place shold be feed
"""
  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS

import functools
import numpy as np

import gezi
import melt

logging = melt.logging

from deepiu.util.rank_loss import dot, compute_sim, normalize

#@TODO should not use conf... change to config file parse instead  @FIXME!! this will cause problem in deepiu package
try:
  import conf
  from conf import IMAGE_FEATURE_LEN,TEXT_MAX_WORDS, MAX_EMB_WORDS
except Exception:
  from deepiu.image_caption.conf import IMAGE_FEATURE_LEN,TEXT_MAX_WORDS, MAX_EMB_WORDS

from deepiu.image_caption.algos.discriminant_trainer import DiscriminantTrainer

class DiscriminantPredictor(DiscriminantTrainer, melt.PredictorBase):
  def __init__(self, encoder_type='bow', sess=None):
    #super(DiscriminantPredictor, self).__init__()
    melt.PredictorBase.__init__(self, sess=sess)
    DiscriminantTrainer.__init__(self, encoder_type=encoder_type, is_training=False, is_predict=True)

    self.text = None
    self.text_feed = None
    self.text2_feed = None

    self.image_feature = None
    self.image_feature_feed = None

    #input image feature, text ids
    self.score = None
    self.textsim_score = None
    #input image feature, assume text ids already load
    self.fixed_text_score = None
    #input image feature, assume text final feature already load
    self.fixed_text_feature_score = None

    self.image_feature_len = IMAGE_FEATURE_LEN 

    self.image_model = None


  def init_predict(self, text_max_words=TEXT_MAX_WORDS):
    self.score = self.build_predict_graph(text_max_words)
    tf.get_variable_scope().reuse_variables()
    self.textsim_score = self.build_textsim_predict_graph(text_max_words)
    self.text_emb_sim_score = self.build_text_emb_sim_predict_graph(text_max_words)

    self.words_importance = self.build_words_importance_graph(self.get_text_feed(text_max_words))
    try:
      self.encoder_words_importance = self.encoder_words_importance(self.get_text_feed(text_max_words), self.emb)
    except Exception:
      logging.info('predictor not support encoder_words_importance')
      self.encoder_words_importance = None

    self.image_encode, self.text_encode = self.build_encode_graph(text_max_words)

    self.image_words_score = self.build_image_words_sim_graph()
    self.text_words_score = self.build_text_words_sim_graph(text_max_words)
    self.text_words_emb_score = self.build_text_words_emb_sim_graph(text_max_words)

    tf.add_to_collection('score', self.score)
    tf.add_to_collection('textsim', self.textsim_score)
    tf.add_to_collection('text_emb_sim', self.text_emb_sim_score)
    tf.add_to_collection('text_encode', self.text_encode)
    tf.add_to_collection('image_encode', self.image_encode)
    tf.add_to_collection('words_importance', self.words_importance)
    if self.encoder_words_importance is not None:
      tf.add_to_collection('encoder_words_importance', self.encoder_words_importance)

    tf.add_to_collection('image_words_score', self.image_words_score)
    tf.add_to_collection('text_words_score', self.text_words_score)
    tf.add_to_collection('text_words_emb_score', self.text_words_emb_score)

    return self.score

  def predict(self, image, text):
    #hack for big feature problem, input is reading raw image...
    if FLAGS.pre_calc_image_feature and isinstance(image[0], (str, np.string_)):
      if self.image_model is None:
        self.image_model = melt.ImageModel(FLAGS.image_checkpoint_file, FLAGS.image_model_name, feature_name=FLAGS.image_endpoint_feature_name)
      image = self.image_model.gen_feature(image)  
  
    feed_dict = {
      #self.image_feature_feed: image.reshape([1, self.image_feature_len]),
      self.image_feature_feed: image,
      self.text_feed: text
      }
    score = self.sess.run(self.score, feed_dict)
    return score

  def predict_textsim(self, text, text2):
    feed_dict = {
      self.text_feed: text,
      self.text2_feed: text2,
      }
    score = self.sess.run(self.textsim_score, feed_dict)
    return score
  
  def get_image_feature_feed(self):
    if self.image_feature_feed is None:
      if FLAGS.pre_calc_image_feature:
        self.image_feature_feed = tf.placeholder(tf.float32, [None, self.image_feature_len], name='image_feature')
      else:
        self.image_feature_feed =  tf.placeholder(tf.string, [None,], name='image_feature')
      tf.add_to_collection('feed', self.image_feature_feed)
      tf.add_to_collection('lfeed', self.image_feature_feed)
    return self.image_feature_feed

  def get_text_feed(self, text_max_words=TEXT_MAX_WORDS):
    if self.text_feed is None:
      self.text_feed = tf.placeholder(tf.int32, [None, text_max_words], name='text')
      tf.add_to_collection('rfeed', self.text_feed)
    return self.text_feed

  def get_text2_feed(self, text_max_words=TEXT_MAX_WORDS):
    if self.text2_feed is None:
      self.text2_feed = tf.placeholder(tf.int32, [None, text_max_words], name='text2')
      tf.add_to_collection('rfeed2', self.text2_feed)
    return self.text2_feed

  def fixed_text_predict(self, images):
    """
    not useful compare to bulk_predict since performance is similar, no gain
    """
    if self.fixed_text_score is None:
      assert self.text is not None, 'call init_evaluate_constant at first'
      self.fixed_text_score = self.build_evaluate_fixed_text_graph(self.get_image_feature_feed())
    return self.sess.run(self.fixed_text_score, feed_dict={self.image_feature_feed: images})

  def fixed_text_feature_predict(self, images):
    """
    this will be fast
    """
    assert self.fixed_text_feature_score is not None, 'call build_fixed_text_feature_graph(self, text_feature_npy) at first'
    return self.sess.run(self.fixed_text_feature_score, feed_dict={self.image_feature_feed: images})

  def build_graph(self, image_feature, text):
    with tf.variable_scope(self.scope):
      image_feature = self.forward_image_feature(image_feature)
      text_feature = self.forward_text(text)
      score = dot(image_feature, text_feature)
      return score

  # TODO may consider better performance, reomve redudant like sim(a,b) not need to calc sim(b,a)
  def build_textsim_graph(self, text,  text2):
    with tf.variable_scope(self.scope):
      text_feature = self.forward_text(text)
      text_feature2 = self.forward_text(text2)
      score = dot(text_feature, text_feature2)
      return score

  def build_text_emb_sim_graph(self, text,  text2):
    """
    for bow is emb sim
    for rnn is sim after rnn encoding
    """
    with tf.variable_scope(self.scope):
      text_feature = self.gen_text_feature(text, self.emb)
      text_feature2 = self.gen_text_feature(text2, self.emb)
      score = melt.cosine(text_feature, text_feature2)
      return score

  def build_predict_graph(self, text_max_words=TEXT_MAX_WORDS):
    score = self.build_graph(self.get_image_feature_feed(), self.get_text_feed(text_max_words))
    return score

  def build_textsim_predict_graph(self, text_max_words=TEXT_MAX_WORDS):
    score = self.build_textsim_graph(self.get_text_feed(text_max_words), self.get_text2_feed(text_max_words))
    return score

  def build_text_emb_sim_predict_graph(self, text_max_words=TEXT_MAX_WORDS):
    score = self.build_text_emb_sim_graph(self.get_text_feed(text_max_words), self.get_text2_feed(text_max_words))
    return score

  def build_fixed_text_feature_graph(self, text_feature_npy): 
    """
    text features directly load to graph, @NOTICE text_feature_npy all vector must of same length
    used in evaluate.py for both fixed text and fixed words
    @FIXME dump text feature should change api
    """
    with tf.variable_scope(self.scope):
      image_feature = self.forward_image_feature(self.image_feature_feed)
      text_feature = melt.load_constant(text_feature_npy, self.sess)
      score = dot(image_feature, text_feature)
      return score

  #def build_fixed_text_graph(self, text_npy): 
  #  self.init_evaluate_constant(text_npy)
  #  score = self.build_evaluate_fixed_image_text_graph(self.get_image_feature_feed())
  #  return score

  def build_fixed_text_graph(self, text_npy): 
    return self.build_fixed_text_feature_graph(text_npy)
    
  #--------- only used during training evaluaion, image_feature and text all small
  def build_train_graph(self, image_feature, text, neg_text, lookup_negs_once=False):
    """
    Only used for train and evaluation, hack!
    """
    return super(DiscriminantPredictor, self).build_graph(image_feature, text, neg_text, lookup_negs_once)
  
  def init_evaluate_constant_image(self, image_feature_npy):
    if self.image_feature is None:
      self.image_feature = tf.constant(image_feature_npy)

  def init_evaluate_constant_text(self, text_npy):
    #self.text = tf.constant(text_npy)
    if self.text is None:
      self.text = melt.load_constant(text_npy, self.sess)

  def init_evaluate_constant(self, image_feature_npy, text_npy):
    self.init_evaluate_constant_image(image_feature_npy)
    self.init_evaluate_constant_text(text_npy)

  def build_evaluate_fixed_image_text_graph(self):
    """
    image features and texts directly load to graph
    """
    score = self.build_graph(self.image_feature, self.text)
    return score

  def forward_word_feature(self):
    #@TODO may need melt.first_nrows so as to avoid caclc to many words
    # du -h comment_feature_final.npy 
    #3.6G	comment_feature_final.npy  so 100w 4G, 800w 32G, 1500w word will exceed cpu 
    num_words = min(self.vocab_size - 1, MAX_EMB_WORDS)
    word_index = tf.reshape(tf.range(num_words), [num_words, 1])
    word_feature = self.forward_text(word_index)
    return word_feature

  def gen_word_feature(self):
    #@TODO may need melt.first_nrows so as to avoid caclc to many words
    # du -h comment_feature_final.npy 
    #3.6G comment_feature_final.npy  so 100w 4G, 800w 32G, 1500w word will exceed cpu 
    num_words = min(self.vocab_size - 1, MAX_EMB_WORDS)
    word_index = tf.reshape(tf.range(num_words), [num_words, 1])
    word_feature = self.gen_text_feature(word_index, self.emb)
    return word_feature

  def build_evaluate_image_word_graph(self, image_feature):
    with tf.variable_scope(self.scope):
      image_feature = self.forward_image_feature(image_feature)
      #no need for embedding lookup
      word_feature = self.forward_word_feature()
      score = dot(image_feature, word_feature)
      return score

  def build_evluate_fixed_image_word_graph(self):
    score = self.build_evaluate_image_word_graph(self.image_feature)
    return score

  def build_evaluate_fixed_text_graph(self, image_feature, step=None): 
    """
    text features directly load to graph,
    used in evaluate.py for both fixed text and fixed words
    """
    #score = self.build_graph(image_feature, self.text)
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

  #@TODO for evaluate random data, choose image_feature, and use all text to calc score, and show ori_text, predicted text

  #----------for offline dump usage
  def forward_allword(self):
    """
    only consider single word, using it's embedding as represention
    """
    with tf.variable_scope(self.scope):
      return self.forward_word_feature()

  def forward_fixed_text(self, text_npy):
    #text = tf.constant(text_npy)  #smaller than 2G, then ok... 
    #but for safe in more application
    text = melt.load_constant(text_npy, self.sess)
    with tf.variable_scope(self.scope):
      text_feature = self.forward_text(text)
      return text_feature

  def build_words_importance_graph(self, text):
    with tf.variable_scope(self.scope):
      text2 = text

      #TODO hack here for rnn with start pad <S> as 2!
      if FLAGS.encode_start_mark:
        start_pad = tf.zeros([1, 1], dtype=text.dtype) + 2
        text2 = tf.concat([start_pad, text], 1)
      if FLAGS.encode_end_mark:
        end_pad = tf.zeros([1, 1], dtype=text.dtype) + 1
        text2 = tf.concat([text2, end_pad], 1)

      #text batch_size must be 1! currently [1, seq_len] -> [seq_len, 1]
      words = tf.transpose(text2, [1, 0])
      #[seq_len, 1] -> [seq_len, emb_dim]
      word_feature = self.forward_text(words)

      #[batch_size, seq_len] -> [batch_size, emb_dim]  [1, emb_dim]
      text_feature = self.forward_text(text)
      
      #[1, seq_len]
      score = dot(text_feature, word_feature)
      return score
  
  def build_encode_graph(self, text_max_words=TEXT_MAX_WORDS):
    with tf.variable_scope(self.scope):
      image_encode = self.forward_image_feature(self.get_image_feature_feed())
      text_encode = self.forward_text(self.get_text_feed(text_max_words))
    return image_encode, text_encode

  def build_image_words_sim_graph(self):
    with tf.variable_scope(self.scope):
      image_feature = self.forward_image_feature(self.get_image_feature_feed())
      word_feature = self.forward_word_feature()
      score = dot(image_feature, word_feature)
      return score

  def build_text_words_sim_graph(self, text_max_words=TEXT_MAX_WORDS):
    with tf.variable_scope(self.scope):
      text_feature = self.forward_text(self.get_text_feed(text_max_words))
      word_feature = self.forward_word_feature()
      score = dot(text_feature, word_feature)
      return score

  def build_text_words_emb_sim_graph(self, text_max_words=TEXT_MAX_WORDS):
    with tf.variable_scope(self.scope):
      text_feature = self.gen_text_feature(self.get_text_feed(text_max_words), self.emb)
      word_feature = self.gen_word_feature()
      score = melt.cosine(text_feature, word_feature)
      return score