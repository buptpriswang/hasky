#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   show_and_tell_predictor.py
#        \author   chenghuige  
#          \date   2016-09-04 17:50:21.017234
#   \Description  
# ==============================================================================

  
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

from deepiu.image_caption.conf import IMAGE_FEATURE_LEN, TEXT_MAX_WORDS
from deepiu.util import vocabulary 
from deepiu.util.text2ids import idslist2texts
from deepiu.image_caption.algos.show_and_tell import ShowAndTell

from deepiu.seq2seq.rnn_decoder import SeqDecodeMethod

class ShowAndTellPredictor(ShowAndTell, melt.PredictorBase):
  def __init__(self):
    melt.PredictorBase.__init__(self)
    ShowAndTell.__init__(self, is_training=False, is_predict=True)

    if FLAGS.pre_calc_image_feature:
      self.image_feature_len = IMAGE_FEATURE_LEN 
      self.image_feature_feed = tf.placeholder(tf.float32, [None, self.image_feature_len], name='image_feature')
    else:
      self.image_feature_feed =  tf.placeholder(tf.string, [None,], name='image_feature')

    tf.add_to_collection('feed', self.image_feature_feed)
    tf.add_to_collection('lfeed', self.image_feature_feed)
    
    self.text_feed = tf.placeholder(tf.int64, [None, TEXT_MAX_WORDS], name='text')
    tf.add_to_collection('rfeed', self.text_feed)

    self.beam_text = None 
    self.beam_text_score = None

    self.image_model = None

  def init_predict_text(self, decode_method='greedy', beam_size=5, convert_unk=True):
    """
    init for generate texts
    """
    text, score = self.build_predict_text_graph(self.image_feature_feed, 
      decode_method, 
      beam_size, 
      convert_unk)
    self.beam_text = text 
    self.beam_text_score = score
    return text, score

  def init_predict(self, exact_loss=False):
    self.score = self.build_predict_graph(self.image_feature_feed, 
                                          self.text_feed, 
                                          exact_loss=exact_loss)
    tf.add_to_collection('score', self.score)
    return self.score
 
  #TODO Notice when training this image always be float...  
  def build_predict_text_graph(self, image, decode_method='greedy', beam_size=5, convert_unk=True):
    image_emb = self.build_image_embeddings(image)

    attention_states = None 
    if FLAGS.show_atten_tell:
      image_emb, attention_states = self.init_attention(image_emb)

    state = None
    if FLAGS.image_as_init_state:
      #for im2txt one more step at first
      with tf.variable_scope(self.decoder.scope) as scope:
        batch_size=melt.get_batch_size(image_emb)
        zero_state = self.decoder.cell.zero_state(batch_size, dtype=tf.float32)
        _, state = self.decoder.cell(image_emb, zero_state)
        image_emb= self.decoder.get_start_embedding_input(batch_size)
  
    max_words = TEXT_MAX_WORDS
    if decode_method == SeqDecodeMethod.greedy:
      return self.decoder.generate_sequence_greedy(image_emb, 
                                     max_words=max_words, 
                                     initial_state=state,
                                     attention_states=attention_states,
                                     convert_unk=convert_unk)
    else:
      if decode_method == SeqDecodeMethod.ingraph_beam:
        decode_func = self.decoder.generate_sequence_ingraph_beam
      elif decode_method == SeqDecodeMethod.outgraph_beam:
        decode_func = self.decoder.generate_sequence_outgraph_beam
      else:
        raise ValueError('not supported decode_method: %s' % decode_method)
      
      return decode_func(image_emb, 
                         max_words=max_words, 
                         initial_state=state,
                         beam_size=beam_size, 
                         convert_unk=convert_unk,
                         attention_states=attention_states,
                         length_normalization_factor=FLAGS.length_normalization_factor)

  def build_predict_graph(self, image, text, exact_loss=False):
    text = tf.reshape(text, [-1, TEXT_MAX_WORDS])
    
    loss = self.build_graph(image, text)
    score = -loss 
    if FLAGS.predict_use_prob:
      score = tf.exp(score)
    return score

  #--------------below depreciated, just use melt.predictor for inference
  def predict(self, image, text):
    """
    default usage is one single image , single text predict one sim score
    """
    #hack for big feature problem, input is reading raw image...
    if FLAGS.pre_calc_image_feature and isinstance(images[0], (str, np.string_)):
      if self.image_model is None:
        #notice must not pass self.sess! will reload fail FIXME TODO
        self.image_model = melt.ImageModel(FLAGS.image_checkpoint_file, FLAGS.image_model_name, feature_name=FLAGS.image_endpoint_feature_name)
      images = self.image_model.gen_feature(images)

    feed_dict = {
      self.image_feature_feed: image,
      self.text_feed: text,
    }
    score = self.sess.run(self.score, feed_dict)
    return score

  def predict_text(self, images):
    """
    for translation evaluation only
    """
    #hack for big feature problem, input is reading raw image...
    if FLAGS.pre_calc_image_feature and isinstance(images[0], (str, np.string_)):
      if self.image_model is None:
        self.image_model = melt.ImageModel(FLAGS.image_checkpoint_file, FLAGS.image_model_name, feature_name=FLAGS.image_endpoint_feature_name)
      images = self.image_model.gen_feature(images)

    if self.beam_text is None:
      self.init_predict_text(decode_method=SeqDecodeMethod.ingraph_beam)

    feed_dict = {
      self.image_feature_feed: images,
      }

    texts, scores = self.sess.run([self.beam_text, self.beam_text_score], feed_dict=feed_dict)

    return texts, scores
