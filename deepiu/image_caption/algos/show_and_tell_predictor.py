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
    #super(ShowAndTellPredictor, self).__init__()
    melt.PredictorBase.__init__(self)
    ShowAndTell.__init__(self, is_training=False, is_predict=True)

    if FLAGS.pre_calc_image_feature:
      self.image_feature_len = IMAGE_FEATURE_LEN 
      self.image_feature_feed = tf.placeholder(tf.float32, [None, self.image_feature_len], name='image_feature')
    else:
      self.image_feature_len = 2048
      self.image_feature_feed =  tf.placeholder(tf.string, [None,], name='image_feature')
    
    self.text_feed = tf.placeholder(tf.int64, [None, TEXT_MAX_WORDS], name='text')

  def init(self, reuse=True):
    #self.image_process_fn = functools.partial(melt.image.image2feature_fn,
    self.image_process_fn = functools.partial(melt.image.create_image2feature_fn(), 
                                              height=FLAGS.image_height, 
                                              width=FLAGS.image_width,
                                              reuse=reuse)

  def init_predict_text(self, decode_method=0, beam_size=5, convert_unk=True):
    """
    init for generate texts
    """
    text = self.build_predict_text_graph(self.image_feature_feed, 
      decode_method, 
      beam_size, 
      convert_unk)

    return text

  def init_predict(self, exact_loss=False):
    self.score = self.build_predict_graph(self.image_feature_feed, 
                                          self.text_feed, 
                                          exact_loss=exact_loss)
    return self.score
 

  #TODO Notice when training this image always be float...  
  def build_predict_text_graph(self, image, decode_method=0, beam_size=5, convert_unk=True):
    decoder_input = self.build_image_embeddings(image)
    state = None
    
    if decode_method == SeqDecodeMethod.greedy:
      return self.decoder.generate_sequence_greedy(decoder_input, 
                                            max_words=TEXT_MAX_WORDS, 
                                            initial_state=state, 
                                            convert_unk=convert_unk)
    elif decode_method == SeqDecodeMethod.beam:
      return self.decoder.generate_sequence_beam(decoder_input,
                                                 max_words=TEXT_MAX_WORDS, 
                                                 initial_state=state, 
                                                 beam_size=beam_size, 
                                                 convert_unk=convert_unk,
                                                 length_normalization_factor=0.)
    else:
      raise ValueError('not supported decode_method: %d' % decode_method)

  def build_predict_graph(self, image, text, exact_loss=False):
    #image = tf.reshape(image, [-1, self.image_feature_len])
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
    feed_dict = {
      #self.image_feature_feed: image.reshape([-1, self.image_feature_len]),
      self.image_feature_feed: image,
      self.text_feed: text.reshape([-1, TEXT_MAX_WORDS]),
    }
    score = self.sess.run(self.score, feed_dict)
    score = score.reshape((len(text),))
    return score

  def bulk_predict(self, images, texts):
    """
    input multiple images, multiple texts
    outupt: 
    
    image0, text0_score, text1_score ...
    image1, text0_score, text1_score ...
    ...

    """
    scores = []
    for image in images:
      stacked_images = np.array([image] * len(texts))
      score = self.predict(stacked_images, texts)
      scores.append(score)
    return np.array(scores)

  def predict_text(self, images, index=0):
    """
    depreciated will remove
    """
    feed_dict = {
      self.image_feature_feed: images,
      }

    vocab = vocabulary.get_vocab()

    generated_words = self.sess.run(self.text_list[index], feed_dict) 
    texts = idslist2texts(generated_words)

    return texts
