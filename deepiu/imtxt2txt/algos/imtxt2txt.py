#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   show_and_tell.py
#        \author   chenghuige  
#          \date   2016-09-04 17:49:20.030172
#   \Description  
# ==============================================================================
  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

import functools

import melt
logging = melt.logging
from deepiu.imtxt2txt import conf 
from deepiu.imtxt2txt.conf import IMAGE_FEATURE_LEN, TEXT_MAX_WORDS, NUM_RESERVED_IDS, INPUT_TEXT_MAX_WORDS
from deepiu.util import vocabulary
from deepiu.seq2seq import embedding
import deepiu


from deepiu.seq2seq.rnn_decoder import SeqDecodeMethod
from deepiu import seq2seq

from deepiu.util import vocabulary 
from deepiu.util.text2ids import idslist2texts

  
class Imtxt2txt(object):
  """
  ShowAndTell class is a trainer class
  but has is_training mark for ShowAndTell predictor will share some code here
  3 modes
  train,
  evaluate,
  predict
  """
  def __init__(self, is_training=True, is_predict=False):
    super(Imtxt2txt, self).__init__()

    assert FLAGS.add_text_start is True

    self.is_training = is_training 
    self.is_predict = is_predict

    #if is_training:
    logging.info('num_sampled:{}'.format(FLAGS.num_sampled))
    logging.info('use_neg:{}'.format(FLAGS.use_neg))
    logging.info('num_sampled:{}'.format(FLAGS.num_sampled))
    logging.info('log_uniform_sample:{}'.format(FLAGS.log_uniform_sample))
    logging.info('num_layers:{}'.format(FLAGS.num_layers))
    logging.info('keep_prob:{}'.format(FLAGS.keep_prob))
    logging.info('emb_dim:{}'.format(FLAGS.emb_dim))
    logging.info('add_text_start:{}'.format(FLAGS.add_text_start))
    logging.info('zero_as_text_start:{}'.format(FLAGS.zero_as_text_start))

    emb = embedding.get_embedding('emb')
    if is_training and FLAGS.monitor_level > 0:
      melt.monitor_embedding(emb, vocabulary.vocab, vocabulary.vocab_size)

    self.encoder = seq2seq.rnn_encoder.RnnEncoder(is_training, is_predict)
    self.encoder.set_embedding(emb)

    #emb2 = embedding.get_embedding('emb2')
    self.decoder = seq2seq.rnn_decoder.RnnDecoder(is_training, is_predict)
    self.decoder.set_embedding(emb)
    
    print('start_id', self.decoder.start_id)

    self.emb_dim = FLAGS.emb_dim
    self.initializer = tf.random_uniform_initializer(
        minval=-FLAGS.initializer_scale,
        maxval=FLAGS.initializer_scale)

    if not FLAGS.pre_calc_image_feature:
      assert melt.apps.image_processing.image_processing_fn is not None, 'forget melt.apps.image_processing.init()'
      self.image_process_fn = functools.partial(melt.apps.image_processing.image_processing_fn,
                                                height=FLAGS.image_height, 
                                                width=FLAGS.image_width)


  def build_image_embeddings(self, image_feature):
    """
    Builds the image model subgraph and generates image embeddings.
    """
    if not FLAGS.pre_calc_image_feature:
      image_feature = self.image_process_fn(image_feature)
    with tf.variable_scope("image_embedding") as scope:
      image_embeddings = tf.contrib.layers.fully_connected(
          inputs=image_feature,
          num_outputs=self.emb_dim,
          activation_fn=None,
          weights_initializer=self.initializer,
          biases_initializer=None,
          scope=scope)
    return image_embeddings

  def build_graph(self, image_feature, input_text, text, 
                  exact_prob=False, exact_loss=False):
    """
    exact_prob and exact_loss actually do the same thing,
    they only be used on when is_predict is true
    """
    assert not (exact_prob and exact_loss)
    assert not ((not self.is_predict) and (exact_prob or exact_loss))

    with tf.variable_scope("encode"):
      image_emb = self.build_image_embeddings(image_feature)
      encoder_output, state = self.encoder.encode(input_text, input=image_emb)
      #encoder_output, state = self.encoder.encode(input_text, input=None)
      if not FLAGS.use_attention:
        encoder_output = None
    with tf.variable_scope("decode"):
      loss = self.decoder.sequence_loss(text, 
                                        initial_state=state, 
                                        attention_states=encoder_output,
                                        exact_prob=exact_prob, 
                                        exact_loss=exact_loss)

      #this is used in train.py for evaluate eval_scores = tf.get_collection('scores')[-1]
      #because we want to show loss for each instance
      if not self.is_training and not self.is_predict:
        tf.add_to_collection('scores', loss)
    
    if not self.is_predict:
      loss = tf.reduce_mean(loss)

    return loss


  def build_train_graph(self, image_feature, input_text, text):
    return self.build_graph(image_feature, input_text, text)


class Imtxt2txtPredictor(Imtxt2txt, melt.PredictorBase):
  def __init__(self):
    melt.PredictorBase.__init__(self)
    Imtxt2txt.__init__(self, is_training=False, is_predict=True)

    self.input_text_feed = tf.placeholder(tf.int64, [None, INPUT_TEXT_MAX_WORDS], name='input_text')
    tf.add_to_collection('input_text_feed', self.input_text_feed)
    self.text_feed = tf.placeholder(tf.int64, [None, TEXT_MAX_WORDS], name='text')
    tf.add_to_collection('text_feed', self.text_feed)

    if FLAGS.pre_calc_image_feature:
      self.image_feature_feed = tf.placeholder(tf.float32, [None, IMAGE_FEATURE_LEN], name='image_feature')
    else:
      self.image_feature_feed =  tf.placeholder(tf.string, [None,], name='image_feature')

  def init_predict_text(self, decode_method=0, beam_size=5, convert_unk=True):
    """
    init for generate texts
    """
    text, score = self.build_predict_text_graph(self.image_feature_feed,
                                                self.input_text_feed, 
                                                decode_method=decode_method, 
                                                beam_size=beam_size, 
                                                convert_unk=convert_unk)
    return text, score

  def init_predict(self, exact_prob=False, exact_loss=False):
    score = self.build_predict_graph(self.image_feature_feed,
                                     self.input_text_feed, 
                                     self.text_feed, 
                                     exact_prob=exact_prob, 
                                     exact_loss=exact_loss)
    return score

 
  def build_predict_text_graph(self, image_feature, input_text, decode_method='greedy', beam_size=5, convert_unk=True):
    with tf.variable_scope("encode"):
      image_emb = self.build_image_embeddings(image_feature)
      encoder_output, state = self.encoder.encode(input_text, input=image_emb)
      #encoder_output, state = self.encoder.encode(input_text, input=None)
      if not FLAGS.use_attention:
        encoder_output = None
    with tf.variable_scope("decode"):
      batch_size = tf.shape(input_text)[0]
      decoder_input = self.decoder.get_start_embedding_input(batch_size)
      max_words = FLAGS.decode_max_words if FLAGS.decode_max_words else TEXT_MAX_WORDS
      if decode_method == SeqDecodeMethod.greedy:
        return self.decoder.generate_sequence_greedy(decoder_input, 
                                       max_words=max_words, 
                                       initial_state=state,
                                       attention_states=encoder_output,
                                       convert_unk=convert_unk)
      else:
        if decode_method == SeqDecodeMethod.ingraph_beam:
          decode_func = self.decoder.generate_sequence_ingraph_beam
        elif decode_method == SeqDecodeMethod.outgraph_beam:
          decode_func = self.decoder.generate_sequence_outgraph_beam
        else:
          raise ValueError('not supported decode_method: %s' % decode_method)
        
        input_text, input_text_length = melt.pad(input_text, end_id=self.encoder.end_id)
        return decode_func(decoder_input, 
                           max_words=max_words, 
                           initial_state=state,
                           attention_states=encoder_output,
                           beam_size=beam_size, 
                           convert_unk=convert_unk,
                           length_normalization_factor=FLAGS.length_normalization_factor,
                           input_text=input_text,
                           input_text_length=input_text_length)

  def build_predict_graph(self, image_feature, input_text, text, exact_prob=False, exact_loss=False):
    input_text = tf.reshape(input_text, [-1, INPUT_TEXT_MAX_WORDS])
    text = tf.reshape(text, [-1, TEXT_MAX_WORDS])
  
    #loss is -logprob  
    loss = self.build_graph(image_feature, input_text, text, 
                            exact_prob=exact_prob, 
                            exact_loss=exact_loss)
 
    score = -loss 
    if FLAGS.predict_use_prob:
      score = tf.exp(score)
    return score
