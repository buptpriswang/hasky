#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   show_and_tell.py
#        \author   chenghuige  
#          \date   2016-09-04 17:49:20.030172
#   \Description  
# ==============================================================================
"""
lstm based generative model

@TODO try to use seq2seq.py 
* Full sequence-to-sequence models.
  - basic_rnn_seq2seq: The most basic RNN-RNN model.
  - tied_rnn_seq2seq: The basic model with tied encoder and decoder weights.
  - embedding_rnn_seq2seq: The basic model with input embedding.
  - embedding_tied_rnn_seq2seq: The tied model with input embedding.
  - embedding_attention_seq2seq: Advanced model with input embedding and
      the neural attention mechanism; recommended for complex tasks.

      TODO may be self_nomralization is better for performance of train and infernce then sampled softmax

keyword state of art
batch_size:[256] batches/s:[8.44] insts/s:[2160.43]
old version on gpu0 machine batches/s:[4.13]
"""
  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_boolean('use_neg', False, 'use neg means using hinge loss(rank nce)')
flags.DEFINE_boolean('show_neg', False, 'show neg means show neg score')

import functools

import melt
logging = melt.logging
from deepiu.image_caption import conf 
from deepiu.image_caption.conf import IMAGE_FEATURE_LEN, TEXT_MAX_WORDS, NUM_RESERVED_IDS
from deepiu.util import vocabulary
from deepiu.seq2seq import embedding
import deepiu
  
class ShowAndTell(object):
  """
  ShowAndTell class is a trainer class
  but has is_training mark for ShowAndTell predictor will share some code here
  3 modes
  train,
  evaluate,
  predict
  """
  def __init__(self, is_training=True, is_predict=False):
    super(ShowAndTell, self).__init__()

    assert FLAGS.add_text_start is False

    #---------should be show_and_tell/model_init_1
    #print('ShowAndTell init:', tf.get_variable_scope().name)
    #self.abcd = melt.init_bias(3)
    #print('ShowAndTell bias', self.abcd.name)

    self.is_training = is_training 
    self.is_predict = is_predict
    self.is_evaluate = (not is_training) and (not is_predict)

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

    emb = self.emb = embedding.get_or_restore_embedding_cpu()
    if is_training and FLAGS.monitor_level > 0:
      melt.monitor_embedding(emb, vocabulary.vocab, vocabulary.vocab_size)

    self.decoder = deepiu.seq2seq.rnn_decoder.RnnDecoder(is_training, is_predict)
    self.decoder.set_embedding(emb)
    
    self.emb_dim = FLAGS.emb_dim
    #TODO for safe, can add_text_start but add 0 not calc weight or 
    #do additional cell(image_embedding, state) and pass state with start_id as input like im2text

    self.initializer = tf.random_uniform_initializer(
        minval=-FLAGS.initializer_scale,
        maxval=FLAGS.initializer_scale)

    if not FLAGS.pre_calc_image_feature:
      assert melt.apps.image_processing.image_processing_fn is not None, 'forget melt.apps.image_processing.init()'
      self.image_process_fn = functools.partial(melt.apps.image_processing.image_processing_fn,
                                                height=FLAGS.image_height, 
                                                width=FLAGS.image_width,
                                                trainable=FLAGS.finetune_image_model,
                                                is_training=is_training,
                                                random_crop=FLAGS.random_crop_image,
                                                finetune_end_point=FLAGS.finetune_end_point,
                                                distort=FLAGS.distort_image)  

  def feed_ops(self):
    """
    return feed_ops, feed_run_ops
    same as ptm example code
    not used very much, since not imporve result
    """
    if FLAGS.feed_initial_sate:
      return [self.decoder.initial_state], [self.decoder.final_state]
    else:
      return [], []

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

  #NOTICE for generative method, neg support removed to make simple!
  def build_graph(self, image_feature, text, neg_image_feature=None, neg_text=None, exact_loss=False):

    print('train:', self.is_training, 'evaluate:', self.is_evaluate, 'predict:', self.is_predict)
    
    image_emb = self.build_image_embeddings(image_feature)

    scores = self.decoder.sequence_loss(text, input=image_emb, exact_loss=exact_loss)

    if not self.is_training and not self.is_predict: #evaluate mode
      tf.add_to_collection('scores', scores)

    if not self.is_predict:
      loss = tf.reduce_mean(scores)
    else:
      loss = scores
      
    return loss

  def build_train_graph(self, image_feature, text, neg_image_feature=None, neg_text=None):
    return self.build_graph(image_feature, text, neg_text)
