#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   show_and_tell.py
#        \author   chenghuige  
#          \date   2016-09-04 17:49:20.030172
#   \Description  
# ==============================================================================
"""
show and tell not just follow paper it can be viewed as a simple generative method frame work
also support show attend and tell using this framework
there are slightly difference with google open source im2txt see FLAGS.image_as_init_state
"""
  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_boolean('image_as_init_state', False, 'by default will treat image as input not inital state(im2txt usage)')
flags.DEFINE_boolean('show_atten_tell', False, 'wether to use attention as in paper Show,Attend and Tell: Neeural Image Caption Generation with Visual Attention')
flags.DEFINE_string('image_encoder', 'ShowAndTell', '')

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

    if FLAGS.image_as_init_state:
      #just use default method here is ok!
      assert FLAGS.add_text_start is True, 'need to add text start for im2tx mode'
    else:
      #just for experiment to be same as im2txt but result is worse
      assert FLAGS.add_text_start is False, 'normal mode must not add text start'

    self.is_training = is_training 
    self.is_predict = is_predict
    self.is_evaluate = (not is_training) and (not is_predict)

    #if is_training:
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

    self.emb_dim = FLAGS.emb_dim
    
    self.using_attention = FLAGS.image_encoder != 'ShowAndTell'

    ImageEncoder = deepiu.seq2seq.image_encoder.Encoders[FLAGS.image_encoder]
    #juse for scritps backward compact, TODO remove show_atten_tell
    if FLAGS.show_atten_tell:
      logging.info('warning, show_atten_tell mode depreciated, just set --image_encoder=')
      ImageEncoder = deepiu.seq2seq.image_encoder.MemoryEncoder
    
    self.encoder = ImageEncoder(is_training, is_predict, self.emb_dim)

    self.decoder = deepiu.seq2seq.rnn_decoder.RnnDecoder(is_training, is_predict)
    self.decoder.set_embedding(emb)
    

    if not FLAGS.pre_calc_image_feature:
      assert melt.apps.image_processing.image_processing_fn is not None, 'forget melt.apps.image_processing.init()'
      self.image_process_fn = functools.partial(melt.apps.image_processing.image_processing_fn,
                                                height=FLAGS.image_height, 
                                                width=FLAGS.image_width,
                                                trainable=FLAGS.finetune_image_model,
                                                is_training=is_training,
                                                random_crop=FLAGS.random_crop_image,
                                                finetune_end_point=FLAGS.finetune_end_point,
                                                distort=FLAGS.distort_image,
                                                feature_name=FLAGS.image_endpoint_feature_name)  
    else:
      self.image_process_fn = None

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

  def process(self, image_feature):
    if self.image_process_fn is not None:
      image_feature = self.image_process_fn(image_feature) 
    if self.using_attention:
      image_feature = tf.reshape(image_feature, [-1, FLAGS.image_attention_size, int(IMAGE_FEATURE_LEN / FLAGS.image_attention_size)])
    return image_feature


  #NOTICE for generative method, neg support removed to make simple!
  def build_graph(self, image_feature, text, neg_image_feature=None, neg_text=None, exact_loss=False):
    attention_states, initial_state, image_emb = self.encoder.encode(self.process(image_feature))

    if not FLAGS.image_as_init_state:
      #mostly go here
      scores = self.decoder.sequence_loss(text, 
                                          input=image_emb, 
                                          initial_state=initial_state, 
                                          attention_states=attention_states, 
                                          exact_loss=exact_loss)
    else:
      #for im2txt one more step at first, just for exp not used much 
      with tf.variable_scope(self.decoder.scope) as scope:
        zero_state = self.decoder.cell.zero_state(batch_size=melt.get_batch_size(input), dtype=tf.float32)
        _, initial_state = self.decoder.cell(input, zero_state)
      #will pad start in decoder.sequence_loss
      scores = self.decoder.sequence_loss(text,
                                          input=None, 
                                          initial_state=initial_state, 
                                          attention_states=attention_states, 
                                          exact_loss=exact_loss)

    if not self.is_training and not self.is_predict: #evaluate mode
      tf.add_to_collection('scores', scores)

    if not self.is_predict:
      loss = tf.reduce_mean(scores)
    else:
      loss = scores
      
    return loss

  def build_train_graph(self, image_feature, text, neg_image_feature=None, neg_text=None):
    return self.build_graph(image_feature, text, neg_text)
