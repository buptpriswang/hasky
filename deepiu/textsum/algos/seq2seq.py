#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   seq2seq.py
#        \author   chenghuige  
#          \date   2016-12-22 20:07:46.827344
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS

import melt

from deepiu.seq2seq import embedding
from deepiu.seq2seq.rnn_decoder import SeqDecodeMethod
from deepiu import seq2seq

#--for Seq2seqPredictor
from deepiu.textsum.conf import INPUT_TEXT_MAX_WORDS, TEXT_MAX_WORDS
from deepiu.util import vocabulary 
from deepiu.util.text2ids import idslist2texts

class Seq2seq(object):
  def __init__(self, is_training=True, is_predict=False):
    super(Seq2seq, self).__init__()

    #assert FLAGS.rnn_output_method == 'all', 'attention need to encode all'

    self.is_training = is_training 
    self.is_predict = is_predict
    self.is_evaluate = (not is_training) and (not is_predict)

    emb = embedding.get_or_restore_embedding_cpu()
    if is_training and FLAGS.monitor_level > 0:
      melt.monitor_embedding(emb, vocabulary.vocab, vocabulary.vocab_size)

    self.encoder = seq2seq.rnn_encoder.RnnEncoder(is_training, is_predict)
    self.encoder.set_embedding(emb)

    #emb2 = embedding.get_embedding('emb2')
    if not FLAGS.experiment_rnn_decoder:
      self.decoder = seq2seq.rnn_decoder.RnnDecoder(is_training, is_predict)
    else:
      self.decoder = seq2seq.experiment.rnn_decoder.RnnDecoder(is_training, is_predict)
    self.decoder.set_embedding(emb)
    
    print('start_id', self.decoder.start_id)

    assert FLAGS.add_text_start is True 
    assert self.decoder.start_id is not None

  def build_graph(self, input_text, text, 
                  exact_prob=False, exact_loss=False):
    """
    exact_prob and exact_loss actually do the same thing,
    they only be used on when is_predict is true
    """
    print('train:', self.is_training, 'evaluate:', self.is_evaluate, 'predict:', self.is_predict)
    assert not (exact_prob and exact_loss)
    assert not ((not self.is_predict) and (exact_prob or exact_loss))
    with tf.variable_scope("encode"):
      encoder_output, state = self.encoder.encode(input_text, output_method=melt.rnn.OutputMethod.all)
      if not FLAGS.use_attention:
        encoder_output = None
    with tf.variable_scope("decode"):
      #input_text, input_text_length = melt.pad(input_text, end_id=self.encoder.end_id)
      input_text = self.encoder.sequence
      loss = self.decoder.sequence_loss(text, 
                                        input_text=input_text,
                                        initial_state=state, 
                                        attention_states=encoder_output,
                                        exact_prob=exact_prob, 
                                        exact_loss=exact_loss)
      self.ori_loss = self.decoder.ori_loss #without average step for prediction
      #this is used in train.py for evaluate eval_scores = tf.get_collection('scores')[-1]
      #because we want to show loss for each instance
      
      if self.is_evaluate:
       tf.add_to_collection('scores', loss)
    
    if not self.is_predict:
      loss = tf.reduce_mean(loss)

    return loss

  def build_train_graph(self, input_text, text):
    return self.build_graph(input_text, text)

class Seq2seqPredictor(Seq2seq, melt.PredictorBase):
  def __init__(self):
    melt.PredictorBase.__init__(self)
    Seq2seq.__init__(self, is_training=False, is_predict=True)

    self.input_text_feed = tf.placeholder(tf.int64, [None, INPUT_TEXT_MAX_WORDS], name='input_text')
    tf.add_to_collection('input_text_feed', self.input_text_feed)
    tf.add_to_collection('lfeed', self.input_text_feed)
    tf.add_to_collection('feed', self.input_text_feed)
    self.text_feed = tf.placeholder(tf.int64, [None, TEXT_MAX_WORDS], name='text')
    tf.add_to_collection('text_feed', self.text_feed)
    tf.add_to_collection('rfeed', self.text_feed)

  def init_predict_text(self, decode_method=0, beam_size=5, convert_unk=True):
    """
    init for generate texts
    """
    text, score = self.build_predict_text_graph(self.input_text_feed, 
                                                decode_method=decode_method, 
                                                beam_size=beam_size, 
                                                convert_unk=convert_unk)
    return text, score

  def init_predict(self, exact_prob=False, exact_loss=False):
    score, ori_score = self.build_predict_graph(self.input_text_feed, 
                                     self.text_feed, 
                                     exact_prob=exact_prob, 
                                     exact_loss=exact_loss)
    #tf.add_to_collection('score', score)
    self.score = score
    return score, ori_score

  def predict(self, input_text, text):
    """
    default usage is one single image , single text predict one sim score
    """
    feed_dict = {
      self.input_text_feed: input_text,
      self.text_feed: text,
    }
    score = self.sess.run(self.score, feed_dict)
    return score

  def build_predict_text_graph(self, input_text, decode_method='greedy', beam_size=5, convert_unk=True):
    with tf.variable_scope("encode"):
      encoder_output, state = self.encoder.encode(input_text, output_method=melt.rnn.OutputMethod.all)
      if not FLAGS.use_attention:
        encoder_output = None
    with tf.variable_scope("decode"):
      #---try to use static shape if possible
      batch_size = melt.get_batch_size(input_text)
      decoder_input = self.decoder.get_start_embedding_input(batch_size)
      max_words = FLAGS.decode_max_words if FLAGS.decode_max_words else TEXT_MAX_WORDS
      if decode_method == SeqDecodeMethod.greedy:
        input_text = self.encoder.sequence
        return self.decoder.generate_sequence_greedy(decoder_input, 
                                       max_words=max_words, 
                                       initial_state=state,
                                       attention_states=encoder_output,
                                       convert_unk=convert_unk,
                                       input_text=input_text)
      else:
        if decode_method == SeqDecodeMethod.ingraph_beam:
          decode_func = self.decoder.generate_sequence_ingraph_beam
        elif decode_method == SeqDecodeMethod.outgraph_beam:
          decode_func = self.decoder.generate_sequence_outgraph_beam
        else:
          raise ValueError('not supported decode_method: %s' % decode_method)
        
        input_text, input_text_length = melt.pad(input_text, end_id=self.encoder.end_id)
        #input_text = self.encoder.sequence
        #input_text_length = self.encoder.sequence_length
        return decode_func(decoder_input, 
                           max_words=max_words, 
                           initial_state=state,
                           attention_states=encoder_output,
                           beam_size=beam_size, 
                           convert_unk=convert_unk,
                           length_normalization_factor=FLAGS.length_normalization_factor,
                           input_text=input_text,
                           input_text_length=input_text_length)

  def build_predict_graph(self, input_text, text, exact_prob=False, exact_loss=False):
    input_text = tf.reshape(input_text, [-1, INPUT_TEXT_MAX_WORDS])
    text = tf.reshape(text, [-1, TEXT_MAX_WORDS])
  
    #loss is -logprob  
    loss = self.build_graph(input_text, text, 
                            exact_prob=exact_prob, 
                            exact_loss=exact_loss)

 
    #score is logprob
    score = -loss 
    ori_score = -self.ori_loss
    if FLAGS.predict_use_prob:
      score = tf.exp(score)
      ori_score = tf.exp(ori_score)

    return score, ori_score
