#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   rnn_encoder.py
#        \author   chenghuige  
#          \date   2016-12-23 23:59:59.013362
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS
  
flags.DEFINE_string('rnn_method', 'forward', '0 forward, 1 backward, 2 bidirectional')
flags.DEFINE_string('rnn_output_method', 'sum', '0 sumed vec, 1 last vector, 2 first vector, 3 all here first means first to original sequence')

flags.DEFINE_boolean('encode_start_mark', False, """need <S> start mark""")
flags.DEFINE_boolean('encode_end_mark', False, """need </S> end mark""")
flags.DEFINE_string('encoder_end_mark', '</S>', "or <GO> if NUM_RESERVED_IDS >=3, will use id 2  <PAD2> as <GO>, especailly for seq2seq encoding")

import functools
import melt
logging = melt.logging

from deepiu.util import vocabulary  

from deepiu.seq2seq.encoder import Encoder

class RnnEncoder(Encoder):
  """
  this is for text as input will first embedding lookup
  """
  def __init__(self, is_training=True, is_predict=False):
    super(RnnEncoder, self).__init__()
    self.is_training = is_training
    self.is_predict = is_predict
    
    vocabulary.init()

    if FLAGS.encoder_end_mark == '</S>':
      self.end_id =  vocabulary.end_id()
    else:
      self.end_id = vocabulary.go_id() #NOTICE NUM_RESERVED_IDS must >= 3 TODO
    assert self.end_id != vocabulary.vocab.unk_id(), 'input vocab generated without end id'
    
    create_rnn_cell = functools.partial(
        melt.create_rnn_cell, 
        num_units=FLAGS.rnn_hidden_size,
        is_training=is_training, 
        keep_prob=FLAGS.keep_prob, 
        num_layers=FLAGS.num_layers, 
        cell_type=FLAGS.cell)

    #follow models/textsum
    self.cell = create_rnn_cell(initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=123));
    if FLAGS.rnn_method == melt.rnn.EncodeMethod.bidirectional:
      self.bwcell = create_rnn_cell(initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=113));
    else:
      self.bwcell = None

  def pad(self, sequence):
    return melt.pad(sequence, 
                    start_id=(vocabulary.vocab.start_id() if FLAGS.encode_start_mark else None),
                    end_id=(self.end_id if FLAGS.encode_end_mark else None))
  
  #TODO  add scope
  def encode(self, sequence, emb=None, input=None, output_method=None, 
             embedding_lookup=False, method=None):
    if emb is None:
      emb = self.emb 

    #--for debug
    if embedding_lookup:
      sequence, sequence_length = self.pad(sequence)
    else:
      sequence_length = tf.ones([melt.get_batch_size(sequence),], dtype=tf.int32) * tf.shape(sequence)[1]

    self.sequence = sequence
    self.sequence_length = sequence_length

    tf.add_to_collection('debug_seqeuence', sequence)
    tf.add_to_collection('debug_length', sequence_length)
    
    #for attention due to float32 numerice accuracy problem, may has some diff, so not slice it
    #if self.is_predict:
    #  num_steps = tf.cast(tf.reduce_max(sequence_length), dtype=tf.int32)
    #  sequence = tf.slice(sequence, [0,0], [-1, num_steps])   
    
    if embedding_lookup:
      inputs = tf.nn.embedding_lookup(emb, sequence) 
    else:
      inputs = sequence

    if input is not None:
      inputs = tf.concat([tf.expand_dims(input, 1), inputs], 1)
      sequence_length += 1

    if self.is_training and FLAGS.keep_prob < 1:
      inputs = tf.nn.dropout(inputs, FLAGS.keep_prob)

    encode_feature, state = melt.rnn.encode(
          self.cell, 
          inputs, 
          sequence_length, 
          cell_bw=self.bwcell,
          encode_method=method or FLAGS.rnn_method,
          output_method=output_method or FLAGS.rnn_output_method)

    return encode_feature, state

  def words_importance_encode(self, sequence, emb=None, input=None):
    #[batch_size, emb_dim]
    argmax_values = self.encode(sequence, emb, input, output_method=melt.rnn.OutputMethod.argmax)[0]
    indices = melt.batch_values_to_indices(tf.to_int32(argmax_values))
    updates = tf.ones_like(argmax_values)
    shape = tf.shape(sequence)
    scores = tf.scatter_nd(indices, updates, shape=shape) * tf.to_int64(tf.sequence_mask(self.sequence_length, shape[-1]))
    return scores
