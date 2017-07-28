#!/usr/bin/env python
# ==============================================================================
#          \file   decoder.py
#        \author   chenghuige  
#          \date   2017-05-16 16:37:20.386971
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import melt

import pointer_decoder

class Decoder(object):
  def __init__(self, 
               num_units=32, 
               initializer=None):
    self.num_units = num_units
    #self.cell = melt.create_rnn_cell(num_units, cell_type='gru', initializer=initializer)
    self.cell = melt.create_rnn_cell(num_units, cell_type='lstm', initializer=initializer)

  def decode(self, 
             dec_inputs,
             initial_states, 
             enc_states, 
             ori_enc_inputs,
             feed_prev=False,
             encode_attention=None):
    with tf.variable_scope('decoder') as scope:
      #using melt.seq2seq instead of tf.contrib.seq2seq due to scope problem of using Dense see 
      #https://github.com/tensorflow/tensorflow/issues/10731
      #also modify AttentionWrapper for pointer network which do not need context, adding no_context argument
      if encode_attention is None:
        create_attention_mechanism = melt.seq2seq.BahdanauAttention #for this problem bah is much better then luong
        #create_attention_mechanism = melt.seq2seq.LuongAttention
        attention_mechanism = create_attention_mechanism(
            num_units=self.num_units,
            memory=enc_states,
            memory_sequence_length=None)
      else:
        attention_mechanism = encode_attention
      
      cell = melt.seq2seq.AttentionWrapper(
              self.cell,
              attention_mechanism,
              cell_input_fn= lambda inputs, attention: inputs, #seems better then use concat since we use feed prev
              attention_layer_size=None,
              initial_cell_state=initial_states, 
              probability_fn=lambda score: score,
              alignment_history=False,
              no_context=True)
      
      batch_size = tf.shape(ori_enc_inputs)[0]
      #initial_states = cell.zero_state(tf.shape(initial_states)[0], tf.float32)
      initial_states = cell.zero_state(batch_size, tf.float32)
      return pointer_decoder.pointer_decoder(dec_inputs, 
                                             initial_states, 
                                             enc_states, 
                                             ori_enc_inputs,
                                             cell,
                                             feed_prev=feed_prev)
