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
    self.cell = melt.create_rnn_cell(num_units, cell_type='gru', initializer=initializer)


  def decode(self, 
             dec_inputs,
             initial_states, 
             enc_states, 
             ori_enc_inputs,
             feed_prev=False):

    create_attention_mechanism = melt.seq2seq.BahdanauAttention
    attention_mechanism = create_attention_mechanism(
        num_units=self.num_units,
        memory=enc_states,
        memory_sequence_length=None)
    
    cell = melt.seq2seq.AttentionWrapper(
            self.cell,
            attention_mechanism,
            cell_input_fn= lambda inputs, attention: inputs,
            attention_layer_size=None,
            initial_cell_state=initial_states, 
            alignment_history=False)
    initial_states = cell.zero_state(tf.shape(initial_states)[0], tf.float32)
    return pointer_decoder.pointer_decoder(dec_inputs, 
                                           initial_states, 
                                           enc_states, 
                                           ori_enc_inputs,
                                           cell,
                                           feed_prev=feed_prev)
