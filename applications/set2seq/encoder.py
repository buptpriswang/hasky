#!/usr/bin/env python
# ==============================================================================
#          \file   encoder.py
#        \author   chenghuige  
#          \date   2017-05-16 16:37:16.417121
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import melt  

import attention_encoder

class Encoder(object):
  def __init__(self, 
               num_units=32, 
               num_attention_steps=0,
               initializer=None):
    self.num_units = num_units
    self.num_attention_steps = num_attention_steps
    #self.cell = melt.create_rnn_cell(num_units, cell_type='gru', initializer=initializer)
    self.cell = melt.create_rnn_cell(num_units, cell_type='lstm', initializer=initializer)

  def encode(self, inputs):
    with tf.variable_scope('encoder') as scope:
      batch_size = tf.shape(inputs[0])[0]
      if not self.num_attention_steps:
        cell = self.cell 
        # Need for attention
        encoder_outputs, final_state = tf.contrib.rnn.static_rnn(cell, inputs, dtype=tf.float32, scope='pointer_encode')
        # Need a dummy output to point on it. End of decoding.
        #batch_size = tf.shape(final_state)[0]
        encoder_outputs = [tf.zeros([batch_size, self.num_units])] + encoder_outputs
        # First calculate a concatenation of encoder outputs to put attention on.
        top_states = [tf.reshape(e, [-1, 1, cell.output_size])
                    for e in encoder_outputs]
        attention_states = tf.concat(top_states, 1)
        return attention_states, final_state, None
      else:
        inputs = tf.stack(inputs, 1)
        create_attention_mechanism = melt.seq2seq.BahdanauAttention
        #create_attention_mechanism = melt.seq2seq.LuongAttention

        memory = tf.concat([tf.zeros([batch_size, 1, 1]), inputs], 1)

        attention_mechanism = create_attention_mechanism(
            num_units=self.num_units,
            #num_units=512,
            #memory=inputs,
            memory=memory,
            memory_sequence_length=None)
         
        cell = melt.seq2seq.AttentionWrapper(
                self.cell,
                attention_mechanism,
                attention_layer_size=None,
                initial_cell_state=None, 
                alignment_history=False,
                no_context=False)

        initial_state = cell.zero_state(batch_size, tf.float32)
        encoder_inputs = [tf.zeros([batch_size, 1])] * self.num_attention_steps
        #encoder_inputs = [tf.ones([batch_size, 1]) * 0.13] * self.num_attention_steps
        #encoder_inputs = [memory[:,0]] * self.num_attention_steps
        #encoder_inputs = melt.get_weights_truncated('encoder_input', [batch_size, 1])

        encoder_outputs, encoder_states = attention_encoder.attention_encoder(encoder_inputs, initial_state, cell)

        final_state = encoder_states[-1].cell_state

        #attention_states =  tf.concat([tf.zeros([batch_size, 1, 1]), inputs], 1)
        attention_states = memory

        #attention_mechanism = None

        return attention_states, final_state, attention_mechanism
  
