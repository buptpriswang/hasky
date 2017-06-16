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

class Encoder(object):
  def __init__(self, 
               num_units=32, 
               initializer=None):
    self.num_units = num_units
    self.cell = melt.create_rnn_cell(num_units, cell_type='gru', initializer=initializer)
    

  def encode(self, inputs):
    # Need for attention
    encoder_outputs, final_state = tf.contrib.rnn.static_rnn(self.cell, inputs, dtype=tf.float32, scope='pointer_encode')
    # Need a dummy output to point on it. End of decoding.
    batch_size = tf.shape(final_state)[0]
    encoder_outputs = [tf.zeros([batch_size, self.num_units])] + encoder_outputs
    
    # First calculate a concatenation of encoder outputs to put attention on.
    top_states = [tf.reshape(e, [-1, 1, self.cell.output_size])
                  for e in encoder_outputs]
    attention_states = tf.concat(axis=1, values=top_states)

    return attention_states, final_state
  
