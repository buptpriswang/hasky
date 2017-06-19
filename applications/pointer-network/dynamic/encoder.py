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
    # [32,1] ... [32,1] -> [32, 10, 1]
    inputs = tf.stack(inputs, 1)
    encoder_outputs, final_state = tf.nn.dynamic_rnn(self.cell, inputs, dtype=tf.float32, scope='pointer_encode')

    # Need a dummy output to point on it. End of decoding.
    batch_size = tf.shape(final_state)[0]
    encoder_outputs = tf.concat([tf.zeros([batch_size, 1, self.num_units]), encoder_outputs], 1)
    
    attention_states = encoder_outputs

    return attention_states, final_state
  
