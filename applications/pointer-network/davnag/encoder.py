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
               lstm_width=6, 
               initializer=tf.random_normal_initializer(0.0, 0.5)):
    self.lstm_width = lstm_width
    self.initializer = initializer

    self.cell = tf.contrib.rnn.LSTMCell(lstm_width,
                                        input_size=None,
                                        use_peepholes=False,
                                        initializer=initializer)


  def encode(self, inputs, batch_size=1024,  input_dimensions=1, max_length=60):
    enc_state = self.cell.zero_state(batch_size, tf.float32)                # B x L: 0 is starting state for RNN
    enc_states = []
    lstm_width = self.lstm_width
    init = self.initializer
    with tf.variable_scope("rnn_encoder"):
      for j in xrange(max_length):
        if j > 0:
          tf.get_variable_scope().reuse_variables()
        input_ = inputs[:, j:j+1]                                 # B x S : step through input, 1 batch at time
        # Map the raw input to the LSTM dimensions
        W_e = tf.get_variable("W_e", [input_dimensions, lstm_width], initializer=init)  # S x L
        b_e = tf.get_variable("b_e", [batch_size, lstm_width], initializer=init)        # B x L (bias matrix)
        #b_e = tf.get_variable("b_e", [lstm_width], initializer=init)        # B x L (bias matrix)
        cell_input = tf.nn.elu(tf.matmul(input_, W_e) + b_e)                            # B x L
        # enc state has c (B x L) and h (B x L)
        output, enc_state = self.cell(cell_input, enc_state)
        enc_states.append(enc_state)   # c and h are each  B x L, and there will be J of them in list

    return enc_states
  
