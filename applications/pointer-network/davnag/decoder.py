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

def get_lstm_state(cell):
    """Centralize definition of 'state', to swap .c and .h if desired"""
    return cell.c

class Decoder(object):
  def __init__(self, 
               lstm_width=6, 
               initializer=tf.random_normal_initializer(0.0, 0.5)):
    self.cell = tf.contrib.rnn.LSTMCell(lstm_width,
                                        input_size=None,
                                        use_peepholes=False,
                                        initializer=initializer)
    self.lstm_width = lstm_width
    self.initializer = initializer


  def decode(self, inputs, enc_states, 
             batch_size=1024, 
             num_blend_units=6, 
             input_dimensions=1, 
             num_indices=2, 
             input_length=60,
             generation_value=20.0):
   # # special symbol is max_length, which can never come from the actual data
   starting_generation_symbol = tf.constant(generation_value,                              # B x S
                                            shape=(batch_size,input_dimensions),
                                            dtype=tf.float32)
   dec_state = enc_states[-1]  # final enc state, both c and h; they match as 2 ( B x L )
   ptr_outputs = []
   ptr_output_dists = []
   lstm_width = self.lstm_width
   init = self.initializer
   with tf.variable_scope("rnn_decoder"):
     input_ = starting_generation_symbol    # Always B x S
     # Push out each index
     for i in xrange(num_indices):
       if i > 0:
         tf.get_variable_scope().reuse_variables()
         # Map the raw input to the LSTM dimensions
       W_d_in = tf.get_variable("W_d_in", [input_dimensions, lstm_width], initializer=init)   # S x L
       b_d_in = tf.get_variable("b_d_in", [batch_size, lstm_width], initializer=init)         # B x L
       cell_input = tf.nn.elu(tf.matmul(input_, W_d_in) + b_d_in)                               # B x L
       
       output, dec_state = self.cell(cell_input, dec_state)         # Output: B x L    Dec State.c = B x L

       # Enc/dec states (.c) are B x S
       # We want to map these to 1, right?  BxS and something that maps to B alone
       W_1 = tf.get_variable("W_1", [lstm_width, num_blend_units], initializer=init)            # L x D
       W_2 = tf.get_variable("W_2", [lstm_width, num_blend_units], initializer=init)            # L x D
       bias_ptr = tf.get_variable("bias_ptr", [batch_size, num_blend_units], initializer=init)  # B x D
       
       index_predists = []
       # Loop over each input slot to set up the softmax distribution
       dec_portion = tf.matmul(get_lstm_state(dec_state), W_2)                   # B x D
       
       enc_portions = []
       
       # Vector to blend
       v_blend = tf.get_variable("v_blend", [num_blend_units, 1], initializer=init)   # D x 1
       
       for input_length_index in xrange(input_length):
         # Use the cell values (.c), not the output (.h) values of each state
         # Each is B x 1, and there are J of them. Flatten to J x B
         enc_portion = tf.matmul(get_lstm_state(enc_states[input_length_index]), W_1)         # B x D
         raw_blend = tf.nn.elu(enc_portion + dec_portion + bias_ptr)                          # B x D
         scaled_blend = tf.matmul(raw_blend, v_blend)                                         # B x 1
         index_predist = tf.reshape(scaled_blend, (batch_size,))                              # B
         
         enc_portions.append(enc_portion)
         index_predists.append(index_predist)
         
       idx_predistribution = tf.transpose(tf.stack(index_predists))                             # B x J
       # Now, do softmax over predist, on final dim J (input length), to get to real dist
       idx_distribution = tf.nn.softmax(idx_predistribution, dim=-1)                            # B x J
       ptr_output_dists.append(idx_distribution)
       idx = tf.argmax(idx_distribution, 1)  # over last dim, rank reduc                        # B
       # Pull out the input from that index
       emb = tf.nn.embedding_lookup(tf.transpose(inputs), idx)                                  # B x B
       ptr_output_raw = tf.diag_part(emb)                                                       # B
       ptr_output = tf.reshape(ptr_output_raw, (batch_size, input_dimensions))                  # B x S
       ptr_outputs.append(ptr_output)
       input_ = ptr_output    # The output goes straight back in as next input
       # Compare the one-hot distribution (actuals) vs. the softmax distribution: I x (B x J)
   idx_distributions = tf.stack(ptr_output_dists)                                                   # I x B x J
   return idx_distributions, ptr_outputs
