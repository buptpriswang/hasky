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

from legacy import pointer_decoder

class Decoder(object):
  def __init__(self, 
               num_units=32, 
               initializer=None):
    self.cell = melt.create_rnn_cell(num_units, cell_type='gru', initializer=initializer)


  def decode(self, 
             dec_inputs,
             initial_states, 
             enc_states, 
             ori_enc_inputs,
             feed_prev=False):
    return pointer_decoder.pointer_decoder(dec_inputs, 
                                           initial_states, 
                                           enc_states, 
                                           ori_enc_inputs,
                                           self.cell,
                                           feed_prev=feed_prev)
