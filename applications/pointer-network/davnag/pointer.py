#!/usr/bin/env python
# ==============================================================================
#          \file   pointer.py
#        \author   chenghuige  
#          \date   2017-05-16 16:34:26.083478
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import melt 

from encoder import Encoder
from decoder import Decoder

class Pointer(object):
  def __init__(self):
    self.encoder = Encoder()
    self.decoder = Decoder()
    
    batch_size = 1024
    input_length = 60
    num_indices = 2
    self.inputs = tf.placeholder(tf.float32, name="ptr-in", shape=(batch_size, input_length))      # B x J
    # The one hot (over J) distributions, by batch and by index (start=1 and end=2)
    self.actual_index_dists = tf.placeholder(tf.float32,                                           # I x B x J
                                        name="ptr-out",
                                        shape=(num_indices, batch_size, input_length))

  def build(self):
    enc_states = self.encoder.encode(self.inputs)
    idx_distributions, ptr_outputs = self.decoder.decode(self.inputs, enc_states)
    loss = tf.sqrt(tf.reduce_mean(tf.pow(idx_distributions - self.actual_index_dists, 2.0)))
    return loss
  
