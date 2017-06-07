#!/usr/bin/env python
# ==============================================================================
#          \file   pointer_network.py
#        \author   chenghuige  
#          \date   2017-06-06 22:07:12.547099
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np 
import tensorflow as tf
import melt 

from encoder import Encoder
from decoder import Decoder

class PointerNetwork(object):
  def __init__(self, max_len, batch_size, num_units=32, input_size=1):
    self.encoder = Encoder(num_units=num_units)
    self.decoder = Decoder(num_units=num_units)
    
    self.encoder_inputs = []
    self.decoder_inputs = []
    self.decoder_targets = []
    self.target_weights = []
    
    self.input_size = input_size
    self.batch_size = batch_size

    for i in range(max_len):
      self.encoder_inputs.append(tf.placeholder(
        tf.float32, [batch_size, input_size], name="EncoderInput%d" % i))
    for i in range(max_len + 1):
      self.decoder_inputs.append(tf.placeholder(
        tf.float32, [batch_size, input_size], name="DecoderInput%d" % i))
      self.decoder_targets.append(tf.placeholder(
        tf.int32, [batch_size, 1], name="DecoderTarget%d" % i))  
      self.target_weights.append(tf.placeholder(
        tf.float32, [batch_size, 1], name="TargetWeight%d" % i))
    
  def create_feed_dict(self, encoder_input_data, decoder_input_data, decoder_target_data):
    feed_dict = {}
    for placeholder, data in zip(self.encoder_inputs, encoder_input_data):
      feed_dict[placeholder] = data
     
    for placeholder, data in zip(self.decoder_inputs, decoder_input_data):
      feed_dict[placeholder] = data
      
    for placeholder, data in zip(self.decoder_targets, decoder_target_data):
      feed_dict[placeholder] = data
      
    for placeholder in self.target_weights:
      feed_dict[placeholder] = np.ones([self.batch_size, 1])
      
    return feed_dict
                                 
  def build(self, feed_prev=False):
    encoder_outputs, final_state = self.encoder.encode(self.encoder_inputs)
    encoder_inputs = [tf.zeros([self.batch_size, 1])] + self.encoder_inputs
    
    decoder_inputs = self.decoder_inputs if not feed_prev else [self.decoder_inputs[0]] * len(self.decoder_inputs)
    outputs, states, inps = self.decoder.decode(decoder_inputs, final_state, encoder_outputs, encoder_inputs, feed_prev)
    
    outputs = [tf.expand_dims(e, 1) for e in outputs]

    outputs = tf.concat(outputs, 1)
    targets = tf.concat(self.decoder_targets, 1)
    weights = tf.concat(self.target_weights, 1)
    
    print(outputs, targets, weights)
    loss = melt.seq2seq.sequence_loss_by_example(outputs, targets, weights)
    loss = tf.reduce_mean(loss)
    
    predicts = tf.to_int32(tf.argmax(outputs, 2))

    correct_predict_ratio = tf.reduce_mean(tf.to_float(melt.sequence_equal(predicts, targets)))

    return loss, correct_predict_ratio, predicts, targets
