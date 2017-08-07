#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   rnn.py
#        \author   chenghuige  
#          \date   2016-12-23 14:02:57.513674
#   \Description  
# ==============================================================================

"""
rnn encoding
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from melt.ops import dynamic_last_relevant

import copy

import melt
  
#TODO change from 0, 1 .. to 'forward', 'backward' , 'sum', 'last'
class EncodeMethod:
  forward = 'forward'
  backward = 'backward'
  bidirectional = 'bidirectional'
  bidirectional_sum = 'bidirectional_sum'

class OutputMethod:
  sum = 'sum'
  last = 'last'
  first = 'first'
  all = 'all'
  mean = 'mean'
  max = 'max'
  argmax = 'argmax'

def encode_outputs(outputs, output_method=OutputMethod.last, sequence_length=None):
  #--seems slower convergence and not good result when only using last output, so change to use sum
  if output_method == OutputMethod.sum:
    return tf.reduce_sum(outputs, 1)
  elif output_method == OutputMethod.max:
    assert sequence_length is not None
    #below not work.. sequence is different for each row instance
    #return tf.reduce_max(outputs[:, :sequence_length, :], 1)
    #return tf.reduce_max(outputs, 1) #not exclude padding embeddings
    #return tf.reduce_max(tf.abs(outputs), 1)
    return melt.max_pooling(outputs, sequence_length)
  elif output_method == OutputMethod.argmax:
    assert sequence_length is not None
    #return tf.argmax(outputs[:, :sequence_length, :], 1)
    #return tf.argmax(outputs, 1)
    #return tf.argmax(tf.abs(outputs), 1)
    return melt.argmax_pooling(outputs, sequence_length)
  elif output_method == OutputMethod.mean:
    assert sequence_length is not None
    return tf.reduce_sum(outputs, 1) / tf.to_float(tf.expand_dims(sequence_length, 1)), state 
  elif output_method == OutputMethod.last:
    #TODO actually return state.h is last revlevant?
    return dynamic_last_relevant(outputs, sequence_length)
  elif output_method == OutputMethod.first:
    return outputs[:, 0, :]
  else: # all
    return outputs

def forward_encode(cell, inputs, sequence_length, initial_state=None, dtype=None, output_method=OutputMethod.last):
  outputs, state = tf.nn.dynamic_rnn(
    cell, 
    inputs, 
    initial_state=initial_state, 
    dtype=dtype,
    sequence_length=sequence_length)
  
  return encode_outputs(outputs, output_method, sequence_length), state


def backward_encode(cell, inputs, sequence_length, initial_state=None, dtype=None, output_method=OutputMethod.last):
  outputs, state = tf.nn.dynamic_rnn(
    cell, 
    tf.reverse_sequence(inputs, sequence_length, 1), 
    initial_state=initial_state, 
    dtype=dtype,
    sequence_length=sequence_length)

  return encode_outputs(outputs, output_method, sequence_length), state

def bidirectional_encode(cell_fw, 
                        cell_bw, 
                        inputs, 
                        sequence_length, 
                        initial_state_fw=None, 
                        initial_state_bw=None, 
                        dtype=None,
                        output_method=OutputMethod.last,
                        use_sum=False):
  if cell_bw is None:
    cell_bw = copy.deepcopy(cell_fw)
  if initial_state_bw is None:
    initial_state_bw = initial_state_fw

  outputs, states  = tf.nn.bidirectional_dynamic_rnn(
    cell_fw=cell_fw,
    cell_bw=cell_bw,
    inputs=inputs,
    initial_state_fw=initial_state_fw,
    initial_state_bw=initial_state_bw,
    dtype=dtype,
    sequence_length=sequence_length)

  output_fws, output_bws = outputs

  output_forward = encode_outputs(output_fws, output_method, sequence_length)
  output_backward = encode_outputs(output_bws, output_method, sequence_length)

  if output_method == OutputMethod.sum:
    output_backward = tf.reduce_sum(output_bws, 1) 

  if use_sum:
    output = output_forward + output_backward
  else:
    output = tf.concat([output_forward, output_backward], -1)

  #TODO state[0] ?
  return output, states[0]

def encode(cell, 
           inputs, 
           sequence_length, 
           initial_state=None, 
           cell_bw=None, 
           inital_state_bw=None, 
           dtype=None,
           encode_method=EncodeMethod.forward, 
           output_method=OutputMethod.last):
    
    #needed for bidirectional_dynamic_rnn and backward method
    #without it Input 'seq_lengths' of 'ReverseSequence' Op has type int32 that does not match expected type of int64.
    #int tf.reverse_sequence seq_lengths: A `Tensor` of type `int64`.
    if initial_state is None and dtype is None:
      dtype = tf.float32
    sequence_length = tf.cast(sequence_length, tf.int64)
    if encode_method == EncodeMethod.forward:
      return forward_encode(cell, inputs, sequence_length, initial_state, dtype, output_method)
    elif encode_method == EncodeMethod.backward:
      return backward_encode(cell, inputs, sequence_length, initial_state, dtype, output_method)
    elif encode_method == EncodeMethod.bidirectional:
      return bidirectional_encode(cell, cell_bw, inputs, sequence_length, 
                                 initial_state, inital_state_bw, dtype, output_method)
    elif encode_method == EncodeMethod.bidirectional_sum:
      return bidirectional_encode(cell, cell_bw, inputs, sequence_length, 
                                 initial_state, inital_state_bw, dtype, output_method,
                                 use_sum=True)
    else:
      raise ValueError('Unsupported rnn encode method:', encode_method)