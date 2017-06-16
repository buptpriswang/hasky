# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A pointer-network helper.
Based on attenton_decoder implementation from TensorFlow
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/rnn.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange  # pylint: disable=redefined-builtin

import tensorflow as tf

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import variable_scope as vs

from tensorflow.python.ops import rnn_cell_impl
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.python.layers import core as layers_core

import melt


def pointer_decoder(decoder_inputs, initial_state, attention_states, 
                    ori_encoder_inputs, cell,
                    feed_prev=False, dtype=dtypes.float32, scope=None):
  """RNN decoder with pointer net for the sequence-to-sequence model.
    Args:
      decoder_inputs: a list of 2D Tensors [batch_size x cell.input_size].
      initial_state: 2D Tensor [batch_size x cell.state_size].
      attention_states: 3D Tensor [batch_size x attn_length x attn_size].
      cell: rnn_cell.RNNCell defining the cell function and size.
      dtype: The dtype to use for the RNN initial state (default: tf.float32).
      scope: VariableScope for the created subgraph; default: "pointer_decoder".
    Returns:
      outputs: A list of the same length as decoder_inputs of 2D Tensors of shape
        [batch_size x output_size]. These represent the generated outputs.
        Output i is computed from input i (which is either i-th decoder_inputs.
        First, we run the cell
        on a combination of the input and previous attention masks:
          cell_output, new_state = cell(linear(input, prev_attn), prev_state).
        Then, we calculate new attention masks:
          new_attn = softmax(V^T * tanh(W * attention_states + U * new_state))
        and then we calculate the output:
          output = linear(cell_output, new_attn).
      states: The state of each decoder cell in each time-step. This is a list
        with length len(decoder_inputs) -- one item for each time-step.
        Each item is a 2D Tensor of shape [batch_size x cell.state_size].
  """
  if not decoder_inputs:
    raise ValueError("Must provide at least 1 input to attention decoder.")
  if not attention_states.get_shape()[1:2].is_fully_defined():
    raise ValueError("Shape[1] and [2] of attention_states must be known: %s"
                     % attention_states.get_shape())

  with vs.variable_scope(scope or "point_decoder"):
    batch_size = array_ops.shape(decoder_inputs[0])[0]  # Needed for reshaping.
    input_size = decoder_inputs[0].get_shape()[1].value
    attn_length = attention_states.get_shape()[1].value
    attn_size = attention_states.get_shape()[2].value
    
    num_units = attn_size
    

    attention_option =  "bahdanau"
    attention_keys, attention_values, attention_score_fn, attention_construct_fn = \
       melt.seq2seq.prepare_attention(attention_states, attention_option, num_units)


    values = attention_states
    
    states = [initial_state]
    
    outputs = []
    
    batch_attn_size = array_ops.stack([batch_size, num_units])
    attns = array_ops.zeros(batch_attn_size, dtype=dtype)
    attns.set_shape([None, num_units])
    
    inps = []
    for i in range(len(decoder_inputs)):
      if i > 0:
        vs.get_variable_scope().reuse_variables()
      inp = decoder_inputs[i]
      
      if feed_prev and i > 0:
        #->[atten_length, batch_size, input_size(1 for sort)]
        inp = tf.stack(ori_encoder_inputs)
        #->[batch_size, atten_length, input_size(1 for sort)]
        inp = tf.transpose(inp, perm=[1, 0, 2])
        inp = tf.reshape(inp, [-1, attn_length, input_size])
        inp = tf.reduce_sum(inp * alignments, [1])
        inp = tf.stop_gradient(inp)
        inps.append(inp)
       
      #inp = tf.concat([inp, attns], 1)

      # Run the RNN.
      cell_output, new_state = cell(inp, states[-1])
      states.append(new_state)
        
      # Run the attention mechanism.
      context_vector, scores, alignments = attention_score_fn(cell_output, attention_keys, attention_values)
      
      #attns = layers.linear(tf.concat([cell_output, context_vector], 1), num_units, scope="attention_layer")

      output = scores
      #output = tf.squeeze(alignments)
      
      outputs.append(output)

    return outputs, states, inps
