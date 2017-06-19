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

"""
experiment:

train + test both feed_prev=True this is like train + evaluate 
since actually for inference test need feed_prev=False  TODO 

--without attention just seq2seq:
['eval_loss:1.298', 'correct_predict_ratio:0.000']
label--: [ 9  2  8  7  6 10  1  4  5  3  0]
predict: [ 9  9  8  7  7 10 10 10  5  5  0]
label--: [10  4  8  3  6  9  2  7  5  1  0]
predict: [10 10  8  6  6  9  9  7  7  3  0]
2017-06-19 15:21:50 0:02:06  train_step:9000 duration:0.018 elapsed:[1.220] batch_size:[32] batches/s:[81.98] insts/s:[2623.29] train_avg_metric:['loss:1.301'] 
[1.2608]
2017-06-19 15:21:50 0:02:06  eval_step: 9000 duration:0.011 elapsed:1.220 eval_time_ratio:0.009 

--with attention just seq2seq no point decode:
2017-06-19 15:29:01 0:02:45  eval_step: 9000 eval_metrics:
['eval_loss:1.029', 'correct_predict_ratio:0.000']
label--: [ 1  6  9  3 10  2  5  8  7  4  0]
predict: [ 1  1  6 10 10 10  8  7  4  5  0]
label--: [10  9  6  7  2  8  5  3  4  1  0]
predict: [10  9  7  7  2  8  8  4  4  1  0]
2017-06-19 15:29:01 0:02:45  train_step:9000 duration:0.025 elapsed:[1.676] batch_size:[32] batches/s:[59.66] insts/s:[1909.05] train_avg_metric:['loss:1.086'] 

---without attention for input just using attention aligments, pointer network!

2017-06-19 16:36:57 0:02:20  eval_step: 9000 eval_metrics:
['eval_loss:0.199', 'correct_predict_ratio:0.625']
label--: [ 5  4  1  6 10  7  8  2  9  3  0]
predict: [ 5  4  1  6 10  7  8  2  3  3  0]
label--: [ 5  1  4  8 10  7  2  9  6  3  0]
predict: [ 5  1  4  8 10  7  2  9  6  3  0]
2017-06-19 16:36:57 0:02:20  train_step:9000 duration:0.026 elapsed:[1.730] batch_size:[32] batches/s:[57.80] insts/s:[1849.54] train_avg_metric:['loss:0.228'] 

Ok, pointer network is much better, but I'm not sure why get aligments using stack 
make speed much slower..., slower even then staic pointer network.. 

compare to static, result is same, static will not be slower if dynamic not using early stop(different length)
['eval_loss:0.199', 'correct_predict_ratio:0.500']
label--: [ 8  1  9  6  3  7 10  2  5  4  0]
predict: [ 8  1  9  6  3  7 10  2  4  4  0]
label--: [10  9  2  8  4  3  6  7  1  5  0]
predict: [10  9  2  8  4  3  6  7  1  5  0]
2017-06-19 16:46:59 0:02:03  train_step:9000 duration:0.020 elapsed:[1.451] batch_size:[32] batches/s:[68.91] insts/s:[2205.21] train_avg_metric:['loss:0.217'] 


change to output alignments directly also ok and faster 
but strange dyanmic_rnn still slower then static version int static2
seems not dynmic_rnn slow but using output_alignment will be slowe... why? TODO
"""
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
  print('feed_prev', feed_prev)
  if not feed_prev:
    outputs, state = tf.nn.dynamic_rnn(cell, 
                                       decoder_inputs, 
                                       initial_state=initial_state, 
                                       dtype=tf.float32,
                                       scope='point_decoder')

    ##-----here is just experiment using normal seq2seq not using aligments how will perform
    ## if using below to see result you need to make sure not to set output_alignments to True!
    
    #alignments_size = cell.state_size.alignments
    #outputs = layers.linear(outputs, alignments_size, scope="output_layer")
 
    #[batch_size, length, alignments_size]  [32, 11, 11]
    #this is ok but speed will hurt.. maybe change attention_wrapper to make output change..
    #so not use below but use modified attention_wrapper to output alignments directly !
    #if want to see how below work you need to set aligment_history=True
    
    #outputs = tf.transpose(state.alignment_history.stack(), [1, 0, 2])
  else:
    raise ValueError("Dynamic pointer decoder not support feed prev right now, TODO")
  
  return outputs, state
