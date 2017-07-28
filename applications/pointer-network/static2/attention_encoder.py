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


def attention_encoder(encoder_inputs, initial_state, 
                      cell,dtype=dtypes.float32, scope=None):
  with vs.variable_scope(scope or "attention_encoder"):
    #[batch_size, attn_length, num_units] - > [batch_size, attn_length, num_units] 
    states = [initial_state]
    outputs = []
    inps = []
    process_steps = len(encoder_inputs)
    for i in range(process_steps):
      #------------since tf version 1.2 do not need below, share by default
      #if i > 0:
      #  vs.get_variable_scope().reuse_variables()
      if i == 0:
        inp = encoder_inputs[i]
      else:
        inp = cell_output
      # Run the RNN.
      cell_output, new_state = cell(inp, states[-1])
      states.append(new_state)
      outputs.append(cell_output)

    return outputs, states
