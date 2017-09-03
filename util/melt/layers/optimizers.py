# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.summary import summary
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import variables as vars_
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import optimizer as optimizer_
from tensorflow.python.training import training as train


import tensorflow as tf

"""
copy from tensorflow.contrib.layers.python.layers.optimerzers.py version 0.10
"""

"""Optimizer ops for use in layers and tf.learn."""

OPTIMIZER_CLS_NAMES = {
    "Adagrad": train.AdagradOptimizer,
    "Adam": train.AdamOptimizer,
    "Ftrl": train.FtrlOptimizer,
    "Momentum": train.MomentumOptimizer,
    "RMSProp": train.RMSPropOptimizer,
    "SGD": train.GradientDescentOptimizer,
}

OPTIMIZER_SUMMARIES = ["learning_rate",
    "loss",
    "gradients",
    "gradient_norm",]

# from cifar10_multi_gpu_train.py
def average_gradients(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.

  Note that this function provides a synchronization point across all towers.

  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(grads, 0)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads

def optimize_loss(losses,
                  global_step,
                  learning_rate,
                  optimizer, 
                  num_gpus=1,
                  gradient_noise_scale=None,
                  gradient_multipliers=None,
                  clip_gradients=None,
                  learning_rate_decay_fn=None,
                  update_ops=None,
                  variables=None,
                  name=None,
                  summaries=None):
  """Given loss and parameters for optimizer, returns a training op.
  Args:
    loss: Tensor, 0 dimensional.
    global_step: Tensor, step counter for each update.
    learning_rate: float or Tensor, magnitude of update per each training step.
    optimizer: string, class or optimizer instance, used as trainer.
               string should be name of optimizer, like 'SGD',
                 'Adam', 'Adagrad'. Full list in OPTIMIZER_CLS_NAMES constant.
               class should be sub-class of tf.Optimizer that implements
                 `compute_gradients` and `apply_gradients` functions.
               optimizer instance should be instantion of tf.Optimizer sub-class
                 and have `compute_gradients` and `apply_gradients` functions.
    gradient_noise_scale: float or None, adds 0-mean normal noise scaled by this
                          value.
    gradient_multipliers: dict of variables or variable names to floats.
                          If present, gradients for specified
                          variables will be multiplied by given constant.
    clip_gradients: float or `None`, clips gradients by this value.
    moving_average_decay: Deprecated. float or None, takes into account previous
                          loss to make learning smoother due to outliers.
    learning_rate_decay_fn: function, takes `learning_rate` and `global_step`
                            `Tensor`s, returns `Tensor`.
                            Can be used to implement any learning rate decay
                            functions.
                            For example: tf.train.exponential_decay.
    update_ops: list of update `Operation`s to execute at each step. If `None`,
                uses elements of UPDATE_OPS collection.
    variables: list of variables to optimize or
               `None` to use all trainable variables.
    name: The name for this operation is used to scope operations and summaries.
    summaries: List of internal quantities to visualize on tensorboard. If not
               set only the loss and the learning rate will be reported. The
               complete list is in OPTIMIZER_SUMMARIES.
  Returns:
    Training op.
  Raises:
    ValueError: if optimizer is wrong type.
  """
  with vs.variable_scope(name, "OptimizeLoss", losses + [global_step]):
    # # Update ops take UPDATE_OPS collection if not provided.
    # if update_ops is None:
    #   update_ops = set(ops.get_collection(ops.GraphKeys.UPDATE_OPS))
    # # Make sure update ops are ran before computing loss.
    # if update_ops:
    #   #loss = control_flow_ops.with_dependencies(list(update_ops), loss)
    #   raise ValueError('update ops not supported yet for multi gpu')

    # Learning rate variable, with possible decay.
    if (isinstance(learning_rate, ops.Tensor) and learning_rate.get_shape().ndims == 0):
      lr = learning_rate
    elif isinstance(learning_rate, float):
      lr = vs.get_variable("learning_rate", [], trainable=False,
          initializer=init_ops.constant_initializer(learning_rate))
    else:
      raise ValueError("Learning rate should be 0d Tensor or float. "
                       "Got %s of type %s" % (str(learning_rate), str(type(learning_rate))))
    if summaries is None:
      summaries = ["loss", "learning_rate"]
    if learning_rate_decay_fn is not None:
      lr = learning_rate_decay_fn(lr, global_step)
      if "learning_rate" in summaries:
        summary.scalar("learning_rate", lr)

    # Create optimizer, given specified parameters.
    if isinstance(optimizer, six.string_types):
      if optimizer not in OPTIMIZER_CLS_NAMES:
        raise ValueError("Optimizer name should be one of [%s], you provided %s." % (", ".join(OPTIMIZER_CLS_NAMES), optimizer))
      opt = OPTIMIZER_CLS_NAMES[optimizer](learning_rate=lr)
    elif isinstance(optimizer, type) and issubclass(optimizer,
                                                    optimizer_.Optimizer):
      opt = optimizer(learning_rate=lr)
    elif isinstance(optimizer, optimizer_.Optimizer):
      opt = optimizer
    else:
      raise ValueError("Unrecognized optimizer: should be string, "
                       "subclass of Optimizer or instance of "
                       "subclass of Optimizer. Got %s." % str(optimizer))


    # Calculate the gradients for each model tower.
    tower_grads = []
    for i in range(num_gpus):
      with tf.device('/gpu:%d' % i):
        with tf.name_scope('%s_%d' % ('tower', i)) as scope:
          # All trainable variables, if specific variables are not specified.
          
          #if variables is None:
          #  variables = vars_.trainable_variables()
          # Compute gradients.
          loss = losses[i]
          #print('------------',)
          #gradients = opt.compute_gradients(loss, variables)
          gradients = opt.compute_gradients(loss)
          #TODO FIXME might have None for example add another predictor to graph 
          #[(None, <tf.Variable 'dual_bow/model_init/emb:0' shape=(29285, 256) dtype=float32_ref>), 
          #(None, <tf.Variable 'dual_bow/main/dual_textsim/encode/text_mlp/linear/weights:0' shape=(256, 256) dtype=float32_ref>),
          #(<tensorflow.python.framework.ops.IndexedSlices object at 0x1b72ff50>, <tf.Variable 'seq2seq/model_init/emb:0' shape=(29285, 256) dtype=float32_ref>)
          #print('-------gradients1', gradients)
          #--now hack use below, TODO why dual_bow.. in introduced when compute gradient of loss as seem not related my seq2seq loss?
          gradients = [x for x in gradients if x[0] is not None]
          # Optionally add gradient noise.
          if gradient_noise_scale is not None:
            gradients = _add_scaled_noise_to_gradients(gradients, gradient_noise_scale)
          # Multiply some gradients.
          if gradient_multipliers is not None:
            gradients = _multiply_gradients(gradients, gradient_multipliers)
          # Optionally clip gradients by global norm.
          if clip_gradients is not None:
            gradients = _clip_gradients_by_norm(gradients, clip_gradients)
          
          #print('-------gradients', gradients)
          tower_grads.append(gradients)
      
    # Add scalar summary for loss.
    if "loss" in summaries:
      summary.scalar("loss", loss)

    #@TODO chg now just remove below  TODO FIXME add gradient monitor
    ## Add histograms for variables, gradients and gradient norms.
    #for gradient, variable in gradients:
    #  if isinstance(gradient, ops.IndexedSlices):
    #    grad_values = gradient.values
    #  else:
    #    grad_values = gradient

    #  if grad_values is not None:
    #    if "gradients" in summaries:
    #      logging_ops.histogram_summary(variable.name + "/gradients",
    #                                    grad_values)
    #    if "gradient_norm" in summaries:
    #      logging_ops.histogram_summary(variable.name + "/gradient_norm",
    #                                    clip_ops.global_norm([grad_values]))

    #if FLAGS.monitor_level > 1 and FLAGS.num_gpus == 0:
    #  melt.monitor_gradients_from_loss(loss)

    gradients = average_gradients(tower_grads)

    # Create gradient updates.
    grad_updates = opt.apply_gradients(gradients,
                                       global_step=global_step,
                                       name="train")
    # # Make sure total_loss is valid.
    # final_loss = array_ops.check_numerics(loss, "Loss is inf or nan")

    # # Ensure the train_tensor computes grad_updates.
    # train_tensor = control_flow_ops.with_dependencies([grad_updates], final_loss)

    #return train_tensor
    return grad_updates


def _clip_gradients_by_norm(grads_and_vars, clip_gradients):
  """Clips gradients by global norm."""
  gradients, variables = zip(*grads_and_vars)
  clipped_gradients, _ = clip_ops.clip_by_global_norm(gradients,
                                                      clip_gradients)
  return list(zip(clipped_gradients, variables))


def _add_scaled_noise_to_gradients(grads_and_vars, gradient_noise_scale):
  """Adds scaled noise from a 0-mean normal distribution to gradients."""
  gradients, variables = zip(*grads_and_vars)
  noisy_gradients = []
  for gradient in gradients:
    if gradient is None:
      noisy_gradients.append(None)
      continue
    if isinstance(gradient, ops.IndexedSlices):
      gradient_shape = gradient.dense_shape
    else:
      gradient_shape = gradient.get_shape()
    noise = random_ops.truncated_normal(gradient_shape) * gradient_noise_scale
    noisy_gradients.append(gradient + noise)
  return list(zip(noisy_gradients, variables))


def _multiply_gradients(grads_and_vars, gradient_multipliers):
  """Multiply specified gradients."""
  multiplied_grads_and_vars = []
  for grad, var in grads_and_vars:
    if (grad is not None and (var in gradient_multipliers or var.name in gradient_multipliers)):
      key = var if var in gradient_multipliers else var.name
      multiplier = constant_op.constant(gradient_multipliers[key], dtype=dtypes.float32)
      if isinstance(grad, ops.IndexedSlices):
        grad_values = grad.values * multiplier
        grad = ops.IndexedSlices(grad_values, grad.indices, grad.dense_shape)
      else:
        grad *= multiplier
    multiplied_grads_and_vars.append((grad, var))
  return multiplied_grads_and_vars
