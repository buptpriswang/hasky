﻿#!/usr/bin/env python
# ==============================================================================
#          \file   util.py
#        \author   chenghuige  
#          \date   2016-08-16 19:32:41.443712
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python import pywrap_tensorflow

import sys, os, glob, traceback
import inspect

import numpy as np

import gezi
import melt 

import melt.utils.logging as logging

#https://stackoverflow.com/questions/35164529/in-tensorflow-is-there-any-way-to-just-initialize-uninitialised-variables
def init_uninitialized_variables(sess, list_of_variables = None):
  if list_of_variables is None:
    list_of_variables = tf.global_variables()
  uninitialized_variables = list(tf.get_variable(name) for name in
                                 sess.run(tf.report_uninitialized_variables(list_of_variables)))
  uninitialized_variables = tf.group(uninitialized_variables, tf.local_variables_initializer())
  sess.run(tf.variables_initializer(uninitialized_variables))
  return unintialized_variables

def get_checkpoint_varnames(model_dir):
  checkpoint_path = get_model_path(model_dir)
  #if model_dir is dir then checkpoint_path should be model path not model dir like
  #/home/gezi/new/temp/image-caption/makeup/model/bow.inceptionResnetV2.finetune.later/model.ckpt-589.5-1011000
  #if user input model_dir is like above model path we assume it to be correct and exists not check!
  if os.path.isdir(checkpoint_path) or not os.path.exists(checkpoint_path + '.meta'):
    return None 
  try:
    reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
    var_to_shape_map = reader.get_variable_to_shape_map()
    varnames = [var_name for var_name in var_to_shape_map]
    return varnames
  except Exception:
    print(traceback.format_exc())
    return None

def varname_in_checkpoint(var_name, model_dir, mode='in'):
  varnames = get_checkpoint_varnames(model_dir)
  if not varnames:
    return False 
  else:
    varnames_exists = False
    for varname in varnames:
      if mode == 'in':
        if var_name in varname:
          varnames_exists = True
          break
      elif mode == 'startswith':
        if varname.startswith(var_name):
          varnames_exists = True 
          break 
    return varnames_exists

def try_add_to_collection(name, op):
  if not tf.get_collection(name):
    tf.add_to_collection(name, op)

def remove_from_collection(key):
  #must use ref get list and set to empty using [:] = [] or py3 can .clear 
  #https://stackoverflow.com/questions/850795/different-ways-of-clearing-lists
  l = tf.get_collection_ref(key)
  l[:] = []  

def rename_from_collection(key, to_key, index=0, scope=None):
  l = tf.get_collection_ref(key)
  if l:
    tf.add_to_collection(to_key, l[index])
    l[:] = []

#https://stackoverflow.com/questions/44251666/how-to-initialize-tensorflow-variable-that-wasnt-saved-other-than-with-tf-globa
def initialize_uninitialized_vars(sess):
  import itertools
  from itertools import compress
  global_vars = tf.global_variables()
  is_not_initialized = sess.run([~(tf.is_variable_initialized(var)) \
                                  for var in global_vars])
  not_initialized_vars = list(compress(global_vars, is_not_initialized))

  if len(not_initialized_vars):
    sess.run(tf.variables_initializer(not_initialized_vars))

#In [3]: tf.contrib.layers.OPTIMIZER_CLS_NAMES
#Out[3]: 
#{'Adagrad': tensorflow.python.training.adagrad.AdagradOptimizer,
# 'Adam': tensorflow.python.training.adam.AdamOptimizer,
# 'Ftrl': tensorflow.python.training.ftrl.FtrlOptimizer,
# 'Momentum': tensorflow.python.training.momentum.MomentumOptimizer,
# 'RMSProp': tensorflow.python.training.rmsprop.RMSPropOptimizer,
# 'SGD': tensorflow.python.training.gradient_descent.GradientDescentOptimizer}

optimizers = {
  'grad' : tf.train.GradientDescentOptimizer,
  'sgd' : tf.train.GradientDescentOptimizer,
  'adagrad' : tf.train.AdagradOptimizer
  }

def get_session(log_device_placement=False, allow_soft_placement=True, debug=False):
  """
  TODO FIXME get_session will casue  at last
#Exception UnboundLocalError: "local variable 'status' referenced before assignment" in <bound method Session.__del__ of <tensorflow.python.client.session.Session object at 0x858af10>> ignored
#TRACE: 03-17 08:22:26:   * 0 [clear]: tag init stat error

global or inside function global sess will cause this but not big problem for convenience just accpet right now
  """
  if not hasattr(get_session, 'sess') or get_session.sess is None:
    config=tf.ConfigProto(
      allow_soft_placement=allow_soft_placement, 
      log_device_placement=log_device_placement)
    #config.operation_timeout_in_ms=600000
    #NOTICE https://github.com/tensorflow/tensorflow/issues/2130 but 5000 will cause init problem!
    #config.operation_timeout_in_ms=50000   # terminate on long hangs
    #https://github.com/tensorflow/tensorflow/issues/2292 allow_soft_placement=True
    get_session.sess = tf.Session(config=config)
    if debug:
      from tensorflow.python import debug as tf_debug
      get_session.sess = tf_debug.LocalCLIDebugWrapperSession(get_session.sess)
  return get_session.sess

#def get_session(log_device_placement=False, allow_soft_placement=True, debug=False):
#  """
#  TODO FIXME get_session will casue  at last
##Exception UnboundLocalError: "local variable 'status' referenced before assignment" in <bound method Session.__del__ of <tensorflow.python.client.session.Session object at 0x858af10>> ignored
##TRACE: 03-17 08:22:26:   * 0 [clear]: tag init stat error

#global or inside function global sess will cause this but not big problem for convenience just accpet right now
#  """
#  if not hasattr(get_session, 'sess') or get_session.sess is None:
#    config=tf.ConfigProto(
#      allow_soft_placement=allow_soft_placement, 
#      log_device_placement=log_device_placement)
#    #config.operation_timeout_in_ms=600000
#    #NOTICE https://github.com/tensorflow/tensorflow/issues/2130 but 5000 will cause init problem!
#    #config.operation_timeout_in_ms=50000   # terminate on long hangs
#    sess = tf.Session(config=config)
#    if debug:
#      sess = tf_debug.LocalCLIDebugWrapperSession(sess)
#  return sess

def get_optimizer(name):
  if name in tf.contrib.layers.OPTIMIZER_CLS_NAMES:
    return tf.contrib.layers.OPTIMIZER_CLS_NAMES[name]
  else:
    return optimizers[name.lower()]

def gen_train_op(loss, learning_rate, optimizer=tf.train.AdagradOptimizer):
  train_op = optimizer(learning_rate).minimize(loss)  
  return train_op  

def gen_train_op_byname(loss, learning_rate, name='adagrad'):
  optimizer = optimizers.get(name.lower(), tf.train.AdagradOptimizer)
  train_op = optimizer(learning_rate).minimize(loss)  
  return train_op  

#TODO add name, notice if num_gpus=1 is same as num_gpus=0
#but for num_gpus=0 we will not consider multi gpu mode
#so num_gpus=1 will not use much, just for mlti gpu test purpose
#from https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10_multi_gpu_train.py def train()
def tower_losses(loss_function, num_gpus=1, name=''):
  tower_losses = []
  #with tf.variable_scope("OptimizeLoss"):
  #with tf.variable_scope(tf.get_variable_scope()):
  for i in range(num_gpus):
    with tf.device('/gpu:%d' % i):
      with tf.name_scope('%s_%d' % ('tower', i)) as scope:
          loss = loss_function()
          # Reuse variables for the next tower.
          tf.get_variable_scope().reuse_variables()
          tower_losses.append(loss)
  return tower_losses

def get_num_gpus():
  if 'CUDA_VISIBLE_DEVICES' in os.environ:
    num_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    logging.info('CUDA_VISIBLE_DEVICES is %s'%(os.environ['CUDA_VISIBLE_DEVICES']))
    return num_gpus

rnn_cells = {
  'lstm' : tf.contrib.rnn.LSTMCell, #LSTMCell is faster then BasicLSTMCell
  'gru' : tf.contrib.rnn.GRUCell,
  'lstm_block' : tf.contrib.rnn.LSTMBlockCell, #LSTMBlockCell is faster then LSTMCell
  }
#from https://github.com/tensorflow/models/blob/master/tutorials/rnn/ptb/ptb_word_lm.py lstm_cell()
def create_rnn_cell(num_units, is_training=True, initializer=None, forget_bias=1.0, num_layers=1, 
                    keep_prob=1.0, input_keep_prob=1.0, Cell=None, cell_type='lstm', scope=None):
  with tf.variable_scope(scope or 'create_rnn_cell') as scope:
    if Cell is None:
      Cell = rnn_cells.get(cell_type.lower(), tf.contrib.rnn.LSTMCell)
    # try:
    #   #cell = Cell(num_units, initializer=initializer, state_is_tuple=True)
    #   cell = Cell(num_units, initializer=initializer)
    # except Exception:
    #   #logging.warning('initializer not used as cell type not support, cell_type:%s'%cell_type)
    #   cell = Cell(num_units)

    # Slightly better results can be obtained with forget gate biases
    # initialized to 1 but the hyperparameters of the model would need to be
    # different than reported in the paper.
    def cell_():
      # With the latest TensorFlow source code (as of Mar 27, 2017),
      # the BasicLSTMCell will need a reuse parameter which is unfortunately not
      # defined in TensorFlow 1.0. To maintain backwards compatibility, we add
      # an argument check here:
      if 'reuse' in inspect.getargspec(
          Cell.__init__).args:
        return Cell(
            num_units, forget_bias=forget_bias,
            reuse=tf.get_variable_scope().reuse)
      else:
        return tf.contrib.rnn.BasicLSTMCell(
            num_units, forget_bias=forget_bias, state_is_tuple=True)

    attn_cell = cell_
    if is_training and (keep_prob < 1 or input_keep_prob < 1):
      def attn_cell():
        return tf.contrib.rnn.DropoutWrapper(
            cell_(), 
            input_keep_prob=input_keep_prob,
            output_keep_prob=keep_prob)

    if num_layers > 1:
      cell = tf.contrib.rnn.MultiRNNCell(
        [attn_cell() for _ in range(num_layers)], state_is_tuple=True)
    else:
      cell = attn_cell()
    #--now cell share graph by default so below is wrong.. will share cell for each layer
    ##cell = tf.contrib.rnn.MultiRNNCell([cell] * num_layers, state_is_tuple=True) 
    return cell

#-------for train flow
def show_precision_at_k(result, k=1):
  if len(result) == 1:
    accuracy = result
    print('precision@%d:'%k, '%.3f'%accuracy) 
  else:
    loss = result[0]
    accuracy = result[1]
    print('loss:', '%.3f'%loss, 'precision@%d:'%k, '%.3f'%accuracy)

def print_results(results, names=None):
  """
  standard result print
  """
  results = gezi.get_singles(results)
  if names is None:
    print(gezi.pretty_floats(results))
  else:
    if len(names) == len(results) - 1:
      names.insert(0, 'loss')
    if len(names) == len(results):
      print(gezi.get_value_name_list(results, names))
    else:
      print(gezi.pretty_floats(results))

def logging_results(results, names, tag=''):\
  logging.info('\t'.join(
    [tag] + ['%s:[%.4f]'%(name, result) for name, result in zip(names, results)]))
      
def parse_results(results, names=None):
  #only single values in results!
  if names is None:
    return gezi.pretty_floats(results)
  else:
    if len(names) == len(results) - 1:
      names.insert(0, 'loss')
    if len(names) == len(results):
      return gezi.get_value_name_list(results, names)
    else:
      return gezi.pretty_floats(results)

def value_name_list_str(results, names=None):
  if names is None:
    return gezi.pretty_floats(results)
  else:
    return gezi.get_value_name_list(results, names)

#-------model load
def get_model_path(model_dir, model_name=None):
  """
  if model_dir ok return latest model in this dir
  else return orginal model_dir as a model path
  NOTICE not check if this model path is ok(assume it to be ok) 
  """
  model_path = model_dir
  ckpt = tf.train.get_checkpoint_state(model_dir)
  if ckpt and ckpt.model_checkpoint_path:
    #@NOTICE below code will be ok int tf 0.10 but fail int 0.11.rc tensorflow ValueError: Restore called with invalid save path
    #do not use  ckpt.model_checkpoint_path for we might copy the model to other path so the absolute path(you might train using absoluate path) will not match
    #model_path = '%s/%s'%(model_dir, os.path.basename(ckpt.model_checkpoint_path))
    model_path = os.path.join(model_dir, os.path.basename(ckpt.model_checkpoint_path))
  else:
    model_path = model_dir if model_name is None else os.path.join(model_dir, model_name)
  #assert os.path.exists(model_path), model_path
  #tf.logging.log_if(tf.logging.WARN, '%s not exist'%model_path, not os.path.exists(model_path))
  #if not os.path.exists(model_path):
    #model_path = None 
    #tf.logging.WARN('%s not exist'%model_path)
    #raise FileNotFoundError(model_path)
    #raise ValueError(model_path)
  return model_path 

def latest_checkpoint(model_dir):
  return get_model_path(model_dir)

def get_model_dir_and_path(model_dir, model_name=None):
  model_path = model_dir
  ckpt = tf.train.get_checkpoint_state(model_dir)
  if ckpt and ckpt.model_checkpoint_path:
    #model_path = '%s/%s'%(model_dir, os.path.basename(ckpt.model_checkpoint_path)) 
    model_path = os.path.join(model_dir, os.path.basename(ckpt.model_checkpoint_path))
  else:
    model_path = model_dir if model_name is None else os.path.join(model_dir, model_name)
  if not os.path.exists(model_path):
    raise ValueError(model_path)
  return os.path.dirname(model_path), model_path

#cat checkpoint 
#model_checkpoint_path: "/home/gezi/temp/image-caption//model.flickr.show_and_tell/model.ckpt-256000"
#all_model_checkpoint_paths: "/home/gezi/temp/image-caption//model.flickr.show_and_tell/model.ckpt-252000"
#all_model_checkpoint_paths: "/home/gezi/temp/image-caption//model.flickr.show_and_tell/model.ckpt-253000"
#all_model_checkpoint_paths: "/home/gezi/temp/image-caption//model.flickr.show_and_tell/model.ckpt-254000"
#all_model_checkpoint_paths: "/home/gezi/temp/image-caption//model.flickr.show_and_tell/model.ckpt-255000"
#all_model_checkpoint_paths: "/home/gezi/temp/image-caption//model.flickr.show_and_tell/model.ckpt-256000"
def recent_checkpoint(model_dir, latest=False):
  index = -1 if latest else 1
  return open('%s/checkpoint'%(model_dir)).readlines()[index].split()[-1].strip('"')

def checkpoint_exists_in(model_dir):
  if not os.path.exists(model_dir):
    return False
  ckpt = tf.train.get_checkpoint_state(model_dir)
  if ckpt and ckpt.model_checkpoint_path:
    #input valid dir and return latest model
    return True
  elif os.path.isdir(model_dir):
    return False
  else:
    #this might be user specified model like ./model/model-100.ckpt
    #the file exists and we NOTICE we do not check if it is valid model file!
    return True

def get_model_step(model_path):
  return int(model_path.split('/')[-1].split('-')[-1]) 

def get_model_epoch(model_path):
  try:
    return float(model_path.split('/')[-1].split('-')[-2]) 
  except Exception:
    return None

def get_model_epoch_from_dir(model_dir):
  model_path = get_model_path(model_dir)
  try:
    return float(model_path.split('/')[-1].split('-')[-2]) 
  except Exception:
    return None

def get_model_step_from_dir(model_dir):
  model_path = get_model_path(model_dir)
  return int(model_path.split('/')[-1].split('-')[-1]) 

def save_model(sess, model_dir, step):
  checkpoint_path = os.path.join(model_dir, 'model.ckpt')
  tf.train.Saver().save(sess, checkpoint_path, global_step=step)

def restore(sess, model_dir, var_list=None, model_name=None):
  saver = tf.train.Saver(var_list)
  model_path = get_model_path(model_dir, model_name)
  saver.restore(sess, model_path)
  #@TODO still write to file ? using >
  print('restore ok:', model_path, file=sys.stderr)
  sess.run(tf.initialize_local_variables())
  return saver

def restore_from_path(sess, model_path, var_list=None):
  saver = tf.train.Saver(var_list)
  saver.restore(sess, model_path)
  print('restore ok:', model_path)
  sess.run(tf.local_variables_initializer())
  return saver

def restore_scope_from_path(sess, model_path, scope):
  variables = tf.get_collection(
    tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
  saver = tf.train.Saver(variables)
  saver.restore(sess, model_path)
  print('restore ok:', model_path)
  sess.run(tf.local_variables_initializer())
  return saver

def load(model_dir, model_name=None):
  """
  create sess and load from model,
  return sess
  use load for predictor, be sure to build all predict 
  related graph ready before calling melt.load
  """
  sess = get_session()
  restore(sess, model_dir, model_name)
  return sess

def load_from_path(model_path):
  """
  create sess and load from model,
  return sess
  use load for predictor, be sure to build all predict 
  related graph ready before calling melt.load
  """
  sess = get_session()
  restore_from_path(sess, model_path)
  return sess

def list_models(model_dir, time_descending=True):
  """
  list all models in model_dir
  """
  files = [file for file in glob.glob('%s/model.ckpt-*'%(model_dir)) if not file.endswith('.meta')]
  files.sort(key=lambda x: os.path.getmtime(x), reverse=time_descending)
  return files 

def variables_with_scope(scope):
    #scope is a top scope here, otherwise change startswith part
    return [v for v in tf.all_variables() if v.name.startswith(scope)]

import numpy 
#@TODO better
def npdtype2tfdtype(data_npy):
  if data_npy.dtype == numpy.float32:
    return tf.float32
  if data_npy.dtype == numpy.int32:
    return tf.int32
  if data_npy.dtype == numpy.int64:
    return tf.int64
  if data_npy.dtype == numpy.float64:
    return tf.float64
  return tf.float32

def load_constant(data_npy, sess=None, trainable=False, 
                  dtype=None, shape=None, name=None):
  """
  tf.constant only can be used for small data
  so melt.constant means melt.large_constant and have more general usage
  https://stackoverflow.com/questions/35687678/using-a-pre-trained-word-embedding-word2vec-or-glove-in-tensorflow
  """

  #or if isinstance(data_npy, str)
  if type(data_npy) is str:
    data_npy = np.load(data_npy)

  if dtype is None:
    dtype = npdtype2tfdtype(data_npy)
  #dtype = tf.float32
  if shape is None:
    shape = data_npy.shape
  
  if name is None:
    data_init = tf.placeholder(dtype, shape)
    #@TODO getvariable?
    data = tf.Variable(data_init, trainable=trainable, collections=[], name=name)
    if sess is None:
      sess = melt.get_session()
    sess.run(data.initializer, feed_dict={data_init: data_npy})
    return data
  else:
    data = tf.get_variable(name, shape=shape, initializer=tf.constant_initializer(data_npy), trainable=trainable)
  
  return data

def load_constant_cpu(data_npy, sess=None, trainable=False, 
                      dtype=None, shape=None, name=None):
   with tf.device('/CPU:0'):
    return load_constant(data_npy, 
        sess=sess, 
        trainable=trainable,
        dtype=dtype,
        shape=shape,
        name=name)

def reuse_variables():
  tf.get_variable_scope().reuse_variables()

#---now work.. can set attribute reuse
#def unreuse_variables():
#  tf.get_variable_scope().reuse=None

#------------------------------------tf record save @TODO move to tfrecords
def int_feature(value):
  if not isinstance(value, (list,tuple)):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def int64_feature(value):
  if not isinstance(value, (list,tuple)):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def bytes_feature(value):
  if not isinstance(value, (list,tuple)):
    value = [value]
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def float_feature(value):
  if not isinstance(value, (list,tuple)):
    value = [value]
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

features = lambda d: tf.train.Features(feature=d)

# Helpers for creating SequenceExample objects  copy from \tensorflow\python\kernel_tests\parsing_ops_test.py
feature_list = lambda l: tf.train.FeatureList(feature=l)
feature_lists = lambda d: tf.train.FeatureLists(feature_list=d)


def int64_feature_list(values):
  """Wrapper for inserting an int64 FeatureList into a SequenceExample proto."""
  return tf.train.FeatureList(feature=[int64_feature(v) for v in values])

def bytes_feature_list(values):
  """Wrapper for inserting a bytes FeatureList into a SequenceExample proto."""
  return tf.train.FeatureList(feature=[bytes_feature(v) for v in values])

def float_feature_list(values):
  """Wrapper for inserting a bytes FeatureList into a SequenceExample proto."""
  return tf.train.FeatureList(feature=[float_feature(v) for v in values])

def get_num_records_single(tf_record_file):
  return len([x for x in tf.python_io.tf_record_iterator(tf_record_file)])

def get_num_records(files):
  if isinstance(files, str):
    files = gezi.list_files(files) 
  return sum([get_num_records_single(file) for file in files])

def get_num_records_print(files):
  num_records = 0
  if isinstance(files, str):
    files = gezi.list_files(files) 
  num_inputs = len(files)
  index = 0
  for file in files:
    count = get_num_records_single(file)
    print(file, count,  '%.3f'%(index / num_inputs))
    num_records += count
    index += 1
  print('num_records:', num_records)
  return num_records

def load_num_records(input):
  num_records_file = os.path.dirname(input) + '/num_records.txt'
  num_records = int(open(num_records_file).read()) if os.path.isfile(num_records_file) else 0
  return num_records

def get_num_records():
  import dconf  
  return dconf.NUM_RECORDS 

def get_num_steps_per_epoch(batch_size):
  #need dconf.py with NUM_RECORDS setting 0 at first
  import dconf  
  num_records = dconf.NUM_RECORDS 
  return num_records // batch_size 

#-------------histogram util 
def monitor_train_vars(collections=None):
  for var in tf.trainable_variables():
    tf.histogram_summary(var.op.name, var, collections=collections)

class MonitorKeys():
  TRAIN = 'train_monitor'

#@FIXME seems not work get_collection always None
from tensorflow.python.framework import ops
def monitor_gradients_from_loss(loss, collections=[MonitorKeys.TRAIN]):
  grads = tf.gradients(loss, tf.trainable_variables())
  for grad in grads:
    if grad is not None:
      tf.histogram_summary(grad.op.name, grad, collections=collections)
    else:
      raise ValueError('None grad')

#TODO check op.name or .name ? diff?
def histogram_summary(name, tensor):
  tf.summary.histogram('{}_{}'.format(name, tensor.op.name), tensor)

def scalar_summary(name, tensor):
  tf.summary.scalar('{}_{}'.format(name, tensor.op.name), tensor)

def monitor_embedding(emb, vocab, vocab_size):
  histogram_summary('emb_0', tf.gather(emb, 0))
  histogram_summary('emb_1', tf.gather(emb, 1))
  histogram_summary('emb_2', tf.gather(emb, 2))
  histogram_summary('emb_1/4', tf.gather(emb, vocab_size // 4))
  histogram_summary('emb_middle', tf.gather(emb, vocab_size // 2))
  histogram_summary('emb_3/4', tf.gather(emb, vocab_size // 4 * 3))
  histogram_summary('emb_end', tf.gather(emb, vocab_size - 1))
  histogram_summary('emb_end2', tf.gather(emb, vocab_size - 2))
  histogram_summary('emb_start_id', tf.gather(emb, vocab.start_id()))
  histogram_summary('emb_end_id', tf.gather(emb, vocab.end_id()))
  histogram_summary('emb_unk_id', tf.gather(emb, vocab.unk_id()))

def visualize_embedding(emb, vocab_txt):
  # You can add multiple embeddings. Here we add only one.
  embedding = melt.flow.projector_config.embeddings.add()
  embedding.tensor_name = emb.name
  # Link this tensor to its metadata file (e.g. labels).
  if vocab_txt.endswith('.bin'):
    embedding.metadata_path = vocab_txt.replace('.bin', '.project')
  elif vocab_txt.endswith('.txt'):
    embedding.metadata_path = vocab_txt.replace('.txt', '.project')
  else:
    embedding.metadata_path = vocab_txt[:vocab_txt.rindex('.')] + '.project'

def get_summary_ops():
  return ops.get_collection(ops.GraphKeys.SUMMARIES)

def print_summary_ops():
  print('summary ops:')
  sops = ops.get_collection(ops.GraphKeys.SUMMARIES)
  for sop in sops:
    print(sop)

def print_global_varaiables(sope=None):
  for item in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
    print(item)

def print_varaiables(key, sope=None):
  for item in tf.get_collection(key):
    print(item)

def get_global_int(key):
  if key not in os.environ:
    os.environ[key] = '0'
  return int(os.environ[key])

def get_global_float(key):
  if key not in os.environ:
    os.environ[key] = '0'
  return float(os.environ[key])

def get_global_str(key):
  if key not in os.environ:
    os.environ[key] = ''
  return os.environ[key]

def step():
  return get_global_int('step')

def epoch():
  return get_global_float('epoch')

def batch_size():
  return get_global_int('batch_size')

def num_gpus():
  return get_global_int('num_gpus')

def loss():
  loss_ = get_global_str('eval_loss')
  if not loss_:
    loss_ = get_global_str('train_loss')
  if not loss_:
    loss_ = get_global_str('loss')
  return loss_

def train_loss():
  return get_global_str('train_loss')

def eval_loss():
  return get_global_str('eval_loss')

def duration():
  return get_global_float('duration')

def set_global(key, value):
  os.environ[key] = str(value)

#def step():
#  return melt.flow.global_step

#def epoch():
#  return melt.flow.global_epoch

#---------for flow
def default_names(length):
  names = ['metric%d'%(i - 1) for i in xrange(length)]
  names[0] = 'loss'
  return names 

def adjust_names(values, names):
  if names is None:
    return default_names(len(values))
  else:
    if len(names) == len(values):
      return names
    elif len(names) + 1 == len(values):
      names.insert(0, 'loss')
      return names
    else:
      return default_names(len(values))


def add_summarys(summary, values, names, suffix='', prefix=''):
  for value, name in zip(values, names):
    if suffix:
      summary.value.add(tag='%s_%s'%(name, suffix), simple_value=float(value))
    else:
      if prefix:
        summary.value.add(tag='%s_%s'%(prefix, name), simple_value=float(value))
      else:
        summary.value.add(tag=name, simple_value=float(value))

#-----------deal with text  TODO move 
import melt
def pad(text, start_id=None, end_id=None):
  print('Pad with start_id', start_id, ' end_id', end_id)
  need_start_mark = start_id is not None
  need_end_mark = end_id is not None
  if not need_start_mark and not need_end_mark:
    return text, melt.length(text) 
  
  batch_size = tf.shape(text)[0]
  zero_pad = tf.zeros([batch_size, 1], dtype=text.dtype)

  sequence_length = melt.length(text)

  if not need_start_mark:
   text = tf.concat([text, zero_pad], 1)
  else:
    if need_start_mark:
      start_pad = zero_pad + start_id
      if need_end_mark:
        text = tf.concat([start_pad, text, zero_pad], 1)
      else:
        text = tf.concat([start_pad, text], 1)
      sequence_length += 1

  if need_end_mark:
    text = melt.dynamic_append_with_length(
        text, 
        sequence_length, 
        tf.constant(end_id, dtype=text.dtype)) 
    sequence_length += 1

  return text, sequence_length
