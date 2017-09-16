#!/usr/bin/env python
# ==============================================================================
#          \file   train.py
#        \author   chenghuige  
#          \date   2016-08-17 10:30:20.286494
#   \Description  
# ==============================================================================

"""
not supporting averaging and multi gpu yet  @TODO
 [`tf.moving_average_variables()`](../../api_docs/python/state_ops.md#moving_average_variables)

 here what we do is 
 create train_op from loss
 and may be using multi gpu deal
"""
  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('num_epochs', 0, 'Number of epochs to run trainer.')
flags.DEFINE_integer('num_steps', 0, 'Number of steps to run trainer. 0 means run forever, -1 means you just want to build graph and save without training(chaning model value)')
#-------model
flags.DEFINE_boolean('save_model', True, '')
flags.DEFINE_float('save_interval_epochs', 1, 'if 0 will not save, by default 1 epoch 1 model in modeldir/epoch, you can change to 2, 0.1 etc')
flags.DEFINE_float('save_interval_seconds', 0, 'model/checkpoint save interval by n seconds, if > 0 will use this other wise use save_interval_hours')
flags.DEFINE_float('save_interval_hours', 24, """model/checkpoint save interval by n hours""")
flags.DEFINE_float('save_interval_steps', 1000, 'model/checkpoint save interval steps')
flags.DEFINE_integer('max_models_keep', 2, 'keep recent n models, default 2 for safe')
flags.DEFINE_boolean('restore_from_latest', True, 'more safe to restore from recent but not latest')

#--------show
flags.DEFINE_integer('interval_steps', 100, '')
flags.DEFINE_integer('eval_interval_steps', 1000, """for training suggest 10000, 
                                                     you can check evaluate interval time 
                                                     and evaluate once time the ratio below 0.1""")
flags.DEFINE_integer('metric_eval_interval_steps', 0, 'if > 0 need to be eval_interval_steps * n')
flags.DEFINE_boolean('metric_eval', True, '')

#----------optimize
flags.DEFINE_string('optimizer', 'adagrad', 'https://www.quora.com/Why-is-AdaDelta-not-favored-in-Deep-Learning-communities-while-AdaGrad-is-preferred-by-many-over-other-SGD-variants')
flags.DEFINE_float('learning_rate', 0.1, """Initial learning rate. for adgrad especially, 
                                            notice keras set for adgrad 0.01 
                                            but seems bad perf hard to converge for some seq2seq/lstm training
                                            see textsum/train/shangpinming/seq2seq-gen-copy-switch.sh""")
flags.DEFINE_float('learning_rate_decay_factor', 0, '')
flags.DEFINE_integer('num_epochs_per_decay', 1, '')
flags.DEFINE_float('clip_gradients', 5.0, """follow im2text as 5.0 default, 
                                          set to 1.0 in deeipiu/image_caption try sh ./train/flickr-rnn.sh, 
                                          will show perf from 900inst/s to 870ints/s and also slow convergence""")
flags.DEFINE_boolean('optimize_has_scope', True, '')

#----------train
flags.DEFINE_boolean('train_only', False, '')
flags.DEFINE_string('work_mode', 'full', 'full/train_valid_show_metric, train, test, train_metric, train_valid, train_valid_metric')
flags.DEFINE_integer('monitor_level', 2, '1 will monitor emb, 2 will monitor gradient')
flags.DEFINE_boolean('no_log', False, '')
flags.DEFINE_string('mode', 'train', 'or predict')

flags.DEFINE_boolean('use_tower_loss', True, '')
#----------multi gpu
flags.DEFINE_integer('num_gpus', 0, """How many GPUs to use. set 0 to disable multi gpu mode""")
tf.app.flags.DEFINE_boolean('log_device_placement', False, """Whether to log device placement.""")

#----------scope
flags.DEFINE_boolean('add_global_scope', True, '''default will add global scope as algo name,
                      set to False incase you want to load some old model without algo scope''')
flags.DEFINE_string('global_scope', '', '')
flags.DEFINE_string('main_scope', 'main', 'or use other main_scope like run, this is mainly graph scope for varaible reuse')

from tensorflow.python import debug as tf_debug

__pacage__ = None 
from six.moves import xrange  # pylint: disable=redefined-builti
import os

import melt 

#or from melt.utils import logging
import melt.utils.logging as logging
#import logging

def gen_learning_rate():
  #TODO if app crash reload then we should set smaller learning rate, may adgrad can combine with exponential_decay ?
  #copy from im2txt\im2txt\train.py
  learning_rate_decay_fn = None
  learning_rate = tf.constant(FLAGS.learning_rate)

  if FLAGS.learning_rate_decay_factor > 0:
    num_batches_per_epoch = num_steps_per_epoch
    decay_steps = int(num_batches_per_epoch * FLAGS.num_epochs_per_decay)
    
    def _learning_rate_decay_fn(learning_rate, global_step):
      return tf.train.exponential_decay(
          learning_rate,
          global_step,
          decay_steps=decay_steps,
          decay_rate=FLAGS.learning_rate_decay_factor,
          staircase=True)

      learning_rate_decay_fn = _learning_rate_decay_fn
  return learning_rate, learning_rate_decay_fn

def train_flow(ops, 
               names=None, 
               gen_feed_dict_fn=None, 
               deal_results_fn=melt.print_results, 
               eval_ops=None, 
               eval_names=None,
               gen_eval_feed_dict_fn=None, 
               deal_eval_results_fn=melt.print_results,
               optimizer=None, 
               learning_rate=0.1, 
               num_steps_per_epoch=None,
               model_dir=None, 
               metric_eval_fn=None, 
               debug=False,
               summary_excls=None,
               init_fn=None,
               restore_fn=None,
               restore_scope=None,
               save_all_scope=False,
               variables_to_restore=None,  #variables_to_restore = slim.get_variables_to_restore(exclude=['fc6', 'fc7', 'fc8'])
               variables_to_save=None,
               sess=None):

  if sess is None:
    sess = melt.get_session()
  if debug:
    sess = tf_debug.LocalCLIDebugWrapperSession(sess)

  logging.info('learning_rate:{}'.format(FLAGS.learning_rate))
  logging.info('clip_gradients:{}'.format(FLAGS.clip_gradients))
  logging.info('optimizer:{}'.format(FLAGS.optimizer))
  
  #batch size right now not define here, but in app code like input_app.py
  melt.set_global('batch_size', FLAGS.batch_size)
  num_gpus = FLAGS.num_gpus
  melt.set_global('num_gpus', max(num_gpus, 1))

  #NOTICE since melt.__init__.py with from melt.flow import * then you can not 
  #use melt.flow.train.train_flow but you can always use
  #from melt.flow.train.train_flow import train_flow

  if optimizer is None:
    optimizer = FLAGS.optimizer
  # Set up the training ops.
  #notice '' only works in tf >= 0.11, for 0.10 will always add OptimeizeLoss scope
  #the diff is 0.10 use variable_op_scope and 0.11 use variable_scope
  optimize_scope = None if FLAGS.optimize_has_scope else ''
  #or judge by FLAGS.num_gpus
  if not isinstance(ops[0], (list,tuple)):
    learning_rate, learning_rate_decay_fn = gen_learning_rate()
    train_op = tf.contrib.layers.optimize_loss(
        loss=ops[0],
        global_step=None,
        learning_rate=learning_rate,
        optimizer=melt.util.get_optimizer(optimizer),
        clip_gradients=FLAGS.clip_gradients,
        learning_rate_decay_fn=learning_rate_decay_fn,
        name=optimize_scope)  
  else: 
    #---as in cifa10 example, put all but tower loss on cpu, wiki say, that will be faster,
    #but here I find without setting to cpu will be faster..
    #https://github.com/tensorflow/tensorflow/issues/4881
    #I've noticed same thing on cirrascale GPU machines - putting parameters on gpu:0 and using gpu->gpu transfer was a bit faster. I suppose this depends on particular details of hardware -- if you don't have p2p connectivity between your video cards then keeping parameters on CPU:0 gives faster training.
    #err but for my pc no p2p, with PHB connection nvidia-smi topo -m, still hurt by set cpu.. may be should not put cpu here
    #with tf.device('/cpu:0'):
    learning_rate, learning_rate_decay_fn = gen_learning_rate()
    train_op = melt.layers.optimize_loss(
        losses=ops[0],
        num_gpus=num_gpus,
        global_step=None,
        learning_rate=learning_rate,
        optimizer=melt.util.get_optimizer(optimizer),
        clip_gradients=FLAGS.clip_gradients,
        learning_rate_decay_fn=learning_rate_decay_fn,
        name=optimize_scope)
    #set the last tower loss as loss in ops
    ops[0] = ops[0][-1]
 
  ops.insert(0, train_op)
    
  #-----------post deal
  save_interval_seconds = FLAGS.save_interval_seconds if FLAGS.save_interval_seconds > 0 \
     else FLAGS.save_interval_hours * 3600 

  interval_steps=FLAGS.interval_steps
  eval_interval_steps=FLAGS.eval_interval_steps
  metric_eval_interval_steps=FLAGS.metric_eval_interval_steps
  save_model=FLAGS.save_model 
  save_interval_steps = FLAGS.save_interval_steps 
  if not save_interval_steps:
    save_interval_steps = 1000000000000

  if FLAGS.work_mode == 'train':
    eval_ops = None 
    metric_eval_fn = None
    logging.info('running train only mode')
  elif FLAGS.work_mode == 'train_metric':
    eval_ops = None 
    assert metric_eval_fn is not None, 'set metric_eval to 1'
    logging.info('running train+metric mode')
  elif FLAGS.work_mode == 'train_valid':
    metric_eval_fn = None
    logging.info('running train+valid mode')
  elif FLAGS.work_mode == 'test':
    ops = None
    logging.info('running test only mode')
    interval_steps = 0
    eval_interval_steps = 1
    metric_eval_interval_steps /= FLAGS.eval_interval_steps
    save_model = False

  return melt.flow.train_flow(
             ops, 
             names=names,
             gen_feed_dict_fn=gen_feed_dict_fn,
             deal_results_fn=deal_results_fn,
             eval_ops=eval_ops,
             eval_names=eval_names,
             gen_eval_feed_dict_fn=gen_eval_feed_dict_fn,
             deal_eval_results_fn=deal_eval_results_fn,
             interval_steps=interval_steps,
             eval_interval_steps=eval_interval_steps,
             num_epochs=FLAGS.num_epochs,
             num_steps=FLAGS.num_steps,
             save_interval_seconds=save_interval_seconds,
             save_interval_steps=save_interval_steps,
             save_model=save_model,
             save_interval_epochs=FLAGS.save_interval_epochs,
             #optimizer=optimizer, 
             optimizer=None, #must set None since here we have done choosing optimizer
             learning_rate=learning_rate,
             num_steps_per_epoch=num_steps_per_epoch,
             max_models_keep=FLAGS.max_models_keep,
             model_dir=model_dir,
             restore_from_latest=FLAGS.restore_from_latest,
             metric_eval_fn=metric_eval_fn,
             metric_eval_interval_steps=metric_eval_interval_steps,
             no_log=FLAGS.no_log,
             summary_excls=summary_excls,
             init_fn=init_fn,
             restore_fn=restore_fn,
             restore_scope=restore_scope,
             save_all_scope=save_all_scope,
             variables_to_restore=variables_to_restore,
             variables_to_save=variables_to_save,
             sess=sess)
