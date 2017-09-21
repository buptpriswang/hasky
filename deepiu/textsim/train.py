#!/usr/bin/env python
# -*- coding: gbk -*-
# ==============================================================================
#          \file   train.py
#        \author   chenghuige  
#          \date   2016-08-16 14:05:38.743467
#   \Description  
# ==============================================================================
"""
@TODO using logging ?
NOTICE using feed_dict --feed_neg=1 will slow *10
@TODO why evaluate will use mem up to 13% then down to 5%?
using tf.nn.top_k do sort in graph still evaluate large mem same as before
20w will using 1.6G 32 * 5%
50w 9.1%
200w 16G?
2000w 160g?

@TODO why using gpu will use more cpu mem?
50w keyword MAX_TEXT_WORDS 20
cpu version: 
show_eval=0  train 6.7% eval 11.4%
show_eval=1 train 8% eval 20%

gpu version: 
show_eval=0 train 8% eval 12.8%
show_eval=1 train 35% eval 39%

text means text ids
text_str means orit text str
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('model_dir', '/home/gezi/temp/textsum/model', '')

flags.DEFINE_string('algo', 'dual_bow', 'default algo is bow(cbow), also support rnn, show_and_tell, TODO cnn')

flags.DEFINE_string('vocab', '/home/gezi/new/temp/makeupe/title2name/tfrecord/seq-basic.10w/train/vocab.txt', 'vocabulary file')

flags.DEFINE_boolean('debug', False, '')

import sys, math
import functools
import gezi
import melt
logging = melt.logging

from deepiu.textsim import input_app as InputApp

#TODO maybe not use image_caption eval show
from deepiu.image_caption import eval_show
from deepiu.util import evaluator
from deepiu.util import algos_factory

#debug
from deepiu.util import vocabulary
from deepiu.util import text2ids

sess = None

#TODO do not consider feed dict support right now for tower loss!
def tower_loss(trainer, input_app=None, input_results=None):
  if input_app is None:
    input_app = InputApp.InputApp()
  if input_results is None:
    input_results = input_app.gen_input(train_only=True)

  #--------train
  ltext_str, ltext, rtext, rtext_str = input_results[input_app.input_train_name]
  
  if input_results[input_app.input_train_neg_name]:
    neg_ltext_str, neg_ltext, neg_rtext, neg_rtext_str = input_results[input_app.input_train_neg_name]

  if not FLAGS.neg_left:
    neg_ltext = None

  if not FLAGS.neg_right:
    neg_rtext = None

  loss = trainer.build_train_graph(ltext, rtext, neg_ltext, neg_rtext)
  return loss


def gen_train_graph(input_app, input_results, trainer):
  """
    main flow, key graph
  """
  #--- if you don't want to use mutli gpu, here just for safe(code same with old single gpu cod)
  if FLAGS.num_gpus == 0:
    loss = tower_loss(trainer, input_app, input_results)
  else:
    loss_function = lambda: tower_loss(trainer)
    #here loss is a list of losses
    loss = melt.tower_losses(loss_function, FLAGS.num_gpus)
    print('num tower losses:', len(loss))

  ops = [loss]
    
  deal_debug_results = None

  if FLAGS.debug == True:
    ops += [tf.get_collection('scores')[-1]]
    def _deal_debug_results(results):
      print(results)  

    deal_debug_results = _deal_debug_results

  return ops, deal_debug_results

def gen_train(input_app, input_results, trainer):
  ops, deal_debug_results = gen_train_graph(input_app, input_results, trainer)
  
  def _deal_results(results):
    melt.print_results(results, ['batch_loss'])

    if deal_debug_results is not None:
      debug_results = results
      deal_debug_results(debug_results)

  deal_results = _deal_results

  return ops, None, deal_results

def gen_evalulate(input_app, 
                  input_results,
                  predictor, 
                  eval_ops, 
                  eval_scores,
                  eval_neg_text=None,
                  eval_neg_text_str=None):
  assert algos_factory.is_discriminant(FLAGS.algo)
  eval_ops += eval_show.gen_eval_show_ops(
      input_app, 
      input_results, 
      predictor, 
      eval_scores, 
      eval_neg_text, 
      eval_neg_text_str)
  deal_eval_results = eval_show.deal_eval_results
  return eval_ops, deal_eval_results

def gen_validate(input_app, input_results, trainer, predictor):
  eval_ops = None
  train_with_validation = input_results[input_app.input_valid_name] is not None
  deal_eval_results = None
  if train_with_validation and not FLAGS.train_only:
    eval_ltext_str, eval_ltext, eval_rtext, eval_rtext_str = input_results[input_app.input_valid_name]
    if input_results[input_app.input_valid_neg_name]:
      eval_neg_ltext_str, eval_neg_ltext, eval_neg_rtext, eval_neg_rtext_str = input_results[input_app.input_valid_neg_name]
      if not FLAGS.neg_left:
        eval_neg_ltext = None
      eval_neg_rtext_ = eval_neg_rtext
      if not FLAGS.neg_right:
        eval_neg_rtext_ = None
    eval_loss = trainer.build_train_graph(eval_ltext, eval_rtext, eval_neg_ltext, eval_neg_rtext_)
    eval_scores = tf.get_collection('scores')[-1]
    eval_ops = [eval_loss]

    if FLAGS.show_eval and (predictor is not None):
      eval_ops, deal_eval_results = \
        gen_evalulate(
            input_app, 
            input_results, 
            predictor, 
            eval_ops, 
            eval_scores,
            eval_neg_rtext,
            eval_neg_rtext_str)
    else:
      deal_eval_results = lambda x: melt.print_results(x, ['eval_batch_loss'])

  return eval_ops, None, deal_eval_results

def gen_predict_graph(predictor): 
  predictor.init_predict()

#step = 0
def train():
  input_app = InputApp.InputApp()
  input_results = input_app.gen_input()

  with tf.variable_scope(FLAGS.main_scope) as scope:
    trainer, predictor =  algos_factory.gen_trainer_and_predictor(FLAGS.algo)
    logging.info('trainer:{}'.format(trainer))
    logging.info('predictor:{}'.format(predictor))
  
    algos_factory.set_eval_mode(trainer)
    ops, gen_feed_dict, deal_results = gen_train(
      input_app, 
      input_results, 
      trainer)
    scope.reuse_variables()
    algos_factory.set_eval_mode(trainer)
    
    if predictor is not None and FLAGS.gen_predict:
      gen_predict_graph(predictor)

    eval_ops, gen_eval_feed_dict, deal_eval_results = gen_validate(
      input_app, 
      input_results, 
      trainer, 
      predictor)

    metric_eval_fn = None
    if FLAGS.metric_eval:
      #generative can do this also but it is slow so just ingore this
      if not algos_factory.is_generative(FLAGS.algo): 
        metric_eval_fn = lambda: evaluator.evaluate_scores(predictor, random=True)

  melt.print_global_varaiables()
  melt.apps.train_flow(ops, 
                       gen_feed_dict_fn=gen_feed_dict,
                       deal_results_fn=deal_results,
                       eval_ops=eval_ops,
                       gen_eval_feed_dict_fn=gen_eval_feed_dict,
                       deal_eval_results_fn=deal_eval_results,
                       optimizer=FLAGS.optimizer,
                       learning_rate=FLAGS.learning_rate,
                       num_steps_per_epoch=input_app.num_steps_per_epoch,
                       model_dir=FLAGS.model_dir,
                       metric_eval_fn=metric_eval_fn,
                       sess=sess)#notice if use melt.constant in predictor then must pass sess

def main(_):
  #-----------init global resource
  logging.set_logging_path(gezi.get_dir(FLAGS.model_dir))
  melt.apps.train.init()

  vocabulary.init()
  text2ids.init()
  
  #must init before main graph
  evaluator.init()

  logging.info('algo:{}'.format(FLAGS.algo))
  logging.info('monitor_level:{}'.format(FLAGS.monitor_level))
  
  global_scope = ''
  if FLAGS.add_global_scope:
    global_scope = FLAGS.global_scope if FLAGS.global_scope else FLAGS.algo
 
  global sess
  sess = melt.get_session(log_device_placement=FLAGS.log_device_placement)

  print('------------', global_scope)
  with tf.variable_scope(global_scope):
    train()
 
if __name__ == '__main__':
  tf.app.run()
