#!/usr/bin/env python
# ==============================================================================
#          \file   flow.py
#        \author   chenghuige  
#          \date   2016-08-17 10:48:46.141744
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys
import tensorflow as tf
import melt 
import gezi
from melt.utils import logging

def tf_flow(process_once, num_steps=None, sess=None):
  """
  basic flow for tf records, allow most freedom for usage, if not tfrecords no need for flow
  Args:
  train_once: function with 2 inputs sess and step
  """
  init_op = tf.group(tf.global_variables_initializer(),
                    tf.local_variables_initializer())
  if sess is None:
    sess = tf.InteractiveSession()
  sess.run(init_op)

  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(sess=sess, coord=coord)

  try:
    step = 0
    while not coord.should_stop():
      stop = process_once(sess, step)
      if stop is True:
        print('Early stop running %d stpes'%(step))
        raise tf.errors.OutOfRangeError(None, None,'Early stop running %d stpes'%(step))
      step += 1
      if num_steps and step == num_steps:
        raise tf.errors.OutOfRangeError(None, None, 'Reached max num steps')
  except tf.errors.OutOfRangeError:
    print('Done training for %d steps.' % (step))
  finally:
    coord.request_stop()

  coord.join(threads)
  sess.close()
  return step
 
def _get_model_path(model_dir, save_model):
  if not os.path.exists(model_dir):
    if save_model:
      gezi.try_mkdir(model_dir)
    return None
  ckpt = tf.train.get_checkpoint_state(model_dir)
  if ckpt and ckpt.model_checkpoint_path:
    #input valid dir and return latest model
    return os.path.join(model_dir, os.path.basename(ckpt.model_checkpoint_path))
  elif os.path.isdir(model_dir):
    #input valid dir but no models
    return None 
  else:
    #this might be user specified model like ./model/model-100.ckpt
    #the file exists and we NOTICE we do not check if it is valid model file!
    return model_dir

def _get_checkpoint_path(checkpoint_path, step, num_steps_per_epoch):
  if not num_steps_per_epoch:
    return checkpoint_path
  return '%s-%.1f'%(checkpoint_path, step / num_steps_per_epoch)

def tf_train_flow(train_once_fn, 
                  model_dir='./model', 
                  max_models_keep=1, 
                  save_interval_seconds=600, 
                  save_interval_steps=1000, 
                  num_epochs=None,
                  num_steps=None, 
                  save_model=True,
                  save_interval_epochs=1, 
                  num_steps_per_epoch=0,
                  restore_from_latest=True,
                  metric_eval_fn=None,
                  init_fn=None,
                  sess=None):
  """
  similary flow as tf_flow, but add model try reload and save
  """
  if sess is None:
    #TODO melt.get_session is global session but may cause
    sess = melt.get_session()
  logging.info('tf_train_flow start')
  print('max_models_keep:', max_models_keep)
  print('save_interval_seconds:', save_interval_seconds)
  
  saver = tf.train.Saver(
    max_to_keep=max_models_keep, 
    keep_checkpoint_every_n_hours=save_interval_seconds / 3600.0)
  
  #epoch_saver = tf.train.Saver(
  #  max_to_keep=max_models_keep,
  #  keep_checkpoint_every_n_hours=24) # TODO  
  best_epoch_saver = tf.train.Saver() 
  
  #pre_step means the step last saved, train without pretrained,then -1
  pre_step = -1;
  model_path = _get_model_path(model_dir, save_model)
  model_dir = gezi.get_dir(model_dir) #incase you pass ./model/model-ckpt1000 -> ./model
  if model_path is not None:
    if not restore_from_latest:
      print('using recent but not latest model', file=sys.stderr)
      model_path = melt.recent_checkpoint(model_dir)
    model_name = os.path.basename(model_path)
    timer = gezi.Timer('Loading and training from existing model [%s]'%model_path)
    saver.restore(sess, model_path)
    timer.print()
    pre_step = melt.get_model_step(model_path)
    if 'epoch' in model_name:
      pre_step *= num_steps_per_epoch
    #for non 0 eopochs  without this will be
    #Attempting to use uninitialized value input/input_producer/limit_epochs/epochs
    sess.run(tf.local_variables_initializer())
  else:
    print('Train all start step 0', file=sys.stderr)
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    sess.run(init_op)

    if init_fn is not None:
      init_fn(sess)
  
  if save_interval_epochs and num_steps_per_epoch:
    epoch_dir = os.path.join(model_dir, 'epoch')
    gezi.try_mkdir(epoch_dir)
  
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(sess=sess, coord=coord)
  checkpoint_path = os.path.join(model_dir, 'model.ckpt')

  try:
    step = start = pre_step +  1
    #hack just for save one model after load
    if num_steps and num_steps < step:
      print('just load and resave then exit', file=sys.stderr)
      saver.save(sess, 
                 _get_checkpoint_path(checkpoint_path, step, num_steps_per_epoch), 
                 global_step=step)
      sess.close()
      exit(0)

    early_stop = True #TODO allow config
    num_bad_epochs = 0
    pre_epoch_eval_loss = 1e20
    best_epoch_eval_loss = 1e20
    num_allowed_bad_epochs = 4 #allow 5 non decrease eval loss epochs  before stop
    while not coord.should_stop():
      stop = train_once_fn(sess, step, is_start=(step==start))
      if save_model and step:
        #step 0 is also saved! actually train one step and save
        if step % save_interval_steps == 0:
          timer = gezi.Timer('save model step %d to %s'%(step, checkpoint_path))
          saver.save(sess, 
                     _get_checkpoint_path(checkpoint_path, step, num_steps_per_epoch), 
                     global_step=step)
          timer.print()
        if save_interval_epochs and num_steps_per_epoch and step % (num_steps_per_epoch * save_interval_epochs) == 0:
          epoch = step // num_steps_per_epoch
          eval_loss = melt.eval_loss()
          if eval_loss:
            #['eval_loss:3.2','eal_accuracy:4.3']
            eval_loss = float(eval_loss.strip('[]').split(',')[0].strip("'").split(':')[-1])
            if os.path.exists(os.path.join(epoch_dir, 'best_eval_loss.txt')):
              with open(os.path.join(epoch_dir, 'best_eval_loss.txt')) as f:
                best_epoch_eval_loss = float(f.readline().split()[-1].strip())
            if eval_loss < best_epoch_eval_loss:
              best_epoch_eval_loss = eval_loss
              logging.info('Now best eval loss is epoch %d eval_loss:%f' % (epoch, eval_loss))
              with open(os.path.join(epoch_dir, 'best_eval_loss.txt'), 'w') as f:
                f.write('%d %d %f\n'%(epoch, step, best_epoch_eval_loss))
              best_epoch_saver.save(sess, 
                                    os.path.join(epoch_dir,'model.cpkt-best'))

            with open(os.path.join(epoch_dir, 'eval_loss.txt'), 'a') as f:
               f.write('%d %d %f\n'%(epoch, step, eval_loss))
            if eval_loss >= pre_epoch_eval_loss:
              num_bad_epochs += 1
              if num_bad_epochs > num_allowed_bad_epochs:
                logging.warning('Evaluate loss not decrease for last %d epochs'% (num_allowed_bad_epochs + 1))
                if early_stop:
                  stop = True 
            else:
              num_bad_epochs = 0
            pre_epoch_eval_loss = eval_loss

          #epoch_saver.save(sess, 
          #                 os.path.join(epoch_dir,'model.cpkt-%d'%epoch), 
          #                 global_step=step)
      if stop is True:
        print('Early stop running %d stpes'%(step), file=sys.stderr)
        raise tf.errors.OutOfRangeError(None, None,'Early stop running %d stpes'%(step))
      if num_steps and (step + 1) == start + num_steps:
        raise tf.errors.OutOfRangeError(None, None,'Reached max num steps')
      max_num_epochs = 1000
      if num_steps_per_epoch and step // num_steps_per_epoch == max_num_epochs:
        raise tf.errors.OutOfRangeError(None, None,'Reached max num epochs of %d'%max_num_epochs)
      step += 1
  except tf.errors.OutOfRangeError, e:
    if not (step==start) and save_model and step % save_interval_steps != 0:
      saver.save(sess, 
                 _get_checkpoint_path(checkpoint_path, step, num_steps_per_epoch), 
                 global_step=step)
    if metric_eval_fn is not None:
      metric_eval_fn()
    if (num_epochs and step / num_steps_per_epoch >= num_epochs) or (num_steps and (step + 1) == start + num_steps) :
      print('Done training for %d steps.' % (step + 1), file=sys.stderr)
      #FIXME becase coord.join seems not work,  RuntimeError: Coordinator stopped with threads still running: Thread-9
      exit(0)
    else:
      print('Should not stop, but stopped at epoch: %.3f'%(step / num_steps_per_epoch), file=sys.stderr)
      raise e
  finally:
    coord.request_stop()

  coord.join(threads, stop_grace_period_secs=5)
  #FIMXE due to use melt.get_session(global not handle del well)
  #Done training for 3090020 steps.
  #Exception TypeError: "'NoneType' object is not callable" in <bound method Session.__del__ of <tensorflow.python.client.session.Session object at 0x7f6cf33cd450>> ignored
  sess.close()

#@TODO not tested yet
def tf_test_flow(test_once, model_dir='./model', 
                 model_name=None, num_epochs=1, num_steps=0,
                 sess=None):
  """
  basic flow for tf records, allow most freedom for usage, if not tfrecords no need for flow
  Args:
  test_once: function with 2 inputs sess and step
  model_dir: can be dir like ./model will fetch lates model in model dir , or be real model path like ./model/model.0.ckpt
  """
  if sess is None:
    sess = tf.InteractiveSession()

  melt.restore(sess, model_dir, model_name)

  if not os.path.isdir(model_dir):
    model_dir = os.path.dirname(model_dir)
  summary_op = tf.merge_all_summaries()
  summary_writer = tf.train.SummaryWriter(model_dir, sess.graph)

  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(sess=sess, coord=coord)
  try:
    step = 0
    while not coord.should_stop():
      test_once(sess, step)
      step += 1
      if num_steps and step == num_steps:
        raise tf.errors.OutOfRangeError(None, None, 'Reached max num steps')
  except tf.errors.OutOfRangeError:
    print('Done testing for %d epochs, %d steps.' % (num_epochs, step))
  finally:
    # When done, ask the threads to stop.
    coord.request_stop()
  # Wait for threads to finish.
  coord.join(threads)
