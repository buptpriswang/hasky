#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   input_app.py
#        \author   chenghuige  
#          \date   2016-08-27 14:58:25.032042
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

import melt
logging = melt.logging
from deepiu.image_caption import input
import gezi
list_files = gezi.bigdata_util.list_files

try:
  import conf 
  from conf import TEXT_MAX_WORDS, IMAGE_FEATURE_LEN
  from deepiu.util import text2ids
except Exception:
  from deepiu.image_caption.conf import TEXT_MAX_WORDS, IMAGE_FEATURE_LEN 

def print_input_results(input_results):
  print('input_results:')
  for name, tensors in input_results.items():
    print(name)
    if tensors:
      for tensor in tensors:
        print(tensor)

class InputApp(object):
  def __init__(self):
    self.input_train_name = 'input_train'
    self.input_train_neg_name = 'input_train_neg'
    self.input_valid_name = 'input_valid'
    self.fixed_input_valid_name = 'fixed_input_valid'
    self.input_valid_neg_name = 'input_valid_neg'
    #-----------common for all app inputs may be 
    #for evaluate show small results, not for evaluate show cost
    self.num_records = None
    self.num_evaluate_examples = None
    self.num_steps_per_epoch = None

    self.sess = melt.get_session()

    self.train_with_validation = None
    self.eval_fixed_images = None

    # self.step = 0

  def gen_train_input(self, inputs, decode_fn):
     #--------------------- train
    logging.info('train_input: %s'%FLAGS.train_input)
    trainset = list_files(FLAGS.train_input)
    logging.info('trainset:{} {}'.format(len(trainset), trainset[:2]))
    
    assert len(trainset) >= FLAGS.min_records, '%d %d'%(len(trainset), FLAGS.min_records)
    if FLAGS.num_records > 0:
      assert len(trainset) == FLAGS.num_records, len(trainset)

    num_records = gezi.read_int_from(FLAGS.num_records_file)
    logging.info('num_records:{}'.format(num_records))
    logging.info('batch_size:{}'.format(FLAGS.batch_size))
    logging.info('FLAGS.num_gpus:{}'.format(FLAGS.num_gpus))
    num_gpus = max(FLAGS.num_gpus, 1)
    num_steps_per_epoch = num_records // (FLAGS.batch_size * num_gpus)
    logging.info('num_steps_per_epoch:{}'.format(num_steps_per_epoch))
    self.num_records = num_records
    self.num_steps_per_epoch = num_steps_per_epoch
    
    image_name, image_feature, text, text_str = inputs(
      trainset, 
      decode_fn=decode_fn,
      batch_size=FLAGS.batch_size,
      #seed=seed,
      num_threads=FLAGS.num_threads,
      batch_join=FLAGS.batch_join,
      shuffle_files=FLAGS.shuffle_files,
      fix_sequence=FLAGS.fix_sequence,
      num_prefetch_batches=FLAGS.num_prefetch_batches,
      min_after_dequeue=FLAGS.min_after_dequeue,
      name=self.input_train_name)

    if FLAGS.monitor_level > 1:
      lengths = melt.length(text)
      melt.scalar_summary("text/batch_min", tf.reduce_min(lengths))
      melt.scalar_summary("text/batch_max", tf.reduce_max(lengths))
      melt.scalar_summary("text/batch_mean", tf.reduce_mean(lengths))
    return (image_name, image_feature, text, text_str), trainset

  def gen_train_neg_input(self, inputs, decode_neg_fn, trainset):
    assert FLAGS.num_negs > 0
    results = inputs(
      trainset, 
      decode_fn=decode_neg_fn,
      batch_size=FLAGS.batch_size * FLAGS.num_negs,
      num_threads=FLAGS.num_threads,
      batch_join=FLAGS.batch_join,
      shuffle_files=FLAGS.shuffle_files,
      num_prefetch_batches=FLAGS.num_prefetch_batches,
      min_after_dequeue=FLAGS.min_after_dequeue,
      #fix_sequence=FLAGS.fix_sequence,
      name=self.input_train_neg_name)

    results = input.reshape_neg_tensors(results, FLAGS.batch_size, FLAGS.num_negs)
    return results

  def gen_valid_input(self, inputs, decode_fn):
    #---------------------- valid  
    validset = list_files(FLAGS.valid_input)
    logging.info('validset:{} {}'.format(len(validset), validset[:2]))
    eval_image_name, eval_image_feature, eval_text, eval_text_str = inputs(
      validset, 
      decode_fn=decode_fn,
      batch_size=FLAGS.eval_batch_size,
      num_threads=FLAGS.num_threads,
      batch_join=FLAGS.batch_join,
      shuffle_files=FLAGS.eval_shuffle_files,
      seed=FLAGS.eval_seed,
      fix_random=FLAGS.eval_fix_random,
      num_prefetch_batches=FLAGS.num_prefetch_batches,
      min_after_dequeue=FLAGS.min_after_dequeue,
      fix_sequence=FLAGS.fix_sequence,
      name=self.input_valid_name)

    eval_batch_size = FLAGS.eval_batch_size
   
    eval_result = eval_image_name, eval_image_feature, eval_text, eval_text_str
    eval_show_result = None
    
    if FLAGS.show_eval:
      eval_fixed_images = bool(FLAGS.fixed_valid_input)
      self.eval_fixed_images = eval_fixed_images
      if eval_fixed_images:
        assert FLAGS.fixed_eval_batch_size >= FLAGS.num_fixed_evaluate_examples, '%d %d'%(FLAGS.fixed_eval_batch_size, FLAGS.num_fixed_evaluate_examples)
        logging.info('fixed_eval_batch_size:{}'.format(FLAGS.fixed_eval_batch_size))
        logging.info('num_fixed_evaluate_examples:{}'.format(FLAGS.num_fixed_evaluate_examples))
        logging.info('num_evaluate_examples:{}'.format(FLAGS.num_evaluate_examples))
        #------------------- fixed valid
        fixed_validset = list_files(FLAGS.fixed_valid_input)
        logging.info('fixed_validset:{} {}'.format(len(fixed_validset), fixed_validset[:2]))
        fixed_image_name, fixed_image_feature, fixed_text, fixed_text_str = inputs(
          fixed_validset, 
          decode_fn=decode_fn,
          batch_size=FLAGS.fixed_eval_batch_size,
          fix_sequence=True,
          num_prefetch_batches=FLAGS.num_prefetch_batches,
          min_after_dequeue=FLAGS.min_after_dequeue,
          name=self.fixed_input_valid_name)

        #-------------shrink fixed image input as input batch might large then what we want to show, only choose top num_fixed_evaluate_examples
        fixed_image_name = melt.first_nrows(fixed_image_name, FLAGS.num_fixed_evaluate_examples)
        fixed_image_feature = melt.first_nrows(fixed_image_feature, FLAGS.num_fixed_evaluate_examples)
        fixed_text = melt.first_nrows(fixed_text, FLAGS.num_fixed_evaluate_examples)
        if FLAGS.dynamic_batch_length:
          fixed_text = melt.make_batch_compat(fixed_text)
        fixed_text_str = melt.first_nrows(fixed_text_str, FLAGS.num_fixed_evaluate_examples)

        #notice read data always be FLAGS.fixed_eval_batch_size, if only 5 tests then will wrapp the data 
        eval_image_name = tf.concat([fixed_image_name, eval_image_name], 0)
        eval_image_feature = tf.concat([fixed_image_feature, eval_image_feature], 0)

        #melt.align only need if you use dynamic batch/padding
        if FLAGS.dynamic_batch_length:
          fixed_text, eval_text = melt.align_col_padding2d(fixed_text, eval_text)

        eval_text = tf.concat([fixed_text, eval_text], 0)
        eval_text_str = tf.concat([fixed_text_str, eval_text_str], 0)
        eval_batch_size = FLAGS.num_fixed_evaluate_examples + FLAGS.eval_batch_size 
      
      #should aways be FLAGS.num_fixed_evaluate_examples + FLAGS.num_evaluate_examples
      num_evaluate_examples = min(eval_batch_size, FLAGS.num_fixed_evaluate_examples + FLAGS.num_evaluate_examples)

      #------------combine valid and fixed valid 
      evaluate_image_name = melt.first_nrows(eval_image_name, num_evaluate_examples)
      evaluate_image_feature = melt.first_nrows(eval_image_feature, num_evaluate_examples)
      evaluate_text_str = melt.first_nrows(eval_text_str, num_evaluate_examples)
      evaluate_text = melt.first_nrows(eval_text, num_evaluate_examples)

      self.num_evaluate_examples = num_evaluate_examples
      
      eval_result = eval_image_name, eval_image_feature, eval_text, eval_text_str
      eval_show_result = evaluate_image_name, evaluate_image_feature, evaluate_text, evaluate_text_str
    
    return eval_result, eval_show_result, eval_batch_size

  def gen_valid_neg_input(self, inputs, decode_neg_fn, trainset, eval_batch_size):
    results = inputs(
      trainset, 
      decode_fn=decode_neg_fn,
      batch_size=eval_batch_size * FLAGS.num_negs,
      num_threads=FLAGS.num_threads,
      batch_join=FLAGS.batch_join,
      shuffle_files=FLAGS.eval_shuffle_files,
      fix_random=FLAGS.eval_fix_random,
      num_prefetch_batches=FLAGS.num_prefetch_batches,
      min_after_dequeue=FLAGS.min_after_dequeue,
      fix_sequence=FLAGS.fix_sequence,
      name=self.input_valid_neg_name)

    #[batch_size, num_negs, 1] -> [batch_size, num_negs], notice tf.squeeze will get [batch_size] when num_negs == 1
    results = input.reshape_neg_tensors(results, eval_batch_size, FLAGS.num_negs)
    results[0] = tf.squeeze(results[0], squeeze_dims=[-1]) #ltext_str
    results[-1] = tf.squeeze(results[-1], squeeze_dims=[-1]) #rtext_str

    return results

  def gen_input(self, train_only=False):
    timer = gezi.Timer('gen input')

    assert not (FLAGS.feed_dict and FLAGS.dynamic_batch_length), \
          'if use feed dict then must use fixed batch length, or use buket mode(@TODO)'

    input_results = {}

    input_name_list = [self.input_train_name, self.input_train_neg_name, \
                       self.input_valid_name, self.fixed_input_valid_name, \
                       self.input_valid_neg_name]

    for name in input_name_list:
      input_results[name] = None

    assert FLAGS.shuffle_then_decode, "since use sparse data for text, must shuffle then decode"

    inputs, decode_fn, decode_neg_fn = \
     input.get_decodes(use_neg=(FLAGS.num_negs > 0))

    input_results[self.input_train_name], trainset = self.gen_train_input(inputs, decode_fn)

    if decode_neg_fn is not None:
      input_results[self.input_train_neg_name] = self.gen_train_neg_input(inputs, decode_neg_fn, trainset)
    
    if not train_only:
      #---------------------- valid
      train_with_validation = bool(FLAGS.valid_input) 
      self.train_with_validation = train_with_validation
      print('train_with_validation:', train_with_validation)
      if train_with_validation:
        input_results[self.input_valid_name], \
        input_results[self.fixed_input_valid_name], \
        eval_batch_size = self.gen_valid_input(inputs, decode_fn)

        if decode_neg_fn is not None:
          input_results[self.input_valid_neg_name] = self.gen_valid_neg_input(inputs, decode_neg_fn, trainset, eval_batch_size)

    print_input_results(input_results)

    timer.print()

    return input_results
