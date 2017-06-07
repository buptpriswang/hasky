#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   input_flags.py
#        \author   chenghuige  
#          \date   2016-12-25 00:17:18.268341
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS
  
#--------- read data
flags.DEFINE_integer('batch_size', 32, 'Batch size. default as im2text default')
flags.DEFINE_integer('eval_batch_size', 100, 'Batch size.')
flags.DEFINE_integer('fixed_eval_batch_size', 30, """must >= num_fixed_evaluate_examples
                                                     if == real dataset len then fix sequence show
                                                     if not == can be show different fixed each time
                                                     usefull if you want only show see 2 and 
                                                     be different each time
                                                     if you want see 2 by 2 seq
                                                     then num_fixed_evaluate_example = 2
                                                          fixed_eval_batch_size = 2
                                                  """)

flags.DEFINE_integer('num_fixed_evaluate_examples', 30, '')
flags.DEFINE_integer('num_evaluate_examples', 1, '')

flags.DEFINE_integer('num_threads', 12, """threads for reading input tfrecords,
                                           setting to 1 may be faster but less randomness
                                        """)

flags.DEFINE_boolean('shuffle_files', True, '')
flags.DEFINE_boolean('batch_join', True, '')
flags.DEFINE_boolean('shuffle_batch', True, '')

flags.DEFINE_boolean('shuffle_then_decode', True, 
                     """ actually this is decided by is_sequence_example.. 
                     if is_sequence_example then False, if just example not sequence then True since is sparse
                     TODO remove this
                     """)
flags.DEFINE_boolean('is_sequence_example', False, '')
flags.DEFINE_string('buckets', '', 'empty meaning not use, other wise looks like 5,10,15,30')

flags.DEFINE_boolean('dynamic_batch_length', True, 
                     """very important False means all batch same size! 
                        otherwise use dynamic batch size
                        Now only not sequence_example data will support dyanmic_batch_length=False""")
  

flags.DEFINE_integer('num_negs', 1, '0 means no neg')

flags.DEFINE_boolean('feed_dict', False, '')

#---------- input dirs
#@TODO will not use input pattern but use dir since hdfs now can not support glob well
flags.DEFINE_string('train_input', '/tmp/train/train_*', 'must provide')
flags.DEFINE_string('valid_input', '', 'if empty will train only')
flags.DEFINE_string('fixed_valid_input', '', 'if empty wil  not eval fixed images')
flags.DEFINE_string('num_records_file', '', '')
flags.DEFINE_integer('min_records', 12, '')
flags.DEFINE_integer('num_records', 0, 'if not 0, will check equal')


#---------- input reader
flags.DEFINE_integer('min_after_dequeue', 0, """by deafualt will be 500, 
                                                set to large number for production training 
                                                for better randomness""")
flags.DEFINE_integer('num_prefetch_batches', 0, '')


#----------eval
flags.DEFINE_boolean('legacy_rnn_decoder', False, '')
flags.DEFINE_boolean('experiment_rnn_decoder', False, '')

flags.DEFINE_boolean('show_eval', True, '')

flags.DEFINE_boolean('eval_shuffle_files', True, '')
flags.DEFINE_boolean('eval_fix_random', True, '')
flags.DEFINE_integer('eval_seed', 1024, '')

flags.DEFINE_integer('seed', 1024, '')

flags.DEFINE_boolean('fix_sequence', False, '')

#----------strategy 

flags.DEFINE_string('seg_method', 'default', '')
flags.DEFINE_boolean('feed_single', False, '')

flags.DEFINE_boolean('gen_predict', True, '')


flags.DEFINE_string('decode_name', 'text', '')
flags.DEFINE_string('decode_str_name', 'text_str', '')

#--------for image caption  TODO move to image_caption/input.py ?
flags.DEFINE_boolean('pre_calc_image_feature', True, '')
flags.DEFINE_boolean('distort_image', False, '')
flags.DEFINE_string('image_model_name', 'InceptionV3', '')
flags.DEFINE_integer('image_width', 299, 'default width of inception v3')
flags.DEFINE_integer('image_height', 299, 'default height of inception v3')
flags.DEFINE_string('image_checkpoint_file', '/home/gezi/data/inceptionv3/inception_v3.ckpt', '')
#---in melt.apps.image_processing.py
#flags.DEFINE_string('image_model_name', 'InceptionV3', '')
flags.DEFINE_string('one_image', '/home/gezi/data/flickr/flickr30k-images/1000092795.jpg', '')

flags.DEFINE_string('image_feature_name', 'image_feature', '')
