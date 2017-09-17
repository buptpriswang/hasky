#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   embedding.py
#        \author   chenghuige  
#          \date   2016-12-24 19:55:37.327855
#   \Description  
# ==============================================================================
  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS
  
flags.DEFINE_integer('emb_dim', 256, 'embedding dim for each word, notice for rnn bidirectional here should be acutal emb_dim * 2')
flags.DEFINE_float('weight_stddev', 1e-4,  
                                  """weight stddev, 
                                     @Notice if use bias then small stddev like 0.01 might not lead to convergence, 
                                     causing layer weight value always be 0 with random_normal""")
flags.DEFINE_float('initializer_scale', 0.08, 'used for weights initalize using random_uniform, default value 0.08 follow im2txt')

flags.DEFINE_string('word_embedding_file', None, 'load pre trained word embedding from word2vec or glov if not None')
flags.DEFINE_boolean('finetune_word_embedding', True, 'wether update word embedding')
flags.DEFINE_boolean('position_embedding', False, 'wether use postion embedding')

import tensorflow.contrib.slim as slim

import melt
logging = melt.logging

from deepiu.util import vocabulary  

import glob

#TODO try l2_regularizer and compare
#weights = slim.variable('weights',
#                             shape=[10, 10, 3 , 3],
#                             initializer=tf.truncated_normal_initializer(stddev=0.1),
#                             regularizer=slim.l2_regularizer(0.05),
#                             device='/CPU:0')
def get_embedding(name='emb', height=None, emb_dim=None, trainable=True):
  emb_dim = emb_dim or FLAGS.emb_dim
  if height is None:
    vocabulary.init()
    height = vocabulary.get_vocab_size() 
  
  init_width = 0.5 / emb_dim
  emb = melt.variable.get_weights_uniform(name, [height, emb_dim], -init_width, init_width, trainable=trainable)
  #return to above code if this works not better
  #emb = melt.variable.get_weights_truncated(name, [vocab_size, emb_dim], stddev=FLAGS.weight_stddev)
  
  return emb 

def get_embedding_cpu(name='emb', height=None, emb_dim=None, trainable=True):
  with tf.device('/CPU:0'):
    return get_embedding(name, height=height, emb_dim=emb_dim, trainable=trainable)

def get_or_restore_embedding(name='emb'):
  # cpu for adgrad optimizer
  if (not FLAGS.word_embedding_file) or glob.glob(FLAGS.model_dir + '/model.ckpt*'):
    logging.info('Word embedding random init or from model_dir:{} and trainable=:{}'.format(
        FLAGS.model_dir, FLAGS.finetune_word_embedding))
    emb = get_embedding(
        name=name, trainable=FLAGS.finetune_word_embedding)
    melt.try_add_to_collection('word_embedding', emb)
  else:
    # https://github.com/tensorflow/tensorflow/issues/1570
    # still adgrad must cpu..
    # if not fintue emb this will be ok if fintune restart will ok ? must not use word embedding file? os.path.exists(FLAGS.model_dir) ? judge?
    # or will still try to load from check point ? TODO for safe you could re run by setting word_embedding_file as None or ''
    logging.info('Loading word embedding from:{} and trainable=:{}'.format(
        FLAGS.word_embedding_file, FLAGS.finetune_word_embedding))
    emb = melt.load_constant(
        FLAGS.word_embedding_file, name=name, trainable=FLAGS.finetune_word_embedding)
  return emb

def get_or_restore_embedding_cpu(name='emb'):
  with tf.device('/CPU:0'):
    return get_or_restore_embedding(name)

def get_position_embedding(name='pos_emb'):
  if FLAGS.position_embedding:
    logging.info('Using position embedding')
    pos_emb = embedding.get_embedding(name='pos_emb', height=TEXT_MAX_WORDS)
  else:
    pos_emb = None
  return pos_emb

def get_position_embedding_cpu(name='pos_emb'):
  with tf.device('/CPU:0'):
    return get_position_embedding(name)