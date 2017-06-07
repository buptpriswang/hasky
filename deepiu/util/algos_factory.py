#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   algos_factory.py
#        \author   chenghuige  
#          \date   2016-09-17 19:42:42.947589
#   \Description  
# ==============================================================================

"""
Should move to util
"""
  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import melt

from deepiu.image_caption.algos.bow import Bow, BowPredictor 
from deepiu.image_caption.algos.rnn import Rnn, RnnPredictor
from deepiu.image_caption.algos.pooling import Pooling, PoolingPredictor


from deepiu.image_caption.algos.show_and_tell import ShowAndTell 
from deepiu.image_caption.algos.show_and_tell_predictor import ShowAndTellPredictor

from deepiu.textsum.algos.seq2seq import Seq2seq, Seq2seqPredictor

from deepiu.imtxt2txt.algos.imtxt2txt import Imtxt2txt, Imtxt2txtPredictor

class Algos:
  bow = 'bow'    #bow encode for text
  rnn = 'rnn'    #rnn encode for text
  cnn = 'cnn'    #cnn encode for text
  pooling = 'pooling'
  show_and_tell = 'show_and_tell'   #lstm decode for text
  seq2seq = 'seq2seq'
  imtxt2txt = 'imtxt2txt'

class AlgosType:
   discriminant = 0
   generative = 1

AlgosTypeMap = {
  Algos.bow: AlgosType.discriminant,
  Algos.rnn: AlgosType.discriminant,
  Algos.pooling: AlgosType.discriminant,
  Algos.show_and_tell: AlgosType.generative,
  Algos.seq2seq : AlgosType.generative,
  Algos.imtxt2txt : AlgosType.generative,
}

def is_discriminant(algo):
  return AlgosTypeMap[algo] == AlgosType.discriminant

def is_generative(algo):
  return AlgosTypeMap[algo] == AlgosType.generative

#TODO this is c++ way, use yaxml python way pass BowPredictor.. like this directly
def _gen_predictor(algo):
  if algo == Algos.bow:
    return BowPredictor()
  elif algo == Algos.show_and_tell:
    return ShowAndTellPredictor()
  elif algo == Algos.rnn:
    return RnnPredictor()
  elif algo == Algos.pooling:
    return PoolingPredictor()
  elif algo == Algos.seq2seq:
    return Seq2seqPredictor()
  elif algo == Algos.imtxt2txt:
    return Imtxt2txtPredictor()
  else:
    raise ValueError('Unsupported algo %s'%algo) 

def _gen_trainer(algo):
  if algo == Algos.bow:
    return Bow()
  elif algo == Algos.show_and_tell:
    return ShowAndTell()
  elif algo == Algos.rnn:
    return Rnn()
  elif algo == Algos.pooling:
    return Pooling()
  elif algo == Algos.seq2seq:
    return Seq2seq()
  elif algo == Algos.imtxt2txt:
    return Imtxt2txt()
  else:
    raise ValueError('Unsupported algo %s'%algo) 

#TODO use tf.make_template to remove "model_init" scope?
def gen_predictor(algo, reuse=None):
  with tf.variable_scope("model_init", reuse=reuse):
    predictor = _gen_predictor(algo)
  return predictor
  
def gen_tranier(algo, reuse=None):
  with tf.variable_scope("model_init", reuse=reuse):
    trainer = _gen_trainer(algo)
  return trainer

def gen_trainer_and_predictor(algo):
  trainer = gen_tranier(algo, reuse=None)
  predictor = gen_predictor(algo, reuse=True)
  return trainer, predictor
