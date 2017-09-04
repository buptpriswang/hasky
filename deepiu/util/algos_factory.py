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

import sys, inspect

import melt

from deepiu.image_caption.algos.discriminant_trainer import DiscriminantTrainer
from deepiu.image_caption.algos.discriminant_predictor import DiscriminantPredictor


from deepiu.image_caption.algos.show_and_tell import ShowAndTell 
from deepiu.image_caption.algos.show_and_tell_predictor import ShowAndTellPredictor

from deepiu.textsum.algos.seq2seq import Seq2seq, Seq2seqPredictor

from deepiu.imtxt2txt.algos.imtxt2txt import Imtxt2txt, Imtxt2txtPredictor

from deepiu.textsim.algos.dual_textsim import DualTextsim, DualTextsimPredictor

from deepiu.textsim.algos.decomposable_nli import DecomposableNLI, DecomposableNLIPredictor


class Algos:
  bow = 'bow'    #bow encode for text
  rnn = 'rnn'    #rnn encode for text
  cnn = 'cnn'    #cnn encode for text
  pooling = 'pooling'
  show_and_tell = 'show_and_tell'   #lstm decode for text
  seq2seq = 'seq2seq'
  imtxt2txt = 'imtxt2txt'
  dual_bow = 'dual_bow'
  dual_rnn = 'dual_rnn'
  dual_cnn = 'dual_cnn'
  decomposable_nli = 'decomposable_nli'

class AlgosType:
   discriminant = 0
   generative = 1

AlgosTypeMap = {
  Algos.bow: AlgosType.discriminant,
  Algos.rnn: AlgosType.discriminant,
  Algos.cnn: AlgosType.discriminant,
  Algos.pooling: AlgosType.discriminant,
  Algos.show_and_tell: AlgosType.generative,
  Algos.seq2seq : AlgosType.generative,
  Algos.imtxt2txt : AlgosType.generative,
  Algos.dual_bow : AlgosType.discriminant,
  Algos.dual_rnn : AlgosType.discriminant,
  Algos.dual_cnn : AlgosType.discriminant,
  Algos.decomposable_nli: AlgosType.discriminant
}

def is_discriminant(algo):
  return AlgosTypeMap[algo] == AlgosType.discriminant

def is_generative(algo):
  return AlgosTypeMap[algo] == AlgosType.generative

#TODO this is c++ way, use yaxml python way pass BowTrainer.. like this directly
#'cnn'.. parmas might remove? just as trainer, predictor normal params
# --algo discriminant --encoder bow or --algo discriminant --encoder rnn

def _gen_predictor(algo):
  predictor_map = {
    Algos.bow: DiscriminantPredictor,
    Algos.rnn: DiscriminantPredictor,
    Algos.cnn: DiscriminantPredictor,
    Algos.show_and_tell: ShowAndTellPredictor, 
    Algos.seq2seq: Seq2seqPredictor, 
    Algos.imtxt2txt: Imtxt2txtPredictor, 
    Algos.dual_bow: DualTextsimPredictor, 
    Algos.dual_rnn: DualTextsimPredictor, 
    Algos.dual_cnn: DualTextsimPredictor, 
    Algos.decomposable_nli: DecomposableNLIPredictor
  }
  if algo not in predictor_map:
    print('Unsupported algo %s'%algo, file=sys.stderr) 
    return None
  precitor_fn = predictor_map[algo]
  if 'encoder_type' in inspect.getargspec(precitor_fn.__init__).args:
    return precitor_fn(encoder_type=algo.split('_')[-1])
  else:
    return precitor_fn()

def _gen_trainer(algo):
  trainer_map = {
    Algos.bow: DiscriminantTrainer,
    Algos.rnn: DiscriminantTrainer,
    Algos.cnn: DiscriminantTrainer,
    Algos.show_and_tell: ShowAndTell, 
    Algos.seq2seq: Seq2seq, 
    Algos.imtxt2txt: Imtxt2txt, 
    Algos.dual_bow: DualTextsim, 
    Algos.dual_rnn: DualTextsim, 
    Algos.dual_cnn: DualTextsim, 
    Algos.decomposable_nli: DecomposableNLI
  }
  if algo not in trainer_map:
    raise ValueError('Unsupported algo %s'%algo) 
  trainer_fn = trainer_map[algo]
  if 'encoder_type' in inspect.getargspec(trainer_fn.__init__).args:
    return trainer_fn(encoder_type=algo.split('_')[-1])
  else:
    return trainer_fn()

#TODO use tf.make_template to remove "model_init" scope?
def gen_predictor(algo, reuse=None):
  with tf.variable_scope("init", reuse=reuse):
    predictor = _gen_predictor(algo)
  return predictor
  
def gen_tranier(algo, reuse=None):
  with tf.variable_scope("init", reuse=reuse):
    trainer = _gen_trainer(algo)
  return trainer

def gen_trainer_and_predictor(algo):
  trainer = gen_tranier(algo, reuse=None)
  predictor = gen_predictor(algo, reuse=True)
  return trainer, predictor

def set_eval_mode(trainer):
  with tf.variable_scope("init", reuse=True):
    trainer.__init__(is_training=False)
