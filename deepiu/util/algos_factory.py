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

from deepiu.image_caption.algos.discriminant_trainer import DiscriminantTrainer
from deepiu.image_caption.algos.discriminant_predictor import DiscriminantTrainer


from deepiu.image_caption.algos.show_and_tell import ShowAndTell 
from deepiu.image_caption.algos.show_and_tell_predictor import ShowAndTellTrainer

from deepiu.textsum.algos.seq2seq import Seq2seq, Seq2seqTrainer

from deepiu.imtxt2txt.algos.imtxt2txt import Imtxt2txt, Imtxt2txtTrainer

from deepiu.textsim.algos.dual_textsim import DualTextsim, DualTextsimTrainer

from deepiu.textsim.algos.decomposable_nli import DecomposableNLI, DecomposableNLITrainer


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
    Algos.bow: DiscriminantTrainer('bow'),
    Algos.rnn: DiscriminantTrainer('rnn'),
    Algos.cnn: DiscriminantTrainer('cnn'),
    Algos.show_and_tell: ShowAndTellTrainer(), 
    Algos.seq2seq: Seq2seqTrainer(), 
    Algos.imtxt2txt: Imtxt2txtTrainer(), 
    Algos.dual_bow: DualTextsimTrainer('bow'), 
    Algos.dual_rnn: DualTextsimTrainer('rnn'), 
    Algos.dual_cnn: DualTextsimTrainer('cnn'), 
    Algos.decomposable_nli: DecomposableNLITrainer()
  }
  if algo not in predictor_map:
    raise ValueError('Unsupported algo %s'%algo) 
  return predictor_map[algo]

def _gen_trainer(algo):
  trainer_map = {
    Algos.bow: DiscriminantTrainer('bow'),
    Algos.rnn: DiscriminantTrainer('rnn'),
    Algos.cnn: DiscriminantTrainer('cnn'),
    Algos.show_and_tell: ShowAndTellTrainer(), 
    Algos.seq2seq: Seq2seqTrainer(), 
    Algos.imtxt2txt: Imtxt2txtTrainer(), 
    Algos.dual_bow: DualTextsimTrainer('bow'), 
    Algos.dual_rnn: DualTextsimTrainer('rnn'), 
    Algos.dual_cnn: DualTextsimTrainer('cnn'), 
    Algos.decomposable_nli: DecomposableNLITrainer()
  }
  if algo not in trainer_map:
    raise ValueError('Unsupported algo %s'%algo) 
  return trainer_map[algo]

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
