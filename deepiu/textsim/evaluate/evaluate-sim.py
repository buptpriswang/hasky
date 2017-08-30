#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   evaluate-sim-score.py
#        \author   chenghuige  
#          \date   2016-09-25 00:46:53.890615
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('model_dir', '/home/gezi/new/temp/makeup/title2name/model/bow/', '')  
#flags.DEFINE_string('model_dir', '/home/gezi/new/temp/makeup/title2name/model/cnn.hic/', '')  

flags.DEFINE_string('exact_model_dir', '/home/gezi/new/temp/makeup/title2name/model/bow.elementwise/', '')  
flags.DEFINE_string('vocab', '/home/gezi/new/temp/makeup/title2name/tfrecord/seq-basic/vocab.txt', '')

flags.DEFINE_bool('use_exact_predictor', False, '')

flags.DEFINE_string('key', 'score', '')
flags.DEFINE_string('lkey', 'dual_bow/main/ltext:0', '')
flags.DEFINE_string('rkey', 'dual_bow/main/rtext:0', '')

#flags.DEFINE_string('lkey', 'dual_cnn/main/ltext:0', '')
#flags.DEFINE_string('rkey', 'dual_cnn/main/rtext:0', '')


flags.DEFINE_string('exact_key', 'score', '')
flags.DEFINE_string('exact_lkey', 'dual_bow2/main/ltext:0', '')
flags.DEFINE_string('exact_rkey', 'dual_bow2/main/rtext:0', '')

flags.DEFINE_float('exact_ratio', 1., '')

flags.DEFINE_integer('np_seed', 1024, '0 random otherwise fixed random')

import sys 
import numpy as np
import melt
logging = melt.logging
import gezi 

from deepiu.util import evaluator
from deepiu.util import algos_factory

class Predictor(melt.PredictorBase):
  def __init__(self, model_dir, key, lkey, rkey, index=0):
    self._predictor = melt.Predictor(model_dir)
    self._key = key
    self._lkey = lkey
    self._rkey = rkey
    self._index = index

  def predict(self, ltext, rtext):
   score = self._predictor.inference(self._key, 
                            feed_dict= {
                                    self._lkey: ltext,
                                    self._rkey: rtext
                                    },
                            index=self._index
                            )
   return score

def evaluate_score():
  evaluator.init()
  text_max_words = evaluator.all_distinct_texts.shape[1]
  print('text_max_words:', text_max_words)
  predictor = Predictor(FLAGS.model_dir, FLAGS.key, FLAGS.lkey, FLAGS.rkey, index=0)
  exact_predictor=None

  if FLAGS.use_exact_predictor:
    exact_predictor = Predictor(FLAGS.exact_model_dir, FLAGS.exact_key, FLAGS.exact_lkey, FLAGS.exact_rkey, index=-1)
  print(tf.get_collection(FLAGS.key))
  seed = FLAGS.np_seed if FLAGS.np_seed else None
  index = evaluator.random_predict_index(seed=seed)
  evaluator.evaluate_scores(predictor, random=True, index=index)
  if exact_predictor is not None:
    ##well for seq2seq did experiment and for makeup title2name score(average time per step) is much better then ori_score
    ##so just juse score will be fine
    #exact_predictor._key = 'ori_score'
    #evaluator.evaluate_scores(predictor, random=True, exact_predictor=exact_predictor, index=index)
    #exact_predictor._key = 'score'
    evaluator.evaluate_scores(predictor, random=True, exact_predictor=exact_predictor, exact_ratio=FLAGS.exact_ratio, index=index)


def main(_):
  logging.init(logtostderr=True, logtofile=False)
  evaluate_score()

if __name__ == '__main__':
  tf.app.run()
