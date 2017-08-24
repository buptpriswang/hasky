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

flags.DEFINE_string('lkey', 'dual_bow/main/ltext:0', '')
flags.DEFINE_string('rkey', 'dual_bow/main/rtext:0', '')

#flags.DEFINE_string('lkey', 'dual_cnn/main/ltext:0', '')
#flags.DEFINE_string('rkey', 'dual_cnn/main/rtext:0', '')


flags.DEFINE_string('exact_lkey', 'dual_bow2/main/ltext:0', '')
flags.DEFINE_string('exact_rkey', 'dual_bow2/main/rtext:0', '')

import sys 
import numpy as np
import melt
logging = melt.logging
import gezi

from deepiu.util import evaluator
from deepiu.util import algos_factory

class Predictor:
  def __init__(self, model_dir, lkey, rkey, index=0):
    self._predictor = melt.Predictor(model_dir)
    self._lkey = lkey
    self._rkey = rkey
    self._index = index

  def bulk_predict(self, ltext, rtext):
   score = self._predictor.inference('score', 
                            feed_dict= {
                                    self._lkey: ltext,
                                    self._rkey: rtext
                                    },
                            index=self._index
                            )
   return score

  def elementwise_bulk_predict(self, ltexts, rtexts):
    scores = []
    if len(rtexts) >= len(ltexts):
      for ltext in ltexts:
        stacked_ltexts = np.array([ltext] * len(rtexts))
        score = self.bulk_predict(stacked_ltexts, rtexts)
        score = np.squeeze(score)
        scores.append(score)
    else:
      for rtext in rtexts:
        stacked_rtexts = np.array([rtext] * len(ltexts))
        score = self.bulk_predict(ltexts, stacked_rtexts)
        score = np.squeeze(score)
        scores.append(score)
    return np.array(scores)  

def evaluate_score():
  evaluator.init()
  text_max_words = evaluator.all_distinct_texts.shape[1]
  print('text_max_words:', text_max_words)
  predictor = Predictor(FLAGS.model_dir, FLAGS.lkey, FLAGS.rkey, index=0)
  exact_predictor=None

  if FLAGS.use_exact_predictor:
    exact_predictor = Predictor(FLAGS.exact_model_dir, FLAGS.exact_lkey, FLAGS.exact_rkey, index=1)
  print(tf.get_collection('score'))
  evaluator.evaluate_scores(predictor, random=True, exact_predictor=exact_predictor)


def main(_):
  logging.init(logtostderr=True, logtofile=False)
  evaluate_score()

if __name__ == '__main__':
  tf.app.run()
