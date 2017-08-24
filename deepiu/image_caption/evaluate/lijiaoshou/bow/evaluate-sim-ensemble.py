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

flags.DEFINE_string('algo', 'bow', 'bow, rnn, show_and_tell')
flags.DEFINE_string('model_dir', '/home/gezi/new/temp/image-caption/lijiaoshou/model/bow.basic.3neg.all/', '')  

#flags.DEFINE_boolean('add_global_scope', True, '''default will add global scope as algo name,
#                                                set to False incase you want to load some old model without algo scope''')

#flags.DEFINE_string('global_scope', '', '')

flags.DEFINE_string('vocab', '/home/gezi/new/temp/image-caption/lijiaoshou/tfrecord/seq-basic/vocab.txt', 'vocabulary binary file')

flags.DEFINE_boolean('print_predict', False, '')
#flags.DEFINE_string('out_file', 'sim-result.html', '')

flags.DEFINE_string('image_feature_name_', 'bow/main/image_feature:0', 'model_init_1 because predictor after trainer init')
flags.DEFINE_string('text_name', 'bow/main/text:0', 'model_init_1 because predictor after trainer init')

import sys 
import numpy as np
import melt
logging = melt.logging
import gezi

from deepiu.util import evaluator
from deepiu.util import algos_factory

class Predictor:
  def __init__(self, model_dir):
    #self._predictor = melt.Predictor('/home/gezi/new/temp/image-caption/lijiaoshou/model/bow.basic.3neg.all/')
    self._predictor = melt.Predictor('/home/gezi/new/temp/image-caption/lijiaoshou/model/bow.basic.negtext/')
    self.score = tf.get_collection('score')[-1]
    self._predictor2 = melt.Predictor('/home/gezi/new/temp/image-caption/lijiaoshou/model/cnn')
    self.score2 = tf.get_collection('score')[-1]

  def bulk_predict(self, image_feature, text_feature):
    score = self._predictor.inference(self.score, 
                            feed_dict= {
                              'bow/main/image_feature:0': image_feature,
                              'bow/main/text:0': text_feature,
                                    })
    score2 = self._predictor2.inference(self.score2, 
                            feed_dict= {
                              'cnn/main/image_feature:0': image_feature,
                              'cnn/main/text:0': text_feature
                                    })


    return (score + score2) / 2.0

def evaluate_score():
  evaluator.init()
  text_max_words = evaluator.all_distinct_texts.shape[1]
  print('text_max_words:', text_max_words)
  predictor = Predictor(FLAGS.model_dir)
  evaluator.evaluate_scores(predictor, random=True)


def main(_):
  logging.init(logtostderr=True, logtofile=False)
  evaluate_score()

if __name__ == '__main__':
  tf.app.run()
