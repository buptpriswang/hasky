#!/usr/bin/env python
# -*- coding: gbk -*-
# ==============================================================================
#          \file   predict.py
#        \author   chenghuige  
#          \date   2016-10-19 06:54:26.594835
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('model_dir', '/home/gezi/temp/textsum/model.seq2seq/', '')
flags.DEFINE_string('vocab', '/home/gezi/temp/textsum/tfrecord/seq-basic.10w/train/vocab.txt', 'vocabulary file')
flags.DEFINE_boolean('debug', False, '')

flags.DEFINE_string('text_file', '/home/gezi/data/lijiaoshou/wenan.txt', '')

import sys, os, math
import gezi, melt
import numpy as np

from deepiu.util import text2ids

import conf  
from conf import TEXT_MAX_WORDS, INPUT_TEXT_MAX_WORDS, NUM_RESERVED_IDS, ENCODE_UNK

#TODO: now copy from prpare/gen-records.py
def _text2ids(text, max_words):
  word_ids = text2ids.text2ids(text, 
                               seg_method='basic', 
                               feed_single=True, 
                               allow_all_zero=True, 
                               pad=False)
  word_ids = word_ids[:max_words]
  word_ids = gezi.pad(word_ids, max_words, 0)

  return word_ids

def predict(predictor, input_text):
  word_ids = _text2ids(input_text, INPUT_TEXT_MAX_WORDS)
  print('word_ids', word_ids, 'len:', len(word_ids))
  print(text2ids.ids2text(word_ids))

  #tf.while_loop has debug problem ValueError: Causality violated in timing relations of debug dumps: seq2seq/main/decode_4/dynamic_rnn_decoder/rnn/while/Merge_7 (1489649052260629): these input(s) are not satisfied: [(u'seq2seq/main/decode_4/dynamic_rnn_decoder/rnn/while/Enter_7', 0), (u'seq2seq/main/decode_4/dynamic_rnn_decoder/rnn/while/NextIteration_7', 0)
  #https://github.com/tensorflow/tensorflow/issues/8337 From your error message, it appears that you are using tf.while_loop. Can you try setting its paralle_iterations parameter to 1 and see if the error still happens?
  #There may be a bug in how tfdbg handles while_loops with parallel_iterations > 1.
  #I think it might be a GPU thing.
  #The example below errors if run as python tf_8337_minimal.py but is fine is run as CUDA_VISIBLE_DEVICES=-1 
  timer = gezi.Timer()
  text, score = predictor.inference(['text', 'text_score'], 
                                    feed_dict= {
                                      'seq2seq/model_init_1/input_text:0': [word_ids]
                                      })
  
  for result in text:
    print(result, text2ids.ids2text(result), 'decode time(ms):', timer.elapsed_ms())
  
  timer = gezi.Timer()
  texts, scores = predictor.inference(['beam_text', 'beam_text_score'], 
                                    feed_dict= {
                                      'seq2seq/model_init_1/input_text:0': [word_ids]
                                      })

  texts = texts[0]
  scores = scores[0]
  for text, score in zip(texts, scores):
    print(text, text2ids.ids2text(text), score)

  print('beam_search using time(ms):', timer.elapsed_ms())


def predicts(predictor, input_texts):
  word_ids_list = [_text2ids(input_text, INPUT_TEXT_MAX_WORDS) for input_text in input_texts]
  timer = gezi.Timer()
  texts_list, scores_list = predictor.inference(['beam_text', 'beam_text_score'], 
                                    feed_dict= {
                                      'seq2seq/model_init_1/input_text:0': word_ids_list
                                      })

  for texts, scores in zip(texts_list, scores_list):
    for text, score in zip(texts, scores):
      if text[0] == text2ids.vocab.unk_id():
        continue
      print(''.join(text2ids.ids2words(text, print_end=False)))
      break

  print('beam_search using time(ms):', timer.elapsed_ms())

def main(_):
  text2ids.init()
  predictor = melt.Predictor(FLAGS.model_dir, debug=FLAGS.debug)

  input_texts = [x.strip() for x in open(FLAGS.text_file)]
  predicts(predictor, input_texts)

if __name__ == '__main__':
  tf.app.run()
