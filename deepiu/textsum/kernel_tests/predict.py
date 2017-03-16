#!/usr/bin/env python
# -*- coding: gbk -*-
# ==============================================================================
#          \file   predict.py
#        \author   chenghuige  
#          \date   2016-09-02 10:52:36.566367
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS
  
flags.DEFINE_string('model_dir', '/home/gezi/temp/textsum/model.seq2seq.attention/', '')

flags.DEFINE_string('algo', 'seq2seq', 'default algo is bow(cbow), also support rnn, show_and_tell, TODO cnn')

flags.DEFINE_string('vocab', '/home/gezi/temp/textsum/tfrecord/seq-basic.10w/train/vocab.txt', 'vocabulary file')

#----------strategy 
#flags.DEFINE_integer('beam_size', 10, 'for seq decode beam search size')

import numpy as np 
import math

import gezi
import melt
logging = melt.logging

from deepiu.image_caption.algos import algos_factory
from deepiu.seq2seq.rnn_decoder import SeqDecodeMethod

#debug
from deepiu.util import text2ids

from conf import INPUT_TEXT_MAX_WORDS

#TODO: now copy from prpare/gen-records.py
def _text2ids(text, max_words):
  word_ids = text2ids.text2ids(text, 
                               seg_method=FLAGS.seg_method, 
                               feed_single=FLAGS.feed_single, 
                               allow_all_zero=True, 
                               pad=False)

  word_ids = word_ids[:max_words]
  word_ids = gezi.pad(word_ids, max_words, 0)

  return word_ids

def main(_):
  text2ids.init()
  global_scope = ''
  if FLAGS.add_global_scope:
    global_scope = FLAGS.global_scope if FLAGS.global_scope else FLAGS.algo
 
  sess = melt.get_session(log_device_placement=FLAGS.log_device_placement)
  with tf.variable_scope(global_scope):
    predictor =  algos_factory.gen_predictor(FLAGS.algo)
    with tf.variable_scope(FLAGS.main_scope) as scope:
      ##--notice if not add below len(tf.get_collection('encode_state') is 1, add below will be 2
      ## even though in init_predict_text(decode_method=SeqDecodeMethod.beam) will call generate_sequence_greedy
      #text, score = predictor.init_predict_text(decode_method=SeqDecodeMethod.greedy, 
      #                                          beam_size=FLAGS.beam_size,
      #                                          convert_unk=False)   
      #scope.reuse_variables()
      beam_text, beam_score = predictor.init_predict_text(decode_method=SeqDecodeMethod.beam, 
                                                          beam_size=FLAGS.beam_size,
                                                          convert_unk=False)  

  predictor.load(FLAGS.model_dir) 
  #input_text = "������������_��������ǰ��Ա���Ƭ"
  input_texts = [
                 #'���������һ�Ը�Ů�ڿ�����ջ�͸����˿¶�δ���������ڿ�Ů��-�Ա���',
                 #'����������ʵ��С��ô��,����������ʵ��С���δ�ʩ',
                 #'����̫����ô����',
                 #'����ף�Ŀǰ4����1�굶',
                 '����������ʵ��С��ô��,����������ʵ��С���δ�ʩ',
                 ]

  for input_text in input_texts:
    word_ids = _text2ids(input_text, INPUT_TEXT_MAX_WORDS)

    print(word_ids)
    print(text2ids.ids2text(word_ids))

    #timer = gezi.Timer()
    #text_, score_ = sess.run([text, score], {predictor.input_text_feed : [word_ids]})
    #print(text_[0], text2ids.ids2text(text_[0]), score_[0], 'time(ms):', timer.elapsed_ms())

    timer = gezi.Timer()
    print(tf.get_collection('encode_state'), len(tf.get_collection('encode_state')))
    texts, scores, state, input_ , atkeys, atvalues = sess.run([beam_text, beam_score, 
                                             tf.get_collection('seq2seq_encode_state')[0],
                                             tf.get_collection('seq2seq_input')[0],
                                             tf.get_collection('attention_keys')[0],
                                             tf.get_collection('attention_values')[0],
                                             ], 
                                            {predictor.input_text_feed : [word_ids]})

    #print(state)
    #print(input_)

    print(atkeys)
    #print(atvalues)

    texts = texts[0]
    scores = scores[0]
    for text_, score_ in zip(texts, scores):
      print(text_, text2ids.ids2text(text_), score_)

    print('beam_search using time(ms):', timer.elapsed_ms())

  input_texts = [
                 '����������ʵ��С��ô��,����������ʵ��С���δ�ʩ',
                 #'���������һ�Ը�Ů�ڿ�����ջ�͸����˿¶�δ���������ڿ�Ů��-�Ա���',
                 '���������һ�Ը�Ů�ڿ�����ջ�͸����˿¶�δ����',
                 #"����̫����ô����",
                 #'����������ʵ��С��ô��,����������ʵ��С���δ�ʩ',
                 #'����������ʵ��С��ô��,����������ʵ��С���δ�ʩ',
                 #'�޺콨�ǰ���˹��',
                 ]

  word_ids_list = [_text2ids(input_text, INPUT_TEXT_MAX_WORDS) for input_text in input_texts]
  timer = gezi.Timer()
  texts_list, scores_list, state, input_, atkeys, atvalues = sess.run([beam_text, beam_score, 
                                             tf.get_collection('seq2seq_encode_state')[0],
                                             tf.get_collection('seq2seq_input')[0],
                                             tf.get_collection('attention_keys')[0],
                                             tf.get_collection('attention_values')[0],
                                             ], 
                             feed_dict={predictor.input_text_feed: word_ids_list})
  
  #print(state)
  #print(input_)
  print(atkeys)
  print(np.shape(atkeys))
  #print(atvalues)

  for texts, scores in zip(texts_list, scores_list):
    for text, score in zip(texts, scores):
      print(text, text2ids.ids2text(text), score, math.log(score))

  print('beam_search using time(ms):', timer.elapsed_ms())

if __name__ == '__main__':
  tf.app.run()