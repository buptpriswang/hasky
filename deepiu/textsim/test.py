#!/usr/bin/env python
# ==============================================================================
#          \file   test-melt.py
#        \author   chenghuige  
#          \date   2016-08-17 15:34:45.456980
#   \Description  
# ==============================================================================
"""
train_input from input_app now te test_input 
valid_input set empty
python ./test.py --train_input test_input --valid_input ''
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('algo', 'seq2seq', '/home/gezi/temp/textsum/model.seq2seq/')
flags.DEFINE_string('model_dir', './model.flickr.show_and_tell2/', '')  
flags.DEFINE_string('vocab', '/home/gezi/temp/textsum/tfrecord/seq-basic/train/vocab.txt', 'vocabulary file')

flags.DEFINE_integer('num_interval_steps', 100, '')
flags.DEFINE_integer('eval_times', 0, '')

#flags.DEFINE_integer('monitor_level', 2, '1 will monitor emb, 2 will monitor gradient')


import sys
import functools

from deepiu.util import vocabulary
from deepiu.util import text2ids

from deepiu.textsum.algos import seq2seq
from deepiu.seq2seq.rnn_decoder import SeqDecodeMethod

import melt
test_flow = melt.flow.test_flow
import input_app as InputApp
logging = melt.utils.logging

from deepiu.util import algos_factory

from deepiu.textsum import eval_show

def gen_evalulate(input_app, 
                  input_results,
                  predictor, 
                  eval_ops, 
                  eval_scores):
  eval_ops += eval_show.gen_eval_generated_texts_ops(
        input_app, 
        input_results, 
        predictor, 
        eval_scores)
  deal_eval_results = eval_show.deal_eval_generated_texts_results
  return eval_ops, deal_eval_results

def gen_predict_graph(predictor):  
  exact_score = predictor.init_predict(exact_loss=True)
  tf.add_to_collection('exact_score', exact_score)

  exact_prob = predictor.init_predict(exact_prob=True)
  tf.add_to_collection('exact_prob', exact_prob)

  #put to last since evaluate use get collection from 'scores'[-1]
  score = predictor.init_predict()
  tf.add_to_collection('score', score)

  #-----generateive
  print('beam_size', FLAGS.beam_size)
  init_predict_text = functools.partial(predictor.init_predict_text, 
                                        beam_size=FLAGS.beam_size, 
                                        convert_unk=False)
  text, text_score = init_predict_text(decode_method=FLAGS.seq_decode_method)
  beam_text, beam_text_score = init_predict_text(decode_method=SeqDecodeMethod.beam)
      
  tf.add_to_collection('text', text)
  tf.add_to_collection('text_score', text_score)
  tf.add_to_collection('beam_text', beam_text)          
  tf.add_to_collection('beam_text_score', beam_text_score)          

  init_predict_text(decode_method=SeqDecodeMethod.beam_search)
  if FLAGS.use_attention:
    tf.add_to_collection('beam_search_alignments', tf.get_collection('attention_alignments')[-1])

  return beam_text, beam_text_score

def test():
  trainer, predictor = algos_factory.gen_trainer_and_predictor(FLAGS.algo)

  trainer.is_training = False

  input_app = InputApp.InputApp()
  sess = input_app.sess = tf.InteractiveSession()

  input_results = input_app.gen_input()
  with tf.variable_scope(FLAGS.main_scope) as scope:
    eval_image_name, eval_text, eval_text_str, eval_input_text, eval_input_text_str = input_results[input_app.input_valid_name]
  
    eval_loss = trainer.build_train_graph(eval_input_text, eval_text)

    scope.reuse_variables()
    gen_predict_graph(predictor)

    print('---------------', tf.get_collection('scores'))
    
    eval_scores = tf.get_collection('scores')[-1]

    eval_names = ['loss']
    print('eval_names:', eval_names)


    print('gen_validate-------------------------', tf.get_collection('scores'))
    eval_ops = [eval_loss]
  
    eval_ops, deal_eval_results = \
         gen_evalulate(
              input_app, 
              input_results, 
              predictor, 
              eval_ops, 
              eval_scores)
  
  test_flow(
    eval_ops, 
    names=eval_names,
    gen_feed_dict=None,
    deal_results=deal_eval_results,
    model_dir=FLAGS.model_dir, 
    num_interval_steps=FLAGS.num_interval_steps,
    num_epochs=FLAGS.num_epochs,
    eval_times=FLAGS.eval_times,
    sess=sess)

def main(_):
  logging.init(logtostderr=True, logtofile=False)
  global_scope = ''
  
  InputApp.init()
  vocabulary.init()
  text2ids.init()
  
  if FLAGS.add_global_scope:
    global_scope = FLAGS.global_scope if FLAGS.global_scope else FLAGS.algo
  with tf.variable_scope(global_scope):
    test()

if __name__ == '__main__':
  tf.app.run()

  
