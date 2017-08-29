#!/usr/bin/env python
# -*- coding: gbk -*-
# ==============================================================================
#          \file   train.py
#        \author   chenghuige  
#          \date   2016-08-16 14:05:38.743467
#   \Description  
# ==============================================================================
"""
@TODO using logging ?
NOTICE using feed_dict --feed_neg=1 will slow *10
@TODO why evaluate will use mem up to 13% then down to 5%?
using tf.nn.top_k do sort in graph still evaluate large mem same as before
20w will using 1.6G 32 * 5%
50w 9.1%
200w 16G?
2000w 160g?

@TODO why using gpu will use more cpu mem?
50w keyword MAX_TEXT_WORDS 20
cpu version: 
show_eval=0  train 6.7% eval 11.4%
show_eval=1 train 8% eval 20%

gpu version: 
show_eval=0 train 8% eval 12.8%
show_eval=1 train 35% eval 39%

text means text ids
text_str means orit text str
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('model_dir', '/home/gezi/temp/textsum/model', '')

flags.DEFINE_string('algo', 'seq2seq', 'default algo is bow(cbow), also support rnn, show_and_tell, TODO cnn')

flags.DEFINE_string('vocab', '/home/gezi/temp/textsum/tfrecord/seq-basic.10w/train/vocab.txt', 'vocabulary file')

flags.DEFINE_boolean('debug', False, '')

import sys, math
import functools
import gezi
import melt
logging = melt.logging

from deepiu.textsum import input_app as InputApp

from deepiu.textsum import eval_show
from deepiu.util import evaluator
from deepiu.util import algos_factory

#debug
from deepiu.util import vocabulary
from deepiu.util import text2ids

from deepiu.textsum.algos import seq2seq
from deepiu.seq2seq.rnn_decoder import SeqDecodeMethod

sess = None

#TODO do not consider feed dict support right now for tower loss!
def tower_loss(trainer, input_app=None, input_results=None):
  if input_app is None:
    input_app = InputApp.InputApp()
  if input_results is None:
    input_results = input_app.gen_input(train_only=True)

  #--------train
  image_name, text, text_str, input_text, input_text_str = input_results[input_app.input_train_name]

  loss = trainer.build_train_graph(input_text, text)
  return loss

def gen_train_graph(input_app, input_results, trainer):
  """
    main flow, key graph
  """
  #--- if you don't want to use mutli gpu, here just for safe(code same with old single gpu cod)
  if FLAGS.num_gpus == 0:
    loss = tower_loss(trainer, input_app, input_results)
  else:
    loss_function = lambda: tower_loss(trainer)
    #here loss is a list of losses
    loss = melt.tower_losses(loss_function, FLAGS.num_gpus)
    print('num tower losses:', len(loss))

  ops = [loss]
  #--------mark train graph finished, all graph after must share variable from train graph
  #melt.reuse_variables()
  trainer.is_training = False
    
  deal_debug_results = None

  #FLAGS.debug = True
  if FLAGS.debug == True:
    #ops += [tf.get_collection('scores')[-1], tf.get_collection('encode_feature')[-1], tf.get_collection('encode_state')[-1]]
    #ops += [tf.get_collection('debug_seqeuence')[-1], tf.get_collection('debug_length')[-1]]
    ops += [tf.get_collection('logits')[-1]]
    def _deal_debug_results(results):
      print(results)
      #_, seq, len = results 
      #for item in seq[0]:
      #  print(item)
      #print('len:', len[0])
      #_, scores, encode_feature, encode_state = results
      #print('scores', scores)
      #print('encode_feature', encode_feature)  
      #print('encode_state', encode_state)      

    deal_debug_results = _deal_debug_results

  return ops, deal_debug_results

def gen_train(input_app, input_results, trainer):
  ops, deal_debug_results = gen_train_graph(input_app, input_results, trainer)
  
  def _deal_results(results):
    melt.print_results(results, ['batch_loss'])

    if deal_debug_results is not None:
      debug_results = results
      deal_debug_results(debug_results)

  deal_results = _deal_results

  return ops, None, deal_results

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

def gen_validate(input_app, input_results, trainer, predictor):
  eval_ops = None
  train_with_validation = input_results[input_app.input_valid_name] is not None
  deal_eval_results = None
  if train_with_validation and not FLAGS.train_only:
    eval_image_name, eval_text, eval_text_str, eval_input_text, eval_input_text_str = input_results[input_app.input_valid_name]

    eval_loss = trainer.build_train_graph(eval_input_text, eval_text)
    eval_scores = tf.get_collection('scores')[-1]
    eval_ops = [eval_loss]

    if FLAGS.show_eval and (predictor is not None):
      eval_ops, deal_eval_results = \
        gen_evalulate(
            input_app, 
            input_results, 
            predictor, 
            eval_ops, 
            eval_scores)
    else:
      deal_eval_results = lambda x: melt.print_results(x, ['eval_batch_loss'])

  return eval_ops, None, deal_eval_results

def gen_predict_graph(predictor):  
  exact_score, exact_ori_score = predictor.init_predict(exact_loss=True)
  tf.add_to_collection('exact_score', exact_score)
  tf.add_to_collection('exact_ori_score', exact_ori_score)

  exact_prob, exact_ori_prob = predictor.init_predict(exact_prob=True)
  tf.add_to_collection('exact_prob', exact_prob)
  tf.add_to_collection('exact_ori_prob', exact_ori_prob)

  #put to last since evaluate use get collection from 'scores'[-1]
  score, ori_score = predictor.init_predict()
  #by default will be averaged score per time step
  tf.add_to_collection('score', score)
  #also record ori score not averaged per time step
  tf.add_to_collection('ori_score', ori_score)

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
  #if FLAGS.use_attention:
  #  tf.add_to_collection('beam_search_alignments', tf.get_collection('attention_alignments')[-1])

  return beam_text, beam_text_score

#step = 0
def train_process(trainer, predictor=None):
  input_app = InputApp.InputApp()
  input_results = input_app.gen_input()

  with tf.variable_scope(FLAGS.main_scope) as scope:
    ops, gen_feed_dict, deal_results = gen_train(
      input_app, 
      input_results, 
      trainer)
    scope.reuse_variables()

    if predictor is not None and FLAGS.gen_predict:
      beam_text, beam_text_score = gen_predict_graph(predictor)

    eval_ops, gen_eval_feed_dict, deal_eval_results = gen_validate(
      input_app, 
      input_results, 
      trainer, 
      predictor)

    metric_eval_fn = None
    if FLAGS.metric_eval:
      #generative can do this also but it is slow so just ingore this
      if not algos_factory.is_generative(FLAGS.algo): 
        metric_eval_fn = lambda: evaluator.evaluate_scores(predictor, random=True)

  if FLAGS.mode == 'train':
    melt.print_global_varaiables()
    melt.apps.train_flow(ops, 
                         gen_feed_dict_fn=gen_feed_dict,
                         deal_results_fn=deal_results,
                         eval_ops=eval_ops,
                         gen_eval_feed_dict_fn=gen_eval_feed_dict,
                         deal_eval_results_fn=deal_eval_results,
                         optimizer=FLAGS.optimizer,
                         learning_rate=FLAGS.learning_rate,
                         num_steps_per_epoch=input_app.num_steps_per_epoch,
                         model_dir=FLAGS.model_dir,
                         metric_eval_fn=metric_eval_fn,
                         sess=sess)#notice if use melt.constant in predictor then must pass sess
  else: #test predict
    predictor.load(FLAGS.model_dir)
    import conf  
    from conf import TEXT_MAX_WORDS, INPUT_TEXT_MAX_WORDS, NUM_RESERVED_IDS, ENCODE_UNK

    #TODO: now copy from prpare/gen-records.py
    def _text2ids(text, max_words):
      word_ids = text2ids.text2ids(text, 
                                   seg_method=FLAGS.seg_method, 
                                   feed_single=FLAGS.feed_single, 
                                   allow_all_zero=True, 
                                   pad=False)
      word_ids_length = len(word_ids)
      word_ids = word_ids[:max_words]
      word_ids = gezi.pad(word_ids, max_words, 0)
      return word_ids

    input_texts = [
                   #'����������һ�Ը�Ů�ڿ������ջ�͸����˿¶�δ����������ڿ�Ů��-�Ա���',
                   '����������ʵ��С��ô��,����������ʵ��С���δ�ʩ',
                   ]

    for input_text in input_texts:
      word_ids = _text2ids(input_text, INPUT_TEXT_MAX_WORDS)
      print('word_ids', word_ids, 'len:', len(word_ids))
      print(text2ids.ids2text(word_ids))
      #similar as inference.py this is only ok for no attention mode TODO FIXME
      texts, scores = sess.run([tf.get_collection('text')[0], tf.get_collection('text_score')[0]], 
                             feed_dict={'seq2seq/model_init_1/input_text:0' : [word_ids]})
      print(texts[0], text2ids.ids2text(texts[0]), scores[0])

      texts, scores  = sess.run([beam_text, beam_text_score], 
                               feed_dict={predictor.input_text_feed: [word_ids]})

      texts = texts[0]
      scores = scores[0]
      for text, score in zip(texts, scores):
        print(text, text2ids.ids2text(text), score)
    
    input_texts = [
                   '����������ʵ��С��ô��,����������ʵ��С���δ�ʩ',
                   #'����������һ�Ը�Ů�ڿ������ջ�͸����˿¶�δ����������ڿ�Ů��-�Ա���',
                   "����̫����ô����",
                   '����������ʵ��С��ô��,����������ʵ��С���δ�ʩ',
                   #'����������ʵ��С��ô��,����������ʵ��С���δ�ʩ',
                   #'�޺콨�ǰ���˹��',
                   ]

    word_ids_list = [_text2ids(input_text, INPUT_TEXT_MAX_WORDS) for input_text in input_texts]
    timer = gezi.Timer()
    texts_list, scores_list = sess.run([beam_text, beam_text_score], 
                               feed_dict={predictor.input_text_feed: word_ids_list})
    
    for texts, scores in zip(texts_list, scores_list):
      for text, score in zip(texts, scores):
        print(text, text2ids.ids2text(text), score, math.log(score))

    print('beam_search using time(ms):', timer.elapsed_ms())



def train():
  trainer, predictor =  algos_factory.gen_trainer_and_predictor(FLAGS.algo)

  logging.info('trainer:{}'.format(trainer))
  logging.info('predictor:{}'.format(predictor))
  train_process(trainer, predictor)

def main(_):
  #-----------init global resource
  logging.set_logging_path(gezi.get_dir(FLAGS.model_dir))

  InputApp.init()
  vocabulary.init()
  text2ids.init()
  
  #evaluator.init()

  logging.info('algo:{}'.format(FLAGS.algo))
  logging.info('monitor_level:{}'.format(FLAGS.monitor_level))
  
  global_scope = ''
  if FLAGS.add_global_scope:
    global_scope = FLAGS.global_scope if FLAGS.global_scope else FLAGS.algo
 
  global sess
  sess = melt.get_session(log_device_placement=FLAGS.log_device_placement)
  with tf.variable_scope(global_scope):
    train()
 
if __name__ == '__main__':
  tf.app.run()
