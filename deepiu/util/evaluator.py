#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   evaluator.py
#        \author   chenghuige  
#          \date   2016-08-22 13:03:44.170552
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('valid_resource_dir', '/home/gezi/new/temp/image-caption/lijiaoshou/tfrecord/seq-basic/valid/', '')
#--if use image dir already info in image_features
flags.DEFINE_string('image_dir', None, 'input images dir')
#----------label fie dprecated
flags.DEFINE_string('label_file', '/home/gezi/data/image-caption/flickr/test/results_20130124.token', '')
flags.DEFINE_string('image_feature_file', '/home/gezi/data/image-caption/flickr/test/img2fea.txt', '')

flags.DEFINE_string('assistant_model_dir', None, 'only this is used for assistant model')
flags.DEFINE_string('assistant_algo', None, '')
flags.DEFINE_string('assistant_key', 'score', '')
flags.DEFINE_integer('assistant_rerank_num', 100, '')
#------depreciated
flags.DEFINE_string('assistant_ltext_key', 'dual_bow/main/ltext:0', '')
flags.DEFINE_string('assistant_rtext_key', 'dual_bow/main/rtext:0', '')

flags.DEFINE_string('img2text', '', 'img id to text labels ids')
flags.DEFINE_string('text2img', '', 'text id to img labels ids')

flags.DEFINE_string('image_name_bin', '', 'image names')
flags.DEFINE_string('image_feature_bin', '', 'image features')

flags.DEFINE_string('text2id', '', 'not used')
flags.DEFINE_string('img2id', '', 'not used')

flags.DEFINE_integer('num_metric_eval_examples', 1000, '')
flags.DEFINE_integer('metric_eval_batch_size', 1000, '')
flags.DEFINE_integer('metric_eval_texts_size', 0, ' <=0 means not limit')
flags.DEFINE_integer('metric_eval_images_size', 0, ' <=0 means not limit')
flags.DEFINE_integer('metric_topn', 100, 'only consider topn results when calcing metrics')

flags.DEFINE_integer('max_texts', 200000, '') 
flags.DEFINE_integer('max_images', 20000, '') 

flags.DEFINE_boolean('eval_img2text', True, '')
flags.DEFINE_boolean('eval_text2img', False, '')

import sys, os
import gezi.nowarning

import gezi
import melt
logging = melt.logging

from deepiu.util import vocabulary
vocab = None
vocab_size = None

from deepiu.util import text2ids
from deepiu.util.text2ids import ids2words, ids2text, texts2ids

try:
  import conf 
  from conf import TEXT_MAX_WORDS, IMAGE_FEATURE_LEN
except Exception:
  print('Warning, no conf.py in current path use util conf')
  from deepiu.util.conf import TEXT_MAX_WORDS, IMAGE_FEATURE_LEN

from deepiu.util import algos_factory

import numpy as np
import math

all_distinct_texts = None
all_distinct_text_strs = None

img2text = None
text2img = None

image_names = None
image_features = None

assistant_predictor = None

inited = False

def init():
  global inited

  if inited:
    return

  #for evaluation without train will also use evaluator so only set log path in train.py
  #logging.set_logging_path(FLAGS.model_dir)
  if FLAGS.assistant_model_dir:
    global assistant_predictor
    #assistant_predictor = algos_factory.gen_predictor(FLAGS.assistant_algo)
    #melt.restore_scope_from_path(melt.get_session(), FLAGS.assistant_model_dir, FLAGS.assistant_algo)
    ##try another session no work... so same session graph
    #assistant_predictor = melt.SimPredictor(FLAGS.assistant_model_dir, sess=tf.Session())
    #--since add 'score'... will confuse, just remove it.. hack!
    assistant_predictor = melt.SimPredictor(FLAGS.assistant_model_dir, key='assistant_score', index=0)
    melt.rename_from_collection('score', 'assistant_score')
    melt.rename_from_collection('scores', 'assistant_scores')
    print('assistant_predictor', assistant_predictor)

  test_dir = FLAGS.valid_resource_dir
  global all_distinct_texts, all_distinct_text_strs
  global vocab, vocab_size
  if all_distinct_texts is None:
    print('loading valid resorce from:', test_dir)
    #vocabulary.init()
    text2ids.init()
    vocab = vocabulary.vocab
    vocab_size = vocabulary.vocab_size
    
    if os.path.exists(test_dir + '/distinct_texts.npy'):
      all_distinct_texts = np.load(test_dir + '/distinct_texts.npy')
    else:
      all_distinct_texts = []
    
    #to avoid outof gpu mem
    #all_distinct_texts = all_distinct_texts[:FLAGS.max_texts]
    print('all_distinct_texts len:', len(all_distinct_texts), file=sys.stderr)
    
    #--padd it as test data set might be smaller in shape[1]
    all_distinct_texts = np.array([gezi.nppad(text, TEXT_MAX_WORDS) for text in all_distinct_texts])
    if FLAGS.feed_dict:
      all_distinct_texts = texts2ids(evaluator.all_distinct_text_strs)
    if os.path.exists(test_dir + '/distinct_text_strs.npy'):
      all_distinct_text_strs = np.load(test_dir + '/distinct_text_strs.npy')
    else:
      all_distinct_text_strs = []

    init_labels()

    inited = True

def init_labels():
  get_bidrectional_lable_map()
  get_bidrectional_lable_map_txt2im()

  get_image_names_and_features()

def get_bidrectional_lable_map():
  global img2text
  if img2text is None:
    img2text_path = os.path.join(FLAGS.valid_resource_dir, 'img2text.npy')
    img2text = np.load(img2text_path).item()
  return img2text 

def get_bidrectional_lable_map_txt2im():
  global text2img
  if text2img is None:
    text2img_path = os.path.join(FLAGS.valid_resource_dir, 'text2img.npy')
    text2img = np.load(text2img_path).item()
  return text2img

def hack_image_features(image_features):
  """
  the hack is for textsim use ltext as image(similar), so hack for it
  """
  #first for real image but not dump feature, use original encoded image since big, we assume
  #pre save binary pics and can refer to pic in disk by pic name and pic dir
  assert len(image_features) > 0
  if isinstance(image_features[0], np.string_):
    #return np.array([melt.read_image(pic_path) for pic_path in image_features])
    return image_features
  try:
    if len(image_features[0]) == IMAGE_FEATURE_LEN and len(image_features[1]) == IMAGE_FEATURE_LEN:
      return image_features 
    else:
      return np.array([gezi.nppad(x, TEXT_MAX_WORDS) for x in image_features])
  except Exception:
    return  np.array([gezi.nppad(x, TEXT_MAX_WORDS) for x in image_features])

def get_image_names_and_features():
  global image_names, image_features
  if image_names is None:
    image_feature_bin = os.path.join(FLAGS.valid_resource_dir, 'distinct_image_features.npy')
    image_name_bin = os.path.join(FLAGS.valid_resource_dir, 'distinct_image_names.npy')
    timer = gezi.Timer('get_image_names_and_features')
    image_names = np.load(image_name_bin)
    image_features = np.load(image_feature_bin)
    image_features = hack_image_features(image_features)
    print('all_distinct_images len:', len(image_features), file=sys.stderr)
    timer.print()
  return image_names, image_features

head_html = '<html><meta http-equiv="Content-Type" content="text/html; charset=utf-8"><body>'
tail_html = '</body> </html>'

img_html = '<p><a href={0} target=_blank><img src={0} height=200></a></p>\n {1} {2} epoch:{3}, step:{4}, train:{5}, eval:{6}, duration:{7}, {8}'
content_html = '<p> {} </p>'

import numpy as np

def print_neareast_texts(scores, num=20, img = None):
  indexes = (-scores).argsort()[:num]
  for i, index in enumerate(indexes):
    used_words = ids2words(all_distinct_texts[index])
    line = ' '.join([str(x) for x in ['%d:['%i,all_distinct_text_strs[index], ']', "%.6f"%scores[index], len(used_words), '/'.join(used_words)]])
    logging.info(content_html.format(line))

def print_neareast_words(scores, num=50):
  indexes = (-scores).argsort()[:num]
  line = ' '.join(['%s:%.6f'%(vocab.key(index), scores[index]) for index in indexes])
  logging.info(content_html.format(line))

def print_neareast_texts_from_sorted(scores, indexes, img = None):
  for i, index in enumerate(indexes):
    used_words = ids2words(all_distinct_texts[index])
    predict_result = ''
    if img:
      init_labels()
      if img in img2text:
        hits = img2text[img]
        predict_result = 'er%d'%i if index not in img2text[img] else 'ok%d'%i
      else:
        predict_result = 'un%d'%i 
    #notice may introduce error, offline scores is orinal scores! so need scores[index] but online input is max_scores will need scores[i]
    if len(scores) == len(indexes):
      line = ' '.join([str(x) for x in [predict_result, '[', all_distinct_text_strs[index], ']', "%.6f"%scores[i], len(used_words), '/'.join(used_words)]])
    else:
      line = ' '.join([str(x) for x in [predict_result, '[', all_distinct_text_strs[index], ']', "%.6f"%scores[index], len(used_words), '/'.join(used_words)]])
    logging.info(content_html.format(line))

def print_neareast_words_from_sorted(scores, indexes):
  if len(scores) == len(indexes):
    line = ' '.join(['%s:%.6f'%(vocab.key(int(index)), scores[i]) for i, index in enumerate(indexes)])
  else:
    line = ' '.join(['%s:%.6f'%(vocab.key(int(index)), scores[index]) for i, index in enumerate(indexes)])
  logging.info(content_html.format(line))

def print_img(img, i):
  img_url = os.path.join(FLAGS.image_dir, img) if not img.startswith("http://") else img
  logging.info(img_html.format(
    img_url, 
    i, 
    img, 
    melt.epoch(), 
    melt.step(), 
    melt.train_loss(), 
    melt.eval_loss(),
    melt.duration(),
    gezi.now_time()))

def print_img_text(img, i, text):
  print_img(img, i)
  logging.info(content_html.format(text))

def print_img_text_score(img, i, text, score):
  print_img(img, i)
  logging.info(content_html.format('{}:{}'.format(text, score)))

def print_img_text_negscore(img, i, text, score, text_ids, neg_text=None, neg_score=None, neg_text_ids=None):
  print_img(img, i)
  text_words = ids2text(text_ids)
  if neg_text is not None:
    neg_text_words = ids2text(neg_text_ids)
  logging.info(content_html.format('pos [ {} ] {:.6f} {}'.format(text, score, text_words)))
  if neg_text is not None:
    logging.info(content_html.format('neg [ {} ] {:.6f} {}'.format(neg_text, neg_score, neg_text_words)))  

#for show and tell 
def print_generated_text(generated_text, id=-1, name='gen'):
  if id >= 0:
    logging.info(content_html.format('{}_{}:[ {} ]'.format(name, id, ids2text(generated_text))))
  else:
    logging.info(content_html.format('{}:[ {} ]'.format(name, ids2text(generated_text))))

def print_generated_text_score(generated_text, score, id=-1, name='gen'):
  if id >= 0:
    logging.info(content_html.format('{}_{}:[ {} ] {:.6f}'.format(name, id, ids2text(generated_text), score)))
  else:
    logging.info(content_html.format('{}:[ {} ] {:.6f}'.format(name, ids2text(generated_text), score)))

def print_img_text_negscore_generatedtext(img, i, text, score,
                                          text_ids,  
                                          generated_text, generated_text_score,
                                          generated_text_beam=None, generated_text_score_beam=None,
                                          neg_text=None, neg_score=None, neg_text_ids=None):
  score = math.exp(-score)
  print_img_text_negscore(img, i, text, score, text_ids, neg_text, neg_score, neg_text_ids)
  try:
    print_generated_text_score(generated_text, generated_text_score)
  except Exception:
    for i, text in enumerate(generated_text):
      print_generated_text_score(text, generated_text_score[i], name='gen__max', id=i)   
  
  print('-----------------------', generated_text_beam, generated_text_beam.shape)
  print(generated_text_score_beam, generated_text_score_beam.shape)
  if generated_text_beam is not None:
    try:
      print_generated_text_score(generated_text_beam, generated_text_score_beam)
    except Exception:
      for i, text in enumerate(generated_text_beam):
        print_generated_text_score(text, generated_text_score_beam[i], name='gen_beam', id=i)


def print_img_text_generatedtext(img, i, input_text, input_text_ids, 
                                 text, score, text_ids,
                                 generated_text, generated_text_beam=None):
  print_img(img, i)
  score = math.exp(-score)
  input_text_words = ids2text(input_text_ids)
  text_words = ids2text(text_ids)
  logging.info(content_html.format('in_ [ {} ] {}'.format(input_text, input_text_words)))
  logging.info(content_html.format('pos [ {} ] {:.6f} {}'.format(text, score, text_words)))
  print_generated_text(generated_text)
  if generated_text_beam is not None:
    print_generated_text(generated_text_beam)

def print_img_text_generatedtext_score(img, i, input_text, input_text_ids, 
                                 text, score, text_ids,
                                 generated_text, generated_text_score, 
                                 generated_text_beam=None, generated_text_score_beam=None):
  print_img(img, i)
  
  score = math.exp(-score)
  input_text_words = ids2text(input_text_ids)
  text_words = ids2text(text_ids)
  logging.info(content_html.format('in_ [ {} ] {}'.format(input_text, input_text_words)))
  logging.info(content_html.format('pos [ {} ] {:.6f} {}'.format(text, score, text_words)))

  try:
    print_generated_text_score(generated_text, generated_text_score)
  except Exception:
    for i, text in enumerate(generated_text):
      print_generated_text_score(text, generated_text_score[i], name='gen_max', id=i)   
  
  if generated_text_beam is not None:
    try:
      print_generated_text_score(generated_text_beam, generated_text_score_beam)
    except Exception:
      for i, text in enumerate(generated_text_beam):
        print_generated_text_score(text, generated_text_score_beam[i], name='gen_beam', id=i)

score_op = None

def predicts(imgs, img_features, predictor, rank_metrics, exact_predictor=None, exact_ratio=1.):
  timer = gezi.Timer('preidctor.predict')
  # TODO gpu outofmem predict for showandtell#

  if exact_predictor is None:
    if assistant_predictor is not None:
      exact_predictor = predictor
      predictor = assistant_predictor

  print(predictor, exact_predictor)

  random = True
  need_shuffle = False
  if FLAGS.max_texts > 0 and len(all_distinct_texts) > FLAGS.max_texts:
    if not random:
      texts = all_distinct_texts[:FLAGS.max_texts]
    else:
      need_shuffle = True
      index = np.random.choice(len(all_distinct_texts), FLAGS.max_texts, replace=False)
      texts = all_distinct_texts[index]
  else:
    texts = all_distinct_texts
  text_strs = all_distinct_text_strs

  step = len(texts)
  if FLAGS.metric_eval_texts_size > 0 and FLAGS.metric_eval_texts_size < step:
    step = FLAGS.metric_eval_texts_size
  start = 0
  scores = []
  while start < len(texts):
    end = start + step 
    if end > len(texts):
      end = len(texts)
    print('predicts texts start:', start, 'end:', end, end='\r', file=sys.stderr)
    score = predictor.predict(img_features, texts[start: end])
    scores.append(score)
    start = end
  score = np.concatenate(scores, 1)
  print('image_feature_shape:', img_features.shape, 'text_feature_shape:', texts.shape, 'score_shape:', score.shape)
  timer.print()
  img2text = get_bidrectional_lable_map()
  num_texts = texts.shape[0]

  for i, img in enumerate(imgs):
    indexes = (-score[i]).argsort()

    #rerank
    if exact_predictor:
      if i == 0:
        print('rerank using exact_predictor')
      top_indexes = indexes[:FLAGS.assistant_rerank_num]
      exact_texts = texts[top_indexes]
      exact_score = exact_predictor.elementwise_predict([img_features[i]], exact_texts)
      exact_score = np.squeeze(exact_score)
      if exact_ratio < 1.:
        for j in range(len(top_indexes)):
          exact_score[j] = exact_ratio * exact_score[j] + (1. - exact_ratio) * score[i][top_indexes[j]]

      #print(exact_score)
      exact_indexes = (-exact_score).argsort()

      #print(exact_indexes)
      
      new_indexes = [x for x in indexes]
      for j in range(len(exact_indexes)):
        new_indexes[j] = indexes[exact_indexes[j]]
      indexes = new_indexes
    
    hits = img2text[img]

    if i % 100 == 0:
      label_text = '|'.join([text_strs[x] for x in hits])
      img_str = img
      if img.startswith('http:') or img.startswith('D:'):
        img_str = '<p><a href={0} target=_blank><img src={0} height=200></a></p>'.format(img)
      logging.info('<P>obj:{} label:{}</P>'.format(img_str, label_text))
      for j in range(5):
        logging.info('<P>{} {} {} {}</P>'.format(j, indexes[j] in hits, ids2text(texts[indexes[j]]), exact_score[exact_indexes[j]] if exact_predictor else score[i][indexes[j]]))

    #notice only work for recall@ or precision@ not work for ndcg@, if ndcg@ must use all
    num_positions = min(num_texts, FLAGS.metric_topn)
    #num_positions = num_texts

    if not need_shuffle:
      labels = [indexes[j] in hits for j in xrange(num_positions)]
    else:
      labels = [index[indexes[j]] in hits for j in xrange(num_positions)]

    rank_metrics.add(labels)

def predicts_txt2im(text_strs, texts, predictor, rank_metrics, exact_predictor=None):
  timer = gezi.Timer('preidctor.predict text2im')
  if exact_predictor is None:
    if assistant_predictor:
      exact_predictor = predictor
      predictor = assistant_predictor

  _, img_features = get_image_names_and_features()
  # TODO gpu outofmem predict for showandtell
  #---NOTICE this might be too much mem cost if image is original encoded binary not image feature
  img_features = img_features[:FLAGS.max_images]
  if isinstance(img_features[0], np.string_):
    assert(len(img_features) < 2000) #otherwise too big mem ..
    img_features = [melt.read_image(pic_path) for pic_path in img_features]
  
  step = len(img_features)
  if FLAGS.metric_eval_images_size > 0 and FLAGS.metric_eval_images_size < step:
    step = FLAGS.metric_eval_images_size
  start = 0
  scores = []
  while start < len(img_features):
    end = start + step 
    if end > len(img_features):
      end = len(img_features)
    print('predicts images start:', start, 'end:', end, file=sys.stderr, end='\r')
    
    score = predictor.predict(img_features[start: end], texts)
   
    scores.append(score)
    start = end
  #score = predictor.predict(img_features, texts)
  score = np.concatenate(scores, 0)
  score = score.transpose()
  print('image_feature_shape:', img_features.shape, 'text_feature_shape:', texts.shape, 'score_shape:', score.shape)
  timer.print()

  text2img = get_bidrectional_lable_map_txt2im()
  num_imgs = img_features.shape[0]

  for i, text_str in enumerate(text_strs):
    indexes = (-score[i]).argsort()

    #rerank
    if exact_predictor:
      top_indexes = indexes[:FLAGS.assistant_rerank_num]
      exact_imgs = img_features[top_indexes]
      exact_score = exact_predictor.elementwise_predict(exact_imgs, [texts[i]])
      exact_score = exact_score[0]
      exact_indexes = (-exact_score).argsort()
      new_indexes = [x for x in indexes]
      for j in range(len(exact_indexes)):
        new_indexes[j] = indexes[exact_indexes[j]]
      indexes = new_indexes
    
    hits = text2img[text_str]

    num_positions = min(num_imgs, FLAGS.metric_topn)
    #num_positions = num_imgs
    
    labels = [indexes[j] in hits for j in xrange(num_positions)]

    rank_metrics.add(labels)

def random_predict_index(seed=None):
  imgs, img_features = get_image_names_and_features()
  num_metric_eval_examples = min(FLAGS.num_metric_eval_examples, len(imgs)) 
  if num_metric_eval_examples <= 0:
    num_metric_eval_examples = len(imgs)
  if seed:
    np.random.seed(seed)
  return  np.random.choice(len(imgs), num_metric_eval_examples, replace=False)

def evaluate_scores(predictor, random=False, index=None, exact_predictor=None, exact_ratio=1.):
  timer = gezi.Timer('evaluate_scores')
  init()
  if FLAGS.eval_img2text:
    imgs, img_features = get_image_names_and_features()
    num_metric_eval_examples = min(FLAGS.num_metric_eval_examples, len(imgs)) 
    if num_metric_eval_examples <= 0:
      num_metric_eval_examples = len(imgs)

    step = FLAGS.metric_eval_batch_size

    if random:
      if index is None:
        index = np.random.choice(len(imgs), num_metric_eval_examples, replace=False)
      imgs = imgs[index]
      img_features = img_features[index]
      if isinstance(img_features[0], np.string_):
        img_features = np.array([melt.read_image(pic_path) for pic_path in img_features])

    rank_metrics = gezi.rank_metrics.RecallMetrics()

    start = 0
    while start < num_metric_eval_examples:
      end = start + step
      if end > num_metric_eval_examples:
        end = num_metric_eval_examples
      print('predicts image start:', start, 'end:', end, file=sys.stderr, end='\r')
      predicts(imgs[start: end], img_features[start: end], predictor, rank_metrics, 
               exact_predictor=exact_predictor, exact_ratio=exact_ratio)
      start = end
      
    melt.logging_results(
      rank_metrics.get_metrics(), 
      rank_metrics.get_names(), 
      tag='evaluate: epoch:{} step:{} train:{} eval:{}'.format(
        melt.epoch(), 
        melt.step(),
        melt.train_loss(),
        melt.eval_loss()))

  if FLAGS.eval_text2img:
    num_metric_eval_examples = min(FLAGS.num_metric_eval_examples, len(all_distinct_texts))

    if random:
      index = np.random.choice(len(all_distinct_texts), num_metric_eval_examples, replace=False)
      text_strs = all_distinct_text_strs[index]
      texts = all_distinct_texts[index]
    else:
      text_strs = all_distinct_text_strs
      texts = all_distinct_texts

    rank_metrics2 = gezi.rank_metrics.RecallMetrics()

    start = 0
    while start < num_metric_eval_examples:
      end = start + step
      if end > num_metric_eval_examples:
        end = num_metric_eval_examples
      print('predicts start:', start, 'end:', end, file=sys.stderr, end='\r')
      predicts_txt2im(text_strs[start: end], texts[start: end], predictor, rank_metrics2, exact_predictor=exact_predictor)
      start = end
    
    melt.logging_results(
      rank_metrics2.get_metrics(), 
      ['t2i' + x for x in rank_metrics2.get_names()],
      tag='text2img')

  timer.print()

  if FLAGS.eval_img2text and FLAGS.eval_text2img:
    return rank_metrics.get_metrics() + rank_metrics2.get_metrics(), rank_metrics.get_names() + ['t2i' + x for x in rank_metrics2.get_names()]
  elif FLAGS.eval_img2text:
    return rank_metrics.get_metrics(), rank_metrics.get_names()
  else:
    return rank_metrics2.get_metrics(), rank_metrics2.get_names()
  
