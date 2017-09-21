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
flags.DEFINE_boolean('eval_rank', True, '')
flags.DEFINE_boolean('eval_translation', True, 'for generative method')
flags.DEFINE_boolean('eval_translation_reseg', True, 'for different seg tranining but use same seg if True')
flags.DEFINE_string('eval_result_dir', None, 'dir for writting result')

flags.DEFINE_integer('show_info_interval', 100, '') 

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

from gezi.metrics import Bleu, Meteor, Rouge, Cider, PTBTokenizer

import deepiu

import numpy as np
import math

all_distinct_texts = None
all_distinct_text_strs = None
all_distinct_text_id_strs = None #used only for evaluate translation

img2text = None
text2img = None

image_names = None
image_features = None
image_model = None

assistant_predictor = None

inited = False

tokenizer = None

def init(image_model_=None):
  global inited

  if inited:
    return

  test_dir = FLAGS.valid_resource_dir
  global all_distinct_texts, all_distinct_text_strs
  global vocab, vocab_size
  if all_distinct_texts is None:
    print('loading valid resource from:', test_dir)
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

  #for evaluation without train will also use evaluator so only set log path in train.py
  #logging.set_logging_path(FLAGS.model_dir)
  if FLAGS.assistant_model_dir:
    global assistant_predictor
    #use another session different from main graph, otherwise variable will be destroy/re initailized in melt.flow
    #by default now Predictor using tf.Session already, here for safe, if use same session then not work
    #NOTICE must use seperate sess!!
    if is_raw_image(image_features) and not melt.varname_in_checkpoint(FLAGS.image_model_name, FLAGS.assistant_model_dir):
      print('assist predictor use deepiu.util.sim_predictor.SimPredictor as is raw image as input')
      global image_model 
      if image_model_ is not None:
        image_model = image_model_ 
      else:
        image_model = melt.image.ImageModel(FLAGS.image_checkpoint_file, 
                                            FLAGS.image_model_name, 
                                            feature_name=None)
      assistant_predictor = deepiu.util.sim_predictor.SimPredictor(FLAGS.assistant_model_dir, image_model=image_model)
      print('assistant predictor init ok')
    else:
      assistant_predictor = melt.SimPredictor(FLAGS.assistant_model_dir)
    print('assistant_predictor', assistant_predictor)

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

def is_raw_image(image_features):
  return isinstance(image_features[0], np.string_)

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

def is_img(img):
  return img.startswith('http:') or img.startswith('D:') or img.endswith('.jpg')

def get_img_url(img):
  return os.path.join(FLAGS.image_dir, img) if FLAGS.image_dir and not img.startswith("http://") else img

def print_img(img, i):
  img_url = get_img_url(img)
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
def print_generated_text(generated_text, id=-1, name='greedy'):
  if id >= 0:
    logging.info(content_html.format('{}_{}:[ {} ]'.format(name, id, ids2text(generated_text))))
  else:
    logging.info(content_html.format('{}:[ {} ]'.format(name, ids2text(generated_text))))

def print_generated_text_score(generated_text, score, id=-1, name='greedy'):
  try:
    if id >= 0:
      logging.info(content_html.format('{}_{}:[ {} ] {:.6f}'.format(name, id, ids2text(generated_text), score)))
    else:
      logging.info(content_html.format('{}:[ {} ] {:.6f}'.format(name, ids2text(generated_text), score)))
  except Exception:
    if id >= 0:
      logging.info(content_html.format('{}_{}:[ {} ] {}'.format(name, id, ids2text(generated_text), score)))
    else:
      logging.info(content_html.format('{}:[ {} ] {}'.format(name, ids2text(generated_text), score)))  

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
      print_generated_text_score(text, generated_text_score[i], name='max', id=i)   
  
  #print('-----------------------', generated_text_beam, generated_text_beam.shape)
  #print(generated_text_score_beam, generated_text_score_beam.shape)
  if generated_text_beam is not None:
    try:
      print_generated_text_score(generated_text_beam, generated_text_score_beam)
    except Exception:
      for i, text in enumerate(generated_text_beam):
        print_generated_text_score(text, generated_text_score_beam[i], name='beam', id=i)


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
      print_generated_text_score(text, generated_text_score[i], name='max', id=i)   
  
  if generated_text_beam is not None:
    try:
      print_generated_text_score(generated_text_beam, generated_text_score_beam)
    except Exception:
      for i, text in enumerate(generated_text_beam):
        print_generated_text_score(text, generated_text_score_beam[i], name='beam', id=i)

score_op = None

def predicts(imgs, img_features, predictor, rank_metrics, exact_predictor=None, exact_ratio=1.):
  # TODO gpu outofmem predict for showandtell#
  if exact_predictor is None:
    if assistant_predictor is not None:
      exact_predictor = predictor
      predictor = assistant_predictor

  #print(predictor, exact_predictor)

  if isinstance(img_features[0], np.string_):
    assert(len(img_features) < 2000) #otherwise too big mem ..
    img_features = np.array([melt.read_image(pic_path) for pic_path in img_features])  

  img2text = get_bidrectional_lable_map()

  random = True
  need_shuffle = False
  if FLAGS.max_texts > 0 and len(all_distinct_texts) > FLAGS.max_texts:
    assert random
    if not random:
      texts = all_distinct_texts[:FLAGS.max_texts]
    else:
      need_shuffle = True

      all_hits = set()
      for img in (imgs):
        hits = img2text[img]
        for hit in hits:
          all_hits.add(hit)
      
      index = np.random.choice(len(all_distinct_texts), FLAGS.max_texts, replace=False)
      index = [x for x in index if x not in all_hits]
      index = list(all_hits) + index 
      index = index[:FLAGS.max_texts]
      index = np.array(index)
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
    #print('predicts texts start:', start, 'end:', end, end='\r', file=sys.stderr)
    score = predictor.predict(img_features, texts[start: end])
    scores.append(score)
    start = end
  score = np.concatenate(scores, 1)
  #print('image_feature_shape:', img_features.shape, 'text_feature_shape:', texts.shape, 'score_shape:', score.shape)
  num_texts = texts.shape[0]

  for i, img in enumerate(imgs):
    indexes = (-score[i]).argsort()
    #rerank
    if exact_predictor:
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

    if FLAGS.show_info_interval and i % FLAGS.show_info_interval == 0:
      label_text = '|'.join([text_strs[x] for x in hits])
      img_str = img
      if is_img(img):
        img_str = '{0}<p><a href={1} target=_blank><img src={1} height=200></a></p>'.format(img, get_img_url(img))
      logging.info('<P>obj: {} label: {}</P>'.format(img_str, label_text))
      for j in range(5):
        is_hit = indexes[j] in hits if not need_shuffle else index[indexes[j]] in hits
        logging.info('<P>{} {} {} {}</P>'.format(j, is_hit, ids2text(texts[indexes[j]]), exact_score[exact_indexes[j]] if exact_predictor else score[i][indexes[j]]))

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
    #print('predicts images start:', start, 'end:', end, file=sys.stderr, end='\r')
    
    #here might not accept raw image for bow predictor as assistant predictor TODO how to add image process here to gen feature first?
    score = predictor.predict(img_features[start: end], texts)
   
    scores.append(score)
    start = end
  #score = predictor.predict(img_features, texts)
  score = np.concatenate(scores, 0)
  score = score.transpose()
  #print('image_feature_shape:', img_features.shape, 'text_feature_shape:', texts.shape, 'score_shape:', score.shape)
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
  """
  actually this is rank metrics evaluation, by default recall@1,2,5,10,50
  """
  timer = gezi.Timer('evaluate_scores')
  init()
  if FLAGS.eval_img2text:
    imgs, img_features = get_image_names_and_features()
    num_metric_eval_examples = min(FLAGS.num_metric_eval_examples, len(imgs)) 
    if num_metric_eval_examples <= 0:
      num_metric_eval_examples = len(imgs)
    if num_metric_eval_examples == len(imgs):
      random = False

    step = FLAGS.metric_eval_batch_size

    if random:
      if index is None:
        index = np.random.choice(len(imgs), num_metric_eval_examples, replace=False)
      imgs = imgs[index]
      img_features = img_features[index]
    else:
      img_features = img_features[:num_metric_eval_examples]

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
  
#---------for eval translation
refs = None
def prepare_refs():
  global refs, all_distinct_text_id_strs
  if refs is None:
    if all_distinct_text_id_strs is None:
      if not FLAGS.eval_translation_reseg:
        all_distinct_text_id_strs = [' '.join([str(x) for x in l if int(x) is not 0]) for l in all_distinct_texts]
      else:
        import jieba
        all_distinct_text_id_strs = [' '.join([x.encode('utf-8') for x in jieba.cut(''.join([vocab.key(int(x)) for x in l if int(x) is not 0]))]) for l in all_distinct_texts]
    refs = {}
    for img, hits in img2text.items():
      refs[img] = [all_distinct_text_id_strs[hit] for hit in hits]

  return refs

def translation_predicts(imgs, img_features, predictor, results):
  if isinstance(img_features[0], np.string_):
    img_features = np.array([melt.read_image(pic_path) for pic_path in img_features])

  texts, _ = predictor.predict_text(img_features)
  #only use top prediction of beam search
  texts = [x[0] for x in texts]
  for i in range(len(texts)):
    #for eval even if only one predict must also be list, also exclude last end id
    if not FLAGS.eval_translation_reseg:
      texts[i] = [' '.join([str(x) for x in texts[i][:list(texts[i]).index(vocab.end_id())]])] 
    else:
      import jieba
      texts[i] = ''.join([vocab.key(int(x)) for x in texts[i][:list(texts[i]).index(vocab.end_id())]])
      texts[i] = [' '.join([x.encode('utf-8') for x in jieba.cut(texts[i])])]
    results[imgs[i]] = texts[i]

def evaluate_translation(predictor, random=False, index=None):
  timer = gezi.Timer('evaluate_translation')

  refs = prepare_refs()

  imgs, img_features = get_image_names_and_features()
  num_metric_eval_examples = min(FLAGS.num_metric_eval_examples, len(imgs))
  if num_metric_eval_examples <= 0:
    num_metric_eval_examples = len(imgs)
  if num_metric_eval_examples == len(imgs):
    random = False

  step = FLAGS.metric_eval_batch_size

  if random:
    if index is None:
      index = np.random.choice(len(imgs), num_metric_eval_examples, replace=False)
    imgs = imgs[index]
    img_features = img_features[index]
  else:
    img_features = img_features[:num_metric_eval_examples]

  results = {}

  start = 0
  while start < num_metric_eval_examples:
    end = start + step
    if end > num_metric_eval_examples:
      end = num_metric_eval_examples
    print('predicts image start:', start, 'end:', end, file=sys.stderr, end='\r')
    translation_predicts(imgs[start: end], img_features[start: end], predictor, results)
    start = end
    
  scorers = [
            (Bleu(4), ["bleu_1", "bleu_2", "bleu_3", "bleu_4"]),
            (Meteor(),"meteor"),
            (Rouge(), "rouge_l"),
            (Cider(), "cider")
        ]

  score_list = []
  metric_list = []

  selected_refs = {}
  selected_results = {}
  #by doing this can force same .keys()
  for key in results:
    selected_refs[key] = refs[key]
    selected_results[key] = results[key]
    assert len(selected_results[key]) == 1, selected_results[key]
  assert selected_results.keys() == selected_refs.keys(), '%d %d'%(len(selected_results.keys()), len(selected_refs.keys())) 

  if FLAGS.eval_translation_reseg:
    print('tokenization...', file=sys.stderr)
    global tokenizer
    if tokenizer is None:
      tokenizer = PTBTokenizer()
    selected_refs  = tokenizer.tokenize(selected_refs)
    selected_results = tokenizer.tokenize(selected_results)

  logging.info('predict&label:{}{}{}'.format('|'.join(selected_results.items()[0][1]), '---', '|'.join(selected_refs.items()[0][1])))

  for scorer, method in scorers:
    print('computing %s score...'%(scorer.method()), file=sys.stderr)
    score, scores = scorer.compute_score(selected_refs, selected_results)
    if type(method) == list:
      for sc, scs, m in zip(score, scores, method):
        score_list.append(sc)
        metric_list.append(m)
        if FLAGS.eval_result_dir:
          out = open(os.path.join(FLAGS.eval_result_dir, m+'.txt'), 'w')
          for i, sc in enumerate(scs):
            key = selected_results.keys()[i]
            result = selected_results[key]
            refs = '\x01'.join(selected_refs[key])
            print(key, result, refs, sc, sep='\t', file=out)
    else:
      score_list.append(score)
      metric_list.append(method)
      if FLAGS.eval_result_dir:
        out = open(os.path.join(FLAGS.eval_result_dir, m+'.txt'), 'w')
        for i, sc in enumerate(scores):
          key = selected_results.keys()[i]
          result = selected_results[key]
          refs = '\x01'.join(selected_refs[key])
          print(key, result, refs, sc, sep='\t', file=out)
  
  #exclude "bleu_1", "bleu_2", "bleu_3"
  score_list, metric_list = score_list[3:], metric_list[3:]
  assert(len(score_list) == 4)

  avg_score = sum(score_list) / len(score_list)
  score_list.append(avg_score)
  metric_list.append('avg')
  metric_list = ['trans_' + x for x in metric_list]

  melt.logging_results(
    score_list,
    metric_list,
    tag='evaluate: epoch:{} step:{} train:{} eval:{}'.format(
      melt.epoch(), 
      melt.step(),
      melt.train_loss(),
      melt.eval_loss()))

  timer.print()

  return score_list, metric_list


def evaluate(predictor, random=False, index=None, eval_rank=True, eval_translation=False):
  if hasattr(predictor, 'image_model'):
    #might be used for show and tell where feature is big, need to convert from image raw data
    if predictor.image_model is None:
      predictor.image_model = image_model 
    print('predictor.image_model:', predictor.image_model)

  scores = []
  metrics = []
  if eval_rank:
    sc, m = evaluate_scores(predictor, random=random, index=index)
    scores += sc 
    metrics += m 
  if eval_translation:
    sc, m = evaluate_translation(predictor, random=random, index=index)
    scores += sc 
    metrics += m     

  return scores, metrics
