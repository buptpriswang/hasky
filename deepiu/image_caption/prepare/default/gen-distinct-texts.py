#!/usr/bin/env python
# ==============================================================================
#          \file   gen-distinct-texts.py
#        \author   chenghuige  
#          \date   2016-07-24 09:21:18.388774
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys,os
import numpy as np

import tensorflow as tf 

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('dir', '/tmp/train/', '')
flags.DEFINE_boolean('shuffle', True, 'shffle data ast last?')
flags.DEFINE_integer('max_texts', 0, 'to avoid oom of gpu, 15w is save for 15 * 22 * 256')
flags.DEFINE_integer('max_images', 0, 'to avoid oom of gpu, 15w is save for 15 * 22 * 256')

import gezi

texts = np.load(FLAGS.dir + '/texts.npy')
text_strs = np.load(FLAGS.dir + '/text_strs.npy')

image_features = np.load(FLAGS.dir + '/image_features.npy')
image_names = np.load(FLAGS.dir + '/image_names.npy')

distinct_image_features = []
distinct_image_names = []

distinct_texts = []
distinct_text_strs = []

image_max_len = 0
for image_feature in image_features:
  if len(image_feature) > image_max_len:
    image_max_len = len(image_feature)

maxlen = 0
for text in texts:
  if len(text) > maxlen:
    maxlen = len(text)

text_set = set()
for text, text_str in zip(list(texts), list(text_strs)):
  if text_str not in text_set:
    text_set.add(text_str)
    distinct_texts.append(gezi.pad(text, maxlen))
    distinct_text_strs.append(text_str)

    if len(distinct_texts) == FLAGS.max_texts:
  	  print('stop at', FLAGS.max_texts, file=sys.stderr)
  	  break

print('num ori texts:', len(texts))
print('num distinct texts:', len(distinct_texts))

image_set = set()
for image_feature, image_name in zip(list(image_features), list(image_names)):
  if image_name not in image_set:
    image_set.add(image_name)
    #distinct_image_features.append(gezi.pad(image_feature, maxlen))
    distinct_image_features.append(image_feature)
    distinct_image_names.append(image_name)

    if len(distinct_image_features) == FLAGS.max_images:
  	  print('stop at', FLAGS.max_images, file=sys.stderr)
  	  break

print('num ori images:', len(image_features))
print('num distinct images:', len(distinct_image_features))

distinct_texts = np.array(distinct_texts)
distinct_text_strs = np.array(distinct_text_strs)
distinct_image_features = np.array(distinct_image_features)
distinct_image_names = np.array(distinct_image_names)
if FLAGS.shuffle:
	distinct_texts, distinct_text_strs = gezi.unison_shuffle(distinct_texts, distinct_text_strs)
	distinct_image_features, distinct_image_names = gezi.unison_shuffle(distinct_image_features, distinct_image_names)

np.save(FLAGS.dir + '/distinct_texts.npy', distinct_texts)
np.save(FLAGS.dir + '/distinct_text_strs.npy', distinct_text_strs)
np.save(FLAGS.dir + '/distinct_image_features.npy', distinct_image_features)
np.save(FLAGS.dir + '/distinct_image_names.npy', distinct_image_names)
