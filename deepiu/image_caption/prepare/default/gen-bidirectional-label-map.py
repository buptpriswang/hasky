#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   gen-bidirectional-label-map.py
#        \author   chenghuige  
#          \date   2016-10-07 22:19:40.675048
#   \Description  
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS
  
flags.DEFINE_string('img2text', '', '')
flags.DEFINE_string('text2id', '', '')
flags.DEFINE_string('all_distinct_text_strs', '', '')
flags.DEFINE_string('all_distinct_image_names', '', '')

flags.DEFINE_string('image_names', '', '')

flags.DEFINE_string('img2id', '', '')
flags.DEFINE_string('text2img', '', '')


import sys
import numpy as np 

all_distinct_text_strs = np.load(FLAGS.all_distinct_text_strs) 
all_distinct_image_names = np.load(FLAGS.all_distinct_image_names)

img2text = {}
text2id = {}
text2img = {}
img2id = {}

for i, text in enumerate(all_distinct_text_strs):
  text2id[text] = i

for i, image_name in enumerate(all_distinct_image_names):
  img2id[image_name] = i

for line in sys.stdin:
  l = line.rstrip('\n').split('\t')
  img = l[0]
  texts = l[1].split('\x01')

  if img not in img2text:
    img2text[img] = set()

  m = img2text[img]
  for text in texts:
    if text not in text2id:
      continue
    id = text2id[text]
    m.add(id)

    if img in img2id:
      img_id = img2id[img]
      if text not in text2img:
        text2img[text] = set([img_id])
      else:
        text2img[text].add(img_id)

img_per_text = sum([len(text2img[x]) for x in text2img]) / len(text2img)
print('img per text:', img_per_text)
text_per_img = sum([len(img2text[x]) for x in img2text]) / len(img2text)
print('text per image:', text_per_img)

np.save(FLAGS.img2text, img2text)
np.save(FLAGS.text2id, text2id)

np.save(FLAGS.text2img, text2img)
np.save(FLAGS.img2id, img2id)

