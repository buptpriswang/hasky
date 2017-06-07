#!/usr/bin/env python
# ==============================================================================
#          \file   to_flickr_caption.py
#        \author   chenghuige  
#          \date   2016-07-11 16:29:27.084402
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""
@TODO could do segment parallel @TODO
now single thread... slow
"""

import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer("most_common", 0, "if > 0 then get vocab with most_common words")
flags.DEFINE_integer("min_count", 0, "if > 0 then cut by min_count")
flags.DEFINE_boolean("add_unknown", True, "treat ignored words as unknow")
flags.DEFINE_boolean("save_count_info", True, "save count info to bin")
flags.DEFINE_string("out_dir", '/tmp/train/', "save count info to bin")
flags.DEFINE_string("captions_file", "/home/gezi/new/data/MSCOCO/annotations/captions_train2014.json", "")


assert FLAGS.most_common > 0 or FLAGS.min_count > 0
import nowarning
from libword_counter import WordCounter
counter = WordCounter(
    addUnknown=FLAGS.add_unknown,
    mostCommon=FLAGS.most_common,
    minCount=FLAGS.min_count,
    saveCountInfo=FLAGS.save_count_info)

import sys,os
import numpy as np
import melt

import conf 
from conf import IMAGE_FEATURE_LEN

import json 
import nltk.tokenize

from libprogress_bar import ProgressBar

START_WORD = '<S>'
END_WORD = '</S>'

with tf.gfile.FastGFile(FLAGS.captions_file, "r") as f:
   caption_data = json.load(f)

pb = ProgressBar(len(caption_data["annotations"]))

id_to_filename = [(x["id"], x["file_name"]) for x in caption_data["images"]]

print(len(id_to_filename))

ids = set()
for x in caption_data["images"]:
  ids.add(x["id"])
print(len(ids))

print(len(caption_data["annotations"]))

caption_data = None

for annotation in caption_data["annotations"]:
  pb.progress()
  caption = annotation["caption"]
  words = nltk.tokenize.word_tokenize(caption.lower())
  counter.add(START_WORD)
  for word in words:
    counter.add(word.encode('utf-8'))
  counter.add(END_WORD)


print(FLAGS.out_dir, file=sys.stderr)
counter.save(FLAGS.out_dir + '/vocab.bin', FLAGS.out_dir + '/vocab.txt')