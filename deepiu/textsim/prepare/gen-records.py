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

import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('vocab', None, 'vocabulary binary file')
flags.DEFINE_boolean('pad', False, 'wether to pad to pad 0 to make fixed length text ids')
flags.DEFINE_string('output_dir', '/tmp/train/',
                         'Directory to download data files and write the '
                         'converted result')

flags.DEFINE_string('input_dir', None, 'input pattern')
flags.DEFINE_string('name', 'train', '')
flags.DEFINE_integer('threads', 12, 'Number of threads for dealing')

flags.DEFINE_integer('num_max_inputs', 0, '')
flags.DEFINE_integer('max_lines', 0, '')

flags.DEFINE_boolean('np_save', False, 'np save text ids and text')

#flags.DEFINE_string('seg_method', 'default', '')

"""
use only top query?
use all queries ?
use all queries but deal differently? add weight to top query?
"""

import sys, os, glob
import multiprocessing
from multiprocessing import Process, Manager, Value

import numpy as np
import melt
import gezi

from deepiu.util import text2ids

import conf  
from conf import TEXT_MAX_WORDS, NUM_RESERVED_IDS, ENCODE_UNK

print('ENCODE_UNK', ENCODE_UNK, file=sys.stderr)
assert ENCODE_UNK == text2ids.ENCODE_UNK

ltexts = []
ltext_strs = []
rtexts = []
rtext_strs = []

#how many records generated
counter = Value('i', 0)
#the max num words of the longest text
max_num_words = Value('i', 0)
#the total words of all text
sum_words = Value('i', 0)

text2ids.init()

def _text2ids(text, max_words):
  word_ids = text2ids.text2ids(text, seg_method=FLAGS.seg_method, feed_single=FLAGS.feed_single, allow_all_zero=True, pad=False)
  word_ids_length = len(word_ids)

  if len(word_ids) == 0:
    return []
  word_ids = word_ids[:max_words]
  if FLAGS.pad:
    word_ids = gezi.pad(word_ids, max_words, 0)

  return word_ids

def deal_file(file):
  out_file = '{}/{}'.format(FLAGS.output_dir, '-'.join([FLAGS.name, file.split('/')[-1].split('-')[-1]]))
  print('out_file:', out_file)
  with melt.tfrecords.Writer(out_file) as writer:
    num = 0
    for line in open(file):
      #line = line.lower()
      if num % 1000 == 0:
        print(num)
      if FLAGS.max_lines and num >= FLAGS.max_lines:
        break
      l = line.rstrip().split('\t')

      if len(l) != 2:
        continue

      ltext, rtext = l 

      lword_ids = _text2ids(ltext, TEXT_MAX_WORDS)
      rword_ids = _text2ids(rtext, TEXT_MAX_WORDS)

      if not lword_ids or not rword_ids:
        continue
      
      if num % 1000 == 0:
        print(ltext, lword_ids, text2ids.ids2text(lword_ids), file=sys.stderr)
        print(rtext, rword_ids, text2ids.ids2text(rword_ids), file=sys.stderr)
      
      example = tf.train.Example(features=tf.train.Features(feature={
        'ltext_str': melt.bytes_feature(ltext),
        'ltext': melt.int_feature(lword_ids),
        'rtext_str': melt.bytes_feature(rtext),
        'rtext': melt.int_feature(rword_ids),
        }))
      writer.write(example)

      if FLAGS.np_save:
        assert FLAGS.threads == 1
        ltexts.append(lword_ids)
        ltext_strs.append(ltext)
        rtexts.append(rword_ids)
        rtext_strs.append(rtext)
      
      global counter, max_num_words, sum_words
      with counter.get_lock():
        counter.value += 1
      
      word_ids = lword_ids
      word_ids_length = len(word_ids)
      if word_ids_length > max_num_words.value:
        with max_num_words.get_lock():
          max_num_words.value = word_ids_length
      with sum_words.get_lock():
        sum_words.value += word_ids_length
      num += 1


def run():
  files =  glob.glob(FLAGS.input_dir + '/*')
  if FLAGS.num_max_inputs:
    files = files[:FLAGS.num_max_inputs]

  print(FLAGS.input_dir, len(files))

  if FLAGS.threads > len(files):
    FLAGS.threads = len(files)
  if FLAGS.threads > 1:
    pool = multiprocessing.Pool(processes = FLAGS.threads)

    pool.map(deal_file, files)

    pool.close()
    pool.join()
  else:
    for file in files:
      deal_file(file)

  num_records = counter.value
  print('num_records:', num_records)
  gezi.write_to_txt(num_records, os.path.join(FLAGS.output_dir, 'num_records.txt'))

  print('counter:', counter.value)
  print('max_num_words:', max_num_words.value)
  print('avg_num_words:', sum_words.value / counter.value)


  if FLAGS.np_save:
    #hack here TODO   now image_name as ltext_str, image_features as ltext(ids)
    #texts as rtext(ids), text_strs as rtext_str
    np.save(os.path.join(FLAGS.output_dir, 'image_names.npy'), np.array(ltext_strs))
    np.save(os.path.join(FLAGS.output_dir, 'image_features.npy'), np.array(ltexts))

    np.save(os.path.join(FLAGS.output_dir, 'texts.npy'), np.array(rtexts))
    np.save(os.path.join(FLAGS.output_dir, 'text_strs.npy'), np.array(rtext_strs))
   



def main(_):
  run()

if __name__ == '__main__':
  tf.app.run()

