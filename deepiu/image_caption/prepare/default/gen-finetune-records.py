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

flags.DEFINE_string('vocab', '/tmp/train/vocab.bin', 'vocabulary binary file')
flags.DEFINE_boolean('pad', False, 'wether to pad to pad 0 to make fixed length text ids')
flags.DEFINE_string('output_directory', '/home/gezi/new/temp/image_caption/keyword/tfrecord',
                         'Directory to download data files and write the '
                         'converted result')

flags.DEFINE_string('input_directory', '/home/gezi/new/data/keyword/feature', '')
flags.DEFINE_string('image_dir', None, '')
flags.DEFINE_string('name', 'train', '')
flags.DEFINE_integer('threads', 12, 'Number of threads for dealing')

flags.DEFINE_boolean('np_save', False, 'np save text ids and text')

flags.DEFINE_integer('num_max_inputs', 0, '')
flags.DEFINE_integer('num_max_records', 0, 'max output records')
#flags.DEFINE_string('seg_method', 'default', '')
flags.DEFINE_boolean('write_sequence_example', False, '')

flags.DEFINE_integer('text_index', 1, '')
flags.DEFINE_integer('image_feature_index', -1, '')

#flags.DEFINE_integer('text_index', 2, '')
#flags.DEFINE_integer('image_feature_index', 1, '')

"""
use only top query?
use all queries ?
use all queries but deal differently? add weight to top query?
"""

import sys,os, glob
import multiprocessing

from multiprocessing import Process, Manager, Value, Pool

import numpy as np
import melt
import gezi

#cp conf.py ../../../ before
from deepiu.util import text2ids

import conf  
from conf import TEXT_MAX_WORDS, NUM_RESERVED_IDS, ENCODE_UNK, IMAGE_FEATURE_LEN

print('ENCODE_UNK', ENCODE_UNK, file=sys.stderr)
assert ENCODE_UNK == text2ids.ENCODE_UNK

gtexts = []
gtext_strs = []

image_labels = {}

image_names = []
image_features = []

#how many records generated
record_counter = Value('i', 0)
image_counter = Value('i', 0)
#the max num words of the longest text
max_num_words = Value('i', 0)
#the total words of all text
sum_words = Value('i', 0)

text2ids.init() 

import libstring_util

#print('----------', FLAGS.seg_method)

def _text2ids(text, max_words):
  word_ids = text2ids.text2ids(text, seg_method=FLAGS.seg_method, feed_single=FLAGS.feed_single, allow_all_zero=True, pad=False)
  word_ids_length = len(word_ids)

  if len(word_ids) == 0:
    return []
  word_ids = word_ids[:max_words]
  if FLAGS.pad:
    word_ids = gezi.pad(word_ids, max_words, 0)

  return word_ids

def is_luanma(words, word_ids):
  #return False
  for word, word_id in zip(words, word_ids):
    if word_id == text2ids.vocab.unk_id():
      #print(word, libstring_util.is_gbk_dual(word))
      if libstring_util.is_gbk_dual(word):
        return True 
  return False

def deal_file(file):
  out_file = '{}/{}'.format(FLAGS.output_directory, '-'.join([FLAGS.name, file.split('/')[-1].split('-')[-1]]))
  print('file:', file, 'out_file:', out_file, file=sys.stderr)
  with melt.tfrecords.Writer(out_file) as writer:
    num = 0
    for line in open(file):
      if num % 1000 == 0:
        print(num, file=sys.stderr)
      
      l = line.rstrip('\n').split('\t')
      img = l[0]

      texts = l[FLAGS.text_index].split('\x01')

      image_path =  os.path.join(FLAGS.image_dir, img.replace('/', '_'))
      encoded_image = melt.read_image(image_path)

      is_top_text = True
      for text in texts:
        if text.strip() == '':
          print('empty line', line, file=sys.stderr)
          continue

        word_ids = _text2ids(text, TEXT_MAX_WORDS)
        word_ids_length = len(word_ids)
        if num % 10000 == 0:
          print(img, text, word_ids, text2ids.ids2text(word_ids), file=sys.stderr)
        if len(word_ids) == 0:
          print('empy wordids!', file=sys.stderr)
          print(img, text, word_ids, text2ids.ids2text(word_ids), file=sys.stderr)
          continue 
        #if is_luanma(words, word_ids):
        #  print('luanma', img, text, word_ids, text2ids.ids2text(word_ids), len(image_feature), file=sys.stderr)
        #  continue 
                  
        word_ids = word_ids[:TEXT_MAX_WORDS]
        if FLAGS.pad:
          word_ids = gezi.pad(word_ids, TEXT_MAX_WORDS, 0)
        if not FLAGS.write_sequence_example:
          example = tf.train.Example(features=tf.train.Features(feature={
            'image_name': melt.bytes_feature(img),
            'image_data': melt.bytes_feature(encoded_image),
            'text_str': melt.bytes_feature(text),
            'text': melt.int64_feature(word_ids),
            }))
        else:
          example = tf.train.SequenceExample(
            context=melt.features(
            {
              'image_name': melt.bytes_feature(img),
              'image_data': melt.bytes_feature(encoded_image),
              'text_str': melt.bytes_feature(text),
            }),
            feature_lists=melt.feature_lists(
            { 
              'text': melt.int64_feature_list(word_ids)
            }))
        writer.write(example)
      
        #global counter, max_num_words, sum_words
        with record_counter.get_lock():
          record_counter.value += 1
        if word_ids_length > max_num_words.value:
          with max_num_words.get_lock():
            max_num_words.value = word_ids_length
        with sum_words.get_lock():
          sum_words.value += word_ids_length
        
        if FLAGS.np_save:
          assert FLAGS.threads == 1
          gtexts.append(word_ids)
          gtext_strs.append(text)
          
          #Depreciated not use image_labels
          if img not in image_labels:
            image_labels[img] = set()
          image_labels[img].add(text)

        if is_top_text:
          is_top_text = False 
          with image_counter.get_lock():
            image_counter.value += 1

          if FLAGS.np_save:
            if img not in image_labels:
              image_labels[img] = set()

            image_names.append(img)
            image_features.append(image_feature)

          if FLAGS.num_max_records > 0:
            #if fixed valid only get one click for each image
            break
        
      num += 1   
      if num == FLAGS.num_max_records:
        break
     
def run():
  files =  glob.glob(FLAGS.input_directory + '/*')
  if FLAGS.num_max_inputs:
    files = files[:FLAGS.num_max_inputs]

  print(FLAGS.input_directory, len(files))

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

  num_images = image_counter.value
  print('num_images:', num_images)

  num_records = record_counter.value
  print('num_records:', num_records)
  gezi.write_to_txt(num_records, os.path.join(FLAGS.output_directory, 'num_records.txt'))

  print('num_records_per_image', num_records / num_images)

  print('num_texts', len(gtexts))

  print('max_num_words:', max_num_words.value)
  print('avg_num_words:', sum_words.value / num_records)


  if FLAGS.np_save:
    print('len(texts):', len(gtexts))
    np.save(os.path.join(FLAGS.output_directory, 'texts.npy'), np.array(gtexts))
    np.save(os.path.join(FLAGS.output_directory, 'text_strs.npy'), np.array(gtext_strs))
   
    np.save(os.path.join(FLAGS.output_directory, 'image_labels.npy'), image_labels)

    np.save(os.path.join(FLAGS.output_directory, 'image_names.npy'), np.array(image_names))
    np.save(os.path.join(FLAGS.output_directory, 'image_features.npy'), np.array(image_features))

def main(_):
  run()

if __name__ == '__main__':
  tf.app.run()
