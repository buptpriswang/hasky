#!/usr/bin/env python
# ==============================================================================
#          \file   imgs2features.py
#        \author   chenghuige  
#          \date   2017-04-09 00:54:22.224729
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS
  
#--------- read data
flags.DEFINE_integer('key_index', 0, '')
flags.DEFINE_integer('val_index', -1, '')
flags.DEFINE_integer('batch_size', 512, '')
flags.DEFINE_boolean('show_decode_error', False, '')
flags.DEFINE_string('out', 'good_pics.pkl', '')
flags.DEFINE_string('format', 'jpeg', '')

import sys, os, glob, traceback
import melt
import urllib 

import cPickle 

sess = None
op = None 
images_feed =  tf.placeholder(tf.string, [None,], name='images')

def build_graph(images, format='jpeg'):
  if format == 'jpeg':
    return tf.image.decode_jpeg(images, chanel=3)
  elif format == 'bmp':
    return tf.image.decode_bmp(images, chanel=3)
  else:
    return tf.image.decode_image(images, chanel=3)

def init():
  init_op = tf.group(tf.global_variables_initializer(),
                     tf.local_variables_initializer())
  sess.run(init_op)

#TODO move imgs2features to ImageModel
bad_imgs = []
def deal_imgs(imgs, pics):
  try:
    sess.run(op, feed_dict={images_feed: imgs})
    return pics
  except Exception:
    good_pics = []
    for i, img in enumerate(imgs):
      try:
        sess.run(op, feed_dict={images_feed : [img]})[0])
        good_pics.append(pics[i])
      except Exception:
        print('!Bad image:', pics[i], file=sys.stderr)
        if FLAGS.show_decode_error:
          print(traceback.format_exc(), file=sys.stderr)
        bad_imgs.append(pics[i])
    return good_pics

all_good_pics = set()
def write_features(imgs, pics):
  pics = deal_imgs(imgs, pics)
  for pic in pics:
    all_good_pics.add(pic)
    print(pic)

def run():
  imgs = []
  all_pics = []
  pics = []
  num_deal = 0
  for line in sys.stdin:
    l = line.strip().split('\t')
    pic, imgtext = l[FLAGS.key_index], l[FLAGS.val_index]
    imgs.append(urllib.unquote_plus(imgtext))
    all_pics.append(pic)
    pics.append(pic)
    if len(imgs) == FLAGS.batch_size:
      write_features(imgs, pics)
      num_deal += len(pics)
      print('convert: %d %f'%(num_deal, num_deal / len(all_pics)), file=sys.stderr)
      imgs = []
      pics = []
  if imgs:
    write_features(imgs, pics)
    num_deal += len(pics)
    print('convert: %d %f'%(num_deal, num_deal / len(all_pics)), file=sys.stderr)

  print(bad_imgs, file=sys.stderr)
  print('All %d, Bad %d, BadRatio: %f'%(len(all_pics), len(bad_imgs), len(bad_imgs) / len(all_pics)), file=sys.stderr)

  cPickle.dump(all_good_pics, open(FLAGS.out, 'wb'))


def main(_):
  global op 
  op = build_graph(images_feed, FLAGS.format)

  global sess
  sess = tf.Session()
  
  init()
  run()


if __name__ == '__main__':
  tf.app.run()
