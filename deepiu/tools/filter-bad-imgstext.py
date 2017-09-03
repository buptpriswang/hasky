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
flags.DEFINE_integer('batch_size', 1, 'only decode image 1 is best, and cpu might be faster then gpu, just use multi core')
flags.DEFINE_boolean('show_decode_error', False, '')
flags.DEFINE_string('out', 'good_pics.pkl', '')
flags.DEFINE_string('format', 'jpeg', '')

import sys, os, glob, traceback
import melt
import urllib 

import cPickle 

sess = None
op = None 
images_feed = tf.placeholder(tf.string, [None,], name='images')

def build_graph(images, image_format='jpeg'):
  def process(image, image_format):
    if image_format == 'jpeg':
      return tf.image.decode_jpeg(image, channels=3)
    elif image_format == 'bmp':
      return tf.image.decode_bmp(image, channels=3)
    else:
      return tf.image.decode_image(image, channels=3)
  if FLAGS.batch_size > 1:
    return tf.map_fn(lambda img: process(img, image_format), images, dtype=tf.uint8)
  else:
    return process(images[0], image_format)

def init():
  init_op = tf.group(tf.global_variables_initializer(),
                     tf.local_variables_initializer())
  sess.run(init_op)

#TODO move imgs2features to ImageModel
bad_imgs = []
def deal_imgs(imgs, pics):
  try:
    sess.run(op, feed_dict={images_feed: [urllib.unquote_plus(x) for x in imgs]})
    return pics, imgs
  except Exception:
    if len(imgs) == 1:
      print('!Bad image:', pics[0], file=sys.stderr)
      if FLAGS.show_decode_error:
        print(traceback.format_exc(), file=sys.stderr)
      bad_imgs.append(pics[0])
      return [], []

    good_pics = []
    good_imgs = []
    for i, img in enumerate(imgs):
     try:
       sess.run(op, feed_dict={images_feed : [urllib.unquote_plus(img)]})
       good_pics.append(pics[i])
       good_imgs.append(imgs[i])
     except Exception:
       print('!Bad image:', i,  pics[i], file=sys.stderr)
       if FLAGS.show_decode_error:
         print(traceback.format_exc(), file=sys.stderr)
       bad_imgs.append(pics[i])
    return good_pics, good_imgs
#
# def deal_imgs(imgs, pics):
#   assert len(imgs) == len(pics)
#   if not imgs or not pics:
#     return [], []
#   try:
#     sess.run(op, feed_dict={images_feed: imgs})
#     return pics, imgs
#   except Exception:
#     if len(imgs) == 1: 
#       bad_imgs.append(pics[0])
#       print('!Bad image:', pics[0], file=sys.stderr)
#       if FLAGS.show_decode_error:
#         print(traceback.format_exc(), file=sys.stderr)
#       return [], []

#     mid = int(len(imgs) / 2)
#     lpics, limgs = deal_imgs(imgs[:mid], pics[:mid])
#     rpics, rimgs = deal_imgs(imgs[mid:], pics[mid:])

#     return lpics + rpics, limgs + rimgs 


def write_features(imgs, pics):
  pics, imgs = deal_imgs(imgs, pics)
  for pic, img in zip(pics, imgs):
    print(pic, img, sep='\t')

def run():
  imgs = []
  all_pics = []
  pics = []
  num_deal = 0
  for line in sys.stdin:
    l = line.strip().split('\t')
    pic, imgtext = l[FLAGS.key_index], l[FLAGS.val_index]
    imgs.append(imgtext)
    all_pics.append(pic)
    pics.append(pic)
    if len(imgs) == FLAGS.batch_size:
      write_features(imgs, pics)
      num_deal += len(pics)
      #print('convert: %d %f'%(num_deal, num_deal / len(all_pics)), file=sys.stderr)
      imgs = []
      pics = []
  if imgs:
    write_features(imgs, pics)
    num_deal += len(pics)
    #print('convert: %d %f'%(num_deal, num_deal / len(all_pics)), file=sys.stderr)

  print(bad_imgs, file=sys.stderr)
  print('All %d, Bad %d, BadRatio: %f'%(len(all_pics), len(bad_imgs), len(bad_imgs) / len(all_pics)), file=sys.stderr)

def main(_):
  global op 
  op = build_graph(images_feed, FLAGS.format)

  global sess
  sess = tf.Session()
  
  init()
  run()

if __name__ == '__main__':
  tf.app.run()
