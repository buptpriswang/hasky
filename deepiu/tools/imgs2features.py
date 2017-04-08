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
flags.DEFINE_string('image_dir', '/home/gezi/data/flickr/flickr30k-images', 'input images dir')
flags.DEFINE_string('image_model_name', 'InceptionV3', '')
flags.DEFINE_integer('image_width', 299, 'default width of inception v3')
flags.DEFINE_integer('image_height', 299, 'default height of inception v3')
flags.DEFINE_string('image_checkpoint_file', '/home/gezi/data/inceptionv3/inception_v3.ckpt', '')
flags.DEFINE_integer('batch_size', 512, '')

import sys, os
import melt


sess = None
images_feed =  tf.placeholder(tf.string, [None,], name='images')
img2feautres_op = None

def build_graph(images):
  melt.apps.image_processing.init(FLAGS.image_model_name)
  return melt.apps.image_processing.image_processing_fn(images,  
                                                        height=FLAGS.image_height, 
                                                        width=FLAGS.image_width)

def init():
  init_op = tf.group(tf.global_variables_initializer(),
                     tf.local_variables_initializer())
  sess.run(init_op)
  #---load inception model check point file
  init_fn = melt.image.create_image_model_init_fn(FLAGS.image_model_name, FLAGS.image_checkpoint_file)
  init_fn(sess)

def imgs2features(imgs):
  return sess.run(img2feautres_op, feed_dict={images_feed: imgs})

def write_features(pics, imgs):
  img_features = imgs2features(imgs)
  for pic, img_feature in zip(pics, img_features):
    print(pic, '\t', '\t'.join([str(x) for x in img_feature]))

def run():
  imgs = []
  pics = []
  for pic in sys.stdin:
    pic = pic.strip()
    pic_path = FLAGS.image_dir + '/' + pic
    #print(sys.stderr, 'pic_path', pic_path)
    imgs.append(melt.image.read_image(pic_path))
    pics.append(pic)
    if len(imgs) == FLAGS.batch_size:
      write_features(pics, imgs)
      imgs = []
      pics = []
  if imgs:
    write_features(pics, imgs)

def main(_):
  global img2feautres_op 
  img2feautres_op = build_graph(images_feed)

  global sess
  sess = tf.Session()
  
  init()
  run()

if __name__ == '__main__':
  tf.app.run()