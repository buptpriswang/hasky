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
flags.DEFINE_string('image_model_name', 'InceptionResnetV2', '')
flags.DEFINE_boolean('slim_preprocessing', True, '')
flags.DEFINE_integer('image_width', 299, 'default width of inception')
flags.DEFINE_integer('image_height', 299, 'default height of inception')
flags.DEFINE_string('image_checkpoint_file', '/home/gezi/data/image_model_check_point/inception_resnet_v2_2016_08_30.ckpt', '')
flags.DEFINE_integer('batch_size', 512, 'for safe use 256, 512 will be fine, 600 will oom for gtx1080')
flags.DEFINE_boolean('show_decode_error', False, '')

import sys, os, glob, traceback
import melt
import urllib
import numpy as np

sess = None
images_feed =  tf.placeholder(tf.string, [None,], name='images')
img2feautres_op = None

def build_graph(images):
  melt.apps.image_processing.init(FLAGS.image_model_name)
  return melt.apps.image_processing.image_processing_fn(images,  
                                                        height=FLAGS.image_height, 
                                                        width=FLAGS.image_width,
                                                        slim_preprocessing=FLAGS.slim_preprocessing)

def init():
  init_op = tf.group(tf.global_variables_initializer(),
                     tf.local_variables_initializer())
  sess.run(init_op)
  #---load inception model check point file
  init_fn = melt.image.create_image_model_init_fn(FLAGS.image_model_name, FLAGS.image_checkpoint_file)

  init_fn(sess)

#TODO move imgs2features to ImageModel
bad_imgs = []

#if fail one by one time:
# All 5283, Bad 180, BadRatio: 0.034072
# real  1m46.274s
# user  2m12.868s
# sys 0m8.713s
# gezi@localhost:/data2/data/product/makeup/tb$ ^C
# gezi@localhost:/data2/data/product/makeup/tb$ wc -l ./tb-pic-inceptionV3/part-00185
# 5103 ./tb-pic-inceptionV3/part-00185

def imgs2features(imgs, pics):
 try:
   return sess.run(img2feautres_op, feed_dict={images_feed: imgs}), pics
 except Exception:
   features = []
   good_pics = []
   for i, img in enumerate(imgs):
     try:
       features.append(sess.run(img2feautres_op, feed_dict={images_feed : [img]})[0])
       good_pics.append(pics[i])
     except Exception:
       print('!Bad image:', pics[i], file=sys.stderr)
       if FLAGS.show_decode_error:
         print(traceback.format_exc(), file=sys.stderr)
       bad_imgs.append(pics[i])
   return features, good_pics

# # All 5283, Bad 180, BadRatio: 0.034072
# # real  1m45.991s
# # user  2m11.235s
# # sys 0m8.862s

# def imgs2features(imgs, pics):
#   assert len(imgs) == len(pics)
#   if not imgs or not pics:
#     return [], []

#   try:
#     return sess.run(img2feautres_op, feed_dict={images_feed: imgs}), pics
#   except Exception:
#     if len(imgs) == 1: 
#       bad_imgs.append(pics[0])
#       print('!Bad image:', pics[0], file=sys.stderr)
#       if FLAGS.show_decode_error:
#         print(traceback.format_exc(), file=sys.stderr)
#       return [], []

#     mid = int(len(imgs) / 2)
#     lfeatures, lpics = imgs2features(imgs[:mid], pics[:mid])
#     rfeatures, rpics = imgs2features(imgs[mid:], pics[mid:])

#     return np.array(list(lfeatures) + list(rfeatures)), lpics + rpics

def write_features(imgs, pics):
  img_features, pics = imgs2features(imgs, pics)
  for pic, img_feature in zip(pics, img_features):
    print(pic, '\x01'.join([str(x) for x in img_feature]), sep='\t')

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


def main(_):
  global img2feautres_op 
  img2feautres_op = build_graph(images_feed)

  global sess
  sess = tf.Session()
  
  init()
  run()


if __name__ == '__main__':
  tf.app.run()
