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
flags.DEFINE_string('kv_file', '', 'input kv file')
flags.DEFINE_string('image_model_name', 'InceptionV3', '')
flags.DEFINE_integer('image_width', 299, 'default width of inception v3')
flags.DEFINE_integer('image_height', 299, 'default height of inception v3')
flags.DEFINE_string('image_checkpoint_file', '/home/gezi/data/inceptionv3/inception_v3.ckpt', '')
flags.DEFINE_integer('batch_size', 512, '')

import sys, os, glob

from struct import unpack


from image_model import ImageModel

def write_features(model, pics, imgs):
  img_features = model.process(imgs)
  for pic, img_feature in zip(pics, img_features):
    print(pic, '\t'.join([str(x) for x in img_feature]), sep='\t')

def run(model):
  imgs = []
  pics = []
  
  with open(FLAGS.kv_file, 'r') as f:
    buf_key_len = f.read(4)
    while buf_key_len :
      key_len = unpack('i', buf_key_len)[0]
      buf_key = f.read(key_len).strip('\x00')
      val_len = unpack('i', f.read(4))[0]
      buf_val = f.read(val_len)

      pics.append(buf_key)
      imgs.append(buf_val)
      if len(imgs) == FLAGS.batch_size:
        write_features(model, pics, imgs)
        imgs = []
        pics = []

      buf_key_len = f.read(4)

  if imgs:
    write_features(model, pics, imgs)

def main(_):
  model = ImageModel(FLAGS.image_model_name, 
                     FLAGS.image_height, 
                     FLAGS.image_width,
                     FLAGS.image_checkpoint_file)
  run(model)

if __name__ == '__main__':
  tf.app.run()
