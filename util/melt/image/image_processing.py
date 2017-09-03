# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
Copy from google im2txt
Helper functions for image preprocessing.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_image_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
#from tensorflow.python.ops import variable

import tensorflow as tf

import tensorflow.contrib.slim as slim
#TODO must set PYTHON_PATH for models/slim
from preprocessing import preprocessing_factory
from nets import nets_factory
#from melt.slim import base_nets_factory 
import gezi
  

def read_image(image_path):
  with tf.gfile.FastGFile(image_path, "r") as f:
    encoded_image = f.read()
  return encoded_image

def create_image_model_init_fn(image_model_name, image_checkpoint_file):
  inception_variables = tf.get_collection(
    tf.GraphKeys.GLOBAL_VARIABLES, scope=image_model_name)
  saver = tf.train.Saver(inception_variables)
  def restore_fn(sess):
    tf.logging.info("Restoring image variables from checkpoint file %s",
                        image_checkpoint_file)
    saver.restore(sess, image_checkpoint_file)
  return restore_fn

def distort_image(image):
  """Perform random distortions on an image.

  Args:
    image: A float32 Tensor of shape [height, width, 3] with values in [0, 1).

  Returns:
    distorted_image: A float32 Tensor of shape [height, width, 3] with values in
      [0, 1].
  """
  # Randomly flip horizontally.
  with tf.name_scope("flip_horizontal", values=[image]):
    image = tf.image.random_flip_left_right(image)

  # Randomly distort the colors based on thread id.
  with tf.name_scope("distort_color", values=[image]):
    def _distort_image_fn1(image):
      image = tf.image.random_brightness(image, max_delta=32. / 255.)
      image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      image = tf.image.random_hue(image, max_delta=0.032)
      image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
      return image 

    def _distort_image_fn2(image):
      image = tf.image.random_brightness(image, max_delta=32. / 255.)
      image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
      image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      image = tf.image.random_hue(image, max_delta=0.032)
      return image 

    p_order = tf.random_uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
    pred = tf.less(p_order, 0.5)
    image = tf.cond(pred, lambda: _distort_image_fn1(image), lambda: _distort_image_fn2(image))

    # The random_* ops do not necessarily clamp.
    image = tf.clip_by_value(image, 0.0, 1.0)

  return image

# #--depreciated just use tf ones
# def decode_image(contents, channels=None, name=None):
#   """Convenience function for `decode_gif`, `decode_jpeg`, and `decode_png`.
#   Detects whether an image is a GIF, JPEG, or PNG, and performs the appropriate 
#   operation to convert the input bytes `string` into a `Tensor` of type `uint8`.

#   Note: `decode_gif` returns a 4-D array `[num_frames, height, width, 3]`, as 
#   opposed to `decode_jpeg` and `decode_png`, which return 3-D arrays 
#   `[height, width, num_channels]`. Make sure to take this into account when 
#   constructing your graph if you are intermixing GIF files with JPEG and/or PNG 
#   files.

#   Args:
#     contents: 0-D `string`. The encoded image bytes.
#     channels: An optional `int`. Defaults to `0`. Number of color channels for 
#       the decoded image.
#     name: A name for the operation (optional)
    
#   Returns:
#     `Tensor` with type `uint8` with shape `[height, width, num_channels]` for 
#       JPEG and PNG images and shape `[num_frames, height, width, 3]` for GIF 
#       images.
#   """
#   with ops.name_scope(name, 'decode_image') as scope: 
#     #return gen_image_ops.decode_jpeg(contents, channels)
#     if channels not in (None, 0, 1, 3):
#       raise ValueError('channels must be in (None, 0, 1, 3)')
#     #substr = string_ops.substr(contents, 0, 4)
#     substr = string_ops.substr(contents, 0, 1)

#     def _gif():
#       raise ValueError('not jpeg or png')
#       # Create assert op to check that bytes are GIF decodable
#       #is_gif = math_ops.equal(substr, b'\x47\x49\x46\x38', name='is_gif')
#       is_gif = math_ops.equal(substr, b'\x47', name='is_gif')

#       decode_msg = 'Unable to decode bytes as JPEG, PNG, or GIF'
#       assert_decode = control_flow_ops.Assert(is_gif, [decode_msg])
#       # Create assert to make sure that channels is not set to 1
#       # Already checked above that channels is in (None, 0, 1, 3)
#       gif_channels = 0 if channels is None else channels
#       good_channels = math_ops.not_equal(gif_channels, 1, name='check_channels')
#       channels_msg = 'Channels must be in (None, 0, 3) when decoding GIF images'
#       assert_channels = control_flow_ops.Assert(good_channels, [channels_msg])
#       with ops.control_dependencies([assert_decode, assert_channels]):
#         return gen_image_ops.decode_gif(contents)
    
#     def _png():
#       return gen_image_ops.decode_png(contents, channels)
    
#     def check_png():
#       #is_png = math_ops.equal(substr, b'\211PNG', name='is_png')
#       is_png = math_ops.equal(substr, b'\x89', name='is_png')
#       return control_flow_ops.cond(is_png, _png, _gif, name='cond_png')
    
#     def _jpeg():
#       return gen_image_ops.decode_jpeg(contents, channels)

#     #is_jpeg = math_ops.equal(substr, b'\xff\xd8\xff\xe0', name='is_jpeg')
#     is_jpeg = math_ops.equal(substr, b'\xff', name='is_jpeg')
#     try:
#       return control_flow_ops.cond(is_jpeg, _jpeg, check_png, name='cond_jpeg')
#     except Exception:
#       #this is unsafe, since might not be jpeg format will raise error in c++ code, unable to catch
#       return gen_image_ops.decode_jpeg(contents, channels, try_recover_truncated=True, acceptable_fraction=10)

def decode_image(contents, channels=3, image_format='jpeg', dtype=None):
  with tf.name_scope("decode", values=[contents]):
    if image_format == "jpeg":
      #---TODO this will cause hang.... try_recover_truncated=True, acceptable_fraction=10 will hang...
      #image = tf.image.decode_jpeg(contents, channels=3, try_recover_truncated=True, acceptable_fraction=10)
      image = tf.image.decode_jpeg(contents, channels=channels)
    elif image_format == "png":
      image = tf.image.decode_png(contents, channels=channels)
    else:
      #--why here will casue no size.. might for gif..., for safe just use image_format jpeg? TODO FIXME
      image = tf.image.decode_image(contents, channels=3)
    if dtype:
      image = tf.image.convert_image_dtype(image, dtype=dtype)
    return image

def process_image(encoded_image,
                  is_training,
                  height,
                  width,
                  resize_height=346,
                  resize_width=346,
                  distort=True,
                  image_format=None):
  """Decode an image, resize and apply random distortions.

  In training, images are distorted slightly differently depending on thread_id.

  Args:
    encoded_image: String Tensor containing the image.
    is_training: Boolean; whether preprocessing for training or eval.
    height: Height of the output image.
    width: Width of the output image.
    resize_height: If > 0, resize height before crop to final dimensions.
    resize_width: If > 0, resize width before crop to final dimensions.
    thread_id: Preprocessing thread id used to select the ordering of color
      distortions. There should be a multiple of 2 preprocessing threads.
    image_format: "jpeg" or "png".

  Returns:
    A float32 Tensor of shape [height, width, 3] with values in [-1, 1].

  Raises:
    ValueError: If image_format is invalid.
  """
  # Helper function to log an image summary to the visualizer. Summaries are
  # only logged in thread 0.  TODO summary
  def image_summary(name, image):
    tf.summary.image(name, tf.expand_dims(image, 0))

  # Decode image into a float32 Tensor of shape [?, ?, 3] with values in [0, 1).
  image = decode_image(encoded_image, channels=3, image_format=image_format, dtype=tf.float32)

  #TODO summary
  #image_summary("original_image", image)

  # Resize image.
  assert (resize_height > 0) == (resize_width > 0)
  if resize_height:
    image = tf.image.resize_images(image,
                                   size=[resize_height, resize_width],
                                   method=tf.image.ResizeMethod.BILINEAR)

  # Crop to final dimensions.
  if is_training:
    image = tf.random_crop(image, [height, width, 3])
  else:
    # Central crop, assuming resize_height > height, resize_width > width.
    image = tf.image.resize_image_with_crop_or_pad(image, height, width)
  
  #image_summary("resized_image", image)

  # Randomly distort the image.
  if is_training and distort:
    image = distort_image(image)
  
  #image_summary("final_image", image)

  # Rescale to [-1,1] instead of [0, 1]
  image = tf.subtract(image, 0.5)
  image = tf.multiply(image, 2.0)
  return image


import melt

#----------mainly for escape scope TODO is there bette method exists?
#TODO allow using other models like vgg
def create_image2feature_fn(name='InceptionV3'):
  #NOTICE how this method solve run/ scope problem, scope must before def
  with tf.variable_scope(name) as scope:
    def construct_fn(encoded_image, 
                     height, 
                     width, 
                     trainable=False,
                     is_training=False,
                     resize_height=346,
                     resize_width=346,
                     distort=True,
                     image_format="jpeg",  #for safe just use decode_jpeg
                     reuse=None):
      #decode image .. only accept one image so use map_fn, but seems map_fn is slow, so use as less as possible
      #here decode image + preprocess into one
      image = tf.map_fn(lambda img: process_image(img,
                                                  is_training=is_training,
                                                  height=height, 
                                                  width=width,
                                                  resize_height=resize_height,
                                                  resize_width=resize_width,
                                                  distort=distort,
                                                  image_format=image_format), 
                                encoded_image,
                                dtype=tf.float32)


      #this is the same as slim for inception v3, and TODO can be modified to accept other models
      image_feature = melt.image.image_embedding.inception_v3(
        image,
        trainable=trainable,
        is_training=is_training,
        reuse=reuse,
        scope=scope)

      #if not set this eval_loss = trainer.build_train_graph(eval_image_feature, eval_text, eval_neg_text) will fail
      #but still need to set reuse for melt.image.image_embedding.inception_v3... confused.., anyway now works..
      #with out reuse=True score = predictor.init_predict() will fail, resue_variables not work for it..
      #trainer create function once use it second time(same function) work here(with scope.reuse_variables)
      #predictor create another function, though seem same name same scope, but you need to set reuse=True again!
      #even if use tf.make_template still need this..
      scope.reuse_variables()
      return image_feature

    return construct_fn


def create_image2feature_slim_fn(name='InceptionV3'):
  """
    #NOTICE how this method solve run/ scope problem, scope must before def
    using slim util to create image feature
    You need to set PythonPath to include models/slim or install it if using on hdfs run TODO check
  """
  with tf.variable_scope('') as scope:
    def construct_fn(encoded_image, 
                     height, 
                     width, 
                     trainable=False,
                     is_training=False,
                     resize_height=346,
                     resize_width=346,
                     distort=True,
                     slim_preprocessing=True,
                     image_format="jpeg",  #for safe just use decode_jpeg
                     reuse=None):

      #preprocess_image
      net_name = gezi.to_gnu_name(name)
      #well this is slightly slow and the result is differnt from im2txt inceptionV3 usage result, 
      #use im2txt code seems ok, not sure if slim preprocess will be better! TODO
      #for inception related model I think im2txt process will be fine, for other models not sure TODO
      #using slim preprocessing real 2m45.737s user  3m12.896s sys 0m10.265s 
      #using im2txt processing real 2m46.709s user  3m8.067s sys 0m8.297s  
      #and the final feature will be slightly differnt 
      #one thing intersting is use 2 tf.map_fn(1 decode image, 1 preprocess) will be much slower then use 1 tf.map_fn (decode image+ preprocess)
      if slim_preprocessing:
        preprocessing_fn = preprocessing_factory.get_preprocessing(net_name, is_training=(is_training and distort))
        image = tf.map_fn(lambda img: preprocessing_fn(decode_image(img, image_format=image_format, dtype=tf.float32), height, width),
                          encoded_image, dtype=tf.float32)
      else:
        #im2txt style preprocessing
        image = tf.map_fn(lambda img: process_image(img,
                                                  is_training=is_training,
                                                  height=height, 
                                                  width=width,
                                                  resize_height=resize_height,
                                                  resize_width=resize_width,
                                                  distort=distort,
                                                  image_format=image_format), 
                                encoded_image,
                                dtype=tf.float32)

      #actually final num class layer not used for image feature purpose, but since in check point train using 1001, for simplicity here set 1001
      num_classes = 1001 
      net_fn = nets_factory.get_network_fn(net_name, num_classes=num_classes, is_training=is_training)
      logits, end_points = net_fn(image)
      if 'PreLogitsFlatten' in end_points:
        image_feature = end_points['PreLogitsFlatten']
      elif 'PreLogits' in end_points:
        net = end_points['PreLogits']
        image_feature = slim.flatten(net, scope="flatten")
      else:
        raise ValueError('not found pre logits!')

      #--below is the same for inception v3
      # image_feature = melt.image.image_embedding.inception_v3(
      #   image_feature,
      #   trainable=trainable,
      #   is_training=is_training,
      #   reuse=reuse,
      #   scope=scope)

      #if not set this eval_loss = trainer.build_train_graph(eval_image_feature, eval_text, eval_neg_text) will fail
      #but still need to set reuse for melt.image.image_embedding.inception_v3... confused.., anyway now works..
      #with out reuse=True score = predictor.init_predict() will fail, resue_variables not work for it..
      #trainer create function once use it second time(same function) work here(with scope.reuse_variables)
      #predictor create another function, though seem same name same scope, but you need to set reuse=True again!
      #even if use tf.make_template still need this..
      scope.reuse_variables()
      return image_feature

    return construct_fn

#TODO... not in affect... Still not very clear with make_template and create_scope_now_ (google/seq2seq set this True, default is False)
# it is like above useing scope.reuse_variables() right after.. I suppose

"""
#encode_fn = SomeEncoderModule(...)

## New variables are created in this call.
#output1 = encode_fn(input1)

## No new variables are created here. The variables from the above call are re-used.
## Note how this is different from normal TensorFlow where you would need to use variable scopes.
#output2 = encode_fn(input2)

## Because this is a new instance a second set of variables is created.
#encode_fn2 = SomeEncoderModule(...)
#output3 = encode_fn2(input3)
"""

#this will be safe as build graph as root scope but has problem always building using image_processing?
#from textsum seems not affected, will not create graph if you not use this function
#_image2feature_fn = create_image2feature_fn()
#make this function to be resued/share without manully set scope reuse
#image2feature_fn = tf.make_template('image2feature', _image2feature_fn, create_scope_now_=True)
#image2feature_fn = tf.make_template('image2feature', _image2feature_fn, create_scope_now_=False)
image2feature_fn =  create_image2feature_fn()
