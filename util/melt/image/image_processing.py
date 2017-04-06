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


import tensorflow as tf

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


def process_image(encoded_image,
                  is_training,
                  height,
                  width,
                  resize_height=346,
                  resize_width=346,
                  distort=True,
                  image_format="jpeg"):
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
  with tf.name_scope("decode", values=[encoded_image]):
    if image_format == "jpeg":
      image = tf.image.decode_jpeg(encoded_image, channels=3)
    elif image_format == "png":
      image = tf.image.decode_png(encoded_image, channels=3)
    else:
      raise ValueError("Invalid image format: %s" % image_format)
  image = tf.image.convert_image_dtype(image, dtype=tf.float32)

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
    def construct_fn(image_feature, 
                     height, 
                     width, 
                     is_training=False,
                     resize_height=346,
                     resize_width=346,
                     distort=True,
                     image_format="jpeg",
                     reuse=None):
      image_feature = tf.map_fn(lambda img: process_image(img,
                                                          is_training=False,
                                                          height=height, 
                                                          width=width,
                                                          resize_height=resize_height,
                                                          resize_width=resize_width,
                                                          distort=distort,
                                                          image_format=image_format), 
                                image_feature,
                                dtype=tf.float32)

      image_feature = melt.image.image_embedding.inception_v3(
        image_feature,
        trainable=False,
        is_training=False,
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
