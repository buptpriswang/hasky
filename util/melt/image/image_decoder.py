#!/usr/bin/env python
# ==============================================================================
#          \file   image_decoder.py
#        \author   chenghuige  
#          \date   2017-03-31 12:08:54.242407
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
  
class ImageDecoder(object):
  """Helper class for decoding images in TensorFlow."""

  def __init__(self):
    # Create a single TensorFlow Session for all image decoding calls.
    self._sess = tf.Session()

    # TensorFlow ops for JPEG decoding.
    self._encoded = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._encoded, channels=3)
    self._decode = tf.image.decode_image(self._encoded, channels=3)

  def decode_jpeg(self, encoded_jpeg):
    image = self._sess.run(self._decode_jpeg,
                           feed_dict={self._encoded: encoded_jpeg})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image

  def decode(self, encoded):
    image = self._sess.run(self._decode,
                           feed_dict={self._encoded: encoded})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image