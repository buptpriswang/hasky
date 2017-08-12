#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   encoder_factory.py
#        \author   chenghuige  
#          \date   2016-12-24 09:16:56.277231
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS
  
from deepiu.seq2seq import bow_encoder, cnn_encoder
from deepiu.seq2seq.rnn_encoder import RnnEncoder

class EncoderType:
	bow = 'bow'
	rnn = 'rnn'
	cnn = 'cnn'

def get_encoder(encoder_type, is_training=True, is_predict=False):
  if encoder_type == EncoderType.bow:
  	return bow_encoder
  elif encoder_type == EncoderType.rnn:
  	return RnnEncoder(is_training, is_predict)
  elif encoder_type == EncoderType.cnn:
  	return cnn_encoder