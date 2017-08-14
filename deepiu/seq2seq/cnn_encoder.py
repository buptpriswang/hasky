#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   cnn_encoder.py
#        \author   chenghuige  
#          \date   2016-12-24 00:00:43.524179
#   \Description  
# ==============================================================================

	
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS
	
#1 -------------https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/learn/text_classification_cnn.py
#may be change the interfance to encode(inputs) ?
#conv2d too much mem to consume!
def encode_conv2d(word_vectors):
  """2 layer ConvNet to predict from sequence of words to a class."""
  N_FILTERS = 10 #filters is as output
  WINDOW_SIZE = 3
  #FILTER_SHAPE1 = [WINDOW_SIZE, EMBEDDING_SIZE]
  FILTER_SHAPE2 = [WINDOW_SIZE, N_FILTERS]
  POOLING_WINDOW = 4
  POOLING_STRIDE = 2

  #[batch_size, length, emb_dim]
  word_vectors = tf.expand_dims(word_vectors, -1)
  emb_dim = word_vectors.get_shape()[-1]
  FILTER_SHAPE1 = [WINDOW_SIZE, emb_dim]
  with tf.variable_scope('CNN_Layer1'):
    # Apply Convolution filtering on input sequence.
    conv1 = tf.layers.conv2d(
        word_vectors,
        filters=N_FILTERS,
        kernel_size=FILTER_SHAPE1,
        padding='VALID',
        # Add a ReLU for non linearity.
        activation=tf.nn.relu)
    print('-----conv1', conv1)
    # Max pooling across output of Convolution+Relu.
    pool1 = tf.layers.max_pooling2d(
        conv1,
        pool_size=POOLING_WINDOW,
        strides=POOLING_STRIDE,
        padding='SAME')
    print('-----pool1', pool1)
    # Transpose matrix so that n_filters from convolution becomes width.
    pool1 = tf.transpose(pool1, [0, 1, 3, 2])
    print('-----pool1', pool1)
  with tf.variable_scope('CNN_Layer2'):
    # Second level of convolution filtering.
    conv2 = tf.layers.conv2d(
        pool1,
        filters=N_FILTERS,
        kernel_size=FILTER_SHAPE2,
        padding='VALID')
    print('-----conv2', conv2)
    # Max across each filter to get useful features for classification.
    pool2 = tf.squeeze(tf.reduce_max(conv2, 1), squeeze_dims=[1])

    print('-----pool2', pool2)
    #return pool2
    return tf.layers.dense(pool2, emb_dim)


#2---------- #http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/
# #https://github.com/dennybritz/cnn-text-classification-tf

try:
  import conf
  from conf import TEXT_MAX_WORDS
except Exception:
  from deepiu.image_caption.conf import TEXT_MAX_WORDS

def encode_conv2_2(word_vectors):
	dropout_keep_prob = 0.8
	filter_sizes = [3, 5, 7]
	#OOM when allocating tensor with shape[64,128,94,128]
	num_filters = 128

	word_vectors = tf.expand_dims(word_vectors, -1)
	emb_dim = word_vectors.get_shape()[-1]

	pooled_outputs = []
	for i, filter_size in enumerate(filter_sizes):
		with tf.name_scope("conv-maxpool-%s" % filter_size):
			# Convolution Layer
			conv = tf.layers.conv2d(
						word_vectors,
						filters=num_filters,
						kernel_size=[filter_size, emb_dim],
						padding='VALID',
						# Add a ReLU for non linearity.
						activation=tf.nn.relu,
						name='conv2d_{}_{}'.format(i, filter_size))
			print('-----------conv:', conv)
			# Maxpooling over the outputs
# 			  File "/home/gezi/mine/hasky/deepiu/seq2seq/cnn_encoder.py", line 101, in encode
#     padding='VALID')
#   File "/usr/lib/python2.7/site-packages/tensorflow/python/layers/pooling.py", line 438, in max_pooling2d
#     name=name)
#   File "/usr/lib/python2.7/site-packages/tensorflow/python/layers/pooling.py", line 405, in __init__
#     padding=padding, data_format=data_format, name=name, **kwargs)
#   File "/usr/lib/python2.7/site-packages/tensorflow/python/layers/pooling.py", line 258, in __init__
#     self.pool_size = utils.normalize_tuple(pool_size, 2, 'pool_size')
#   File "/usr/lib/python2.7/site-packages/tensorflow/python/layers/utils.py", line 87, in normalize_tuple
#     int(single_value)
# TypeError: __int__ returned non-int (type NoneType)
			pooled = tf.layers.max_pooling2d(
							conv,
							#pool_size=[word_vectors.get_shape()[1] - filter_size + 1, 1],
							pool_size=[TEXT_MAX_WORDS - filter_size + 1, 1],
							strides=1,
							padding='VALID')
			print('------------pooled', pooled)
			pooled_outputs.append(pooled)

 	with tf.variable_scope('concat_layer'):
		# Combine all the pooled features
		num_filters_total = num_filters * len(filter_sizes)
		#pooled_outputs [batch_size, 1, 1, num_filters]
		#h_pool [batch_size, 1, ,1, num_filters_total]
		h_pool = tf.concat(pooled_outputs, 3)
		#we combine them into one long feature vector of shape [batch_size, num_filters_total]. 
		#Using -1 in tf.reshape tells TensorFlow to flatten the dimension when possible
		h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

		# Add dropout
		h_drop = tf.nn.dropout(h_pool_flat, dropout_keep_prob)
		#dropout_keep_prob = tf.Variable(tf.constant(options.dropout_keep_prob), name="b")
		#h_drop = tf.nn.dropout(h_pool_flat, dropout_keep_prob)
		return tf.layers.dense(h_drop, emb_dim)


#3--------------https://richliao.github.io/supervised/classification/2016/11/26/textclassifier-convolutional/
# l_cov1= Conv1D(128, 5, activation='relu')(embedded_sequences)
# l_pool1 = MaxPooling1D(5)(l_cov1)
# l_cov2 = Conv1D(128, 5, activation='relu')(l_pool1)
# l_pool2 = MaxPooling1D(5)(l_cov2)
# l_cov3 = Conv1D(128, 5, activation='relu')(l_pool2)
# l_pool3 = MaxPooling1D(35)(l_cov3)  # global max pooling
# l_flat = Flatten()(l_pool3)
# l_dense = Dense(128, activation='relu')(l_flat)
# preds = Dense(2, activation='softmax')
#--works but top1 recall 0.35 while bow can be 0.5

#first exp
#  5 5
#  5 2
#  5 2     35% top1 recall
#2  
# (5,5) (3,3) (2,2) max .. do worse
#3

#--35%
def encode_1(word_vectors):
	num_filters = 128
	x = tf.layers.conv1d(word_vectors, num_filters, 5, padding='same', activation=tf.nn.relu)
	x = tf.layers.max_pooling1d(x, 5, 5)
	x = tf.layers.conv1d(x, num_filters, 5, padding='same', activation=tf.nn.relu)
	x = tf.layers.max_pooling1d(x, 2, 2)
	x = tf.layers.conv1d(x, num_filters, 5, padding='same', activation=tf.nn.relu)
	x = tf.layers.max_pooling1d(x, 2, 2)
	#actually here only one .. [batch_size, 1, dim]
	x = tf.reduce_max(x, 1)
	return x

#less good then encode_1
def encode_2(word_vectors):
	num_filters = 128
	x = tf.layers.conv1d(word_vectors, num_filters, 5, padding='same', activation=tf.nn.relu)
	x = tf.layers.max_pooling1d(x, 5, 5)
	x = tf.layers.conv1d(x, num_filters, 3, padding='same', activation=tf.nn.relu)
	x = tf.layers.max_pooling1d(x, 3, 3)
	x = tf.layers.conv1d(x, num_filters, 2, padding='same', activation=tf.nn.relu)
	x = tf.layers.max_pooling1d(x, 2, 2)
	x = tf.reduce_max(x, 1)
	return x


#this is conv only
def encode_3(word_vectors):
	num_filters = 128
	x = tf.layers.conv1d(word_vectors, num_filters, 5, padding='same', activation=tf.nn.relu)
	#x = tf.layers.max_pooling1d(x, 5, 5)
	x = tf.layers.conv1d(x, num_filters, 3, padding='same', activation=tf.nn.relu)
	#x = tf.layers.max_pooling1d(x, 3, 3)
	x = tf.layers.conv1d(x, num_filters, 2, padding='same', activation=tf.nn.relu)
	#x = tf.layers.max_pooling1d(x, 2, 2)
	x = tf.reduce_max(x, 1)
	return x

#use mean, bad..
def encode_4(word_vectors):
	num_filters = 128
	x = tf.layers.conv1d(word_vectors, num_filters, 5, padding='same', activation=tf.nn.relu)
	#x = tf.layers.max_pooling1d(x, 5, 5)
	x = tf.layers.conv1d(x, num_filters, 3, padding='same', activation=tf.nn.relu)
	#x = tf.layers.max_pooling1d(x, 3, 3)
	x = tf.layers.conv1d(x, num_filters, 2, padding='same', activation=tf.nn.relu)
	#x = tf.layers.max_pooling1d(x, 2, 2)
	x = tf.reduce_mean(x, 1)
	return x


"""
Hierarchical ConvNet
"""
#https://github.com/chenghuige/InferSent/blob/master/models.py
def convnet_encode(word_vectors):
	num_filters = 128
	x = tf.layers.conv1d(word_vectors, num_filters, 3, padding='same', activation=tf.nn.relu)
	u1 = tf.reduce_max(x, 1)
	x = tf.layers.conv1d(x, num_filters, 3, padding='same', activation=tf.nn.relu)
	u2 = tf.reduce_max(x, 1)
	x = tf.layers.conv1d(x, num_filters, 3, padding='same', activation=tf.nn.relu)
	u3 = tf.reduce_max(x, 1)
	x = tf.layers.conv1d(x, num_filters, 3, padding='same', activation=tf.nn.relu)
	u4 = tf.reduce_max(x, 1)
	return tf.concat([u1, u2, u3, u4], 1)


def encode(sequence, emb=None, pos_emb=None):
	with tf.variable_scope('cnn_layer'):
		word_vectors = tf.nn.embedding_lookup(emb, sequence) if emb else sequence
		if pos_emb:
			word_vectors += pos_emb
		emb_dim = word_vectors.get_shape()[-1]
		feature = convnet_encode(word_vectors)
		return tf.layers.dense(feature, emb_dim)
