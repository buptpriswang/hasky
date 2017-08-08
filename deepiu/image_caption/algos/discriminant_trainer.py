#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   discriminant_trainer.py
#        \author   chenghuige  
#          \date   2016-09-22 22:39:10.084671
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('hidden_size', 1024, 'hidden size, depreciated, use mlp_layer_dims instead')
flags.DEFINE_string('image_mlp_layer_dims', '1024,1024', '')
flags.DEFINE_string('text_mlp_layer_dims', '1024,1024', '')

flags.DEFINE_float('margin', 0.5, 'margin for rankloss when rank_loss is hinge loss')
flags.DEFINE_string('activation', 'relu', 
                    """relu/tanh/sigmoid  seems sigmoid will not work here not convergent
                    and relu slightly better than tanh and convrgence speed faster""")

flags.DEFINE_boolean('bias', False, 'wether to use bias. Not using bias can speedup a bit')
flags.DEFINE_string('loss', 'hinge', 'use hinge(hinge_loss) or cross(cross_entropy_loss) or hinge_cross(subtract then cross)')

flags.DEFINE_boolean('fix_image_embedding', False, '')

flags.DEFINE_boolean('fix_text_embedding', False, '')

flags.DEFINE_boolean('fix_word_embedding', False, '')
flags.DEFINE_boolean('pre_train_word_embedding', False, '')


import functools

import tensorflow.contrib.slim as slim
import melt
logging = melt.logging
import melt.slim 

from deepiu.util import vocabulary
from deepiu.seq2seq import embedding

class DiscriminantTrainer(object):
  """
  Only need to set self.gen_text_feature
  """
  def __init__(self, is_training=True, is_predict=False):
    super(DiscriminantTrainer, self).__init__()
    self.is_training = is_training
    self.is_predict = is_predict
    self.gen_text_feature = None

    logging.info('emb_dim:{}'.format(FLAGS.emb_dim))
    logging.info('margin:{}'.format(FLAGS.margin))

    emb_dim = FLAGS.emb_dim
    init_width = 0.5 / emb_dim
    vocabulary.init()
    vocab_size = vocabulary.get_vocab_size() 
    self.vocab_size = vocab_size
    #if not cpu and on gpu run and using adagrad, will fail  TODO check why
    #also this will be more safer, since emb is large might exceed gpu mem   
    #with tf.device('/cpu:0'):
    #  #NOTICE if using bidirectional rnn then actually emb_dim is emb_dim / 2, because will at last step depth-concatate output fw and bw vectors
    #  self.emb = melt.variable.get_weights_uniform('emb', [vocab_size, emb_dim], -init_width, init_width)
    self.emb = embedding.get_embedding_cpu('emb')
    melt.visualize_embedding(self.emb, FLAGS.vocab)
    if is_training and FLAGS.monitor_level > 0:
      melt.monitor_embedding(self.emb, vocabulary.vocab, vocab_size)

    self.activation = melt.activations[FLAGS.activation]

    #TODO can consider global initiallizer like 
    # with tf.variable_scope("Model", reuse=None, initializer=initializer) 
    #https://github.com/tensorflow/models/blob/master/tutorials/rnn/ptb/ptb_word_lm.py
    self.weights_initializer = tf.random_uniform_initializer(-FLAGS.initializer_scale, FLAGS.initializer_scale)
    self.biases_initialzier = melt.slim.init_ops.zeros_initializer if FLAGS.bias else None

    if not FLAGS.pre_calc_image_feature:
      assert melt.apps.image_processing.image_processing_fn is not None, 'forget melt.apps.image_processing.init()'
      self.image_process_fn = functools.partial(melt.apps.image_processing.image_processing_fn,
                                                height=FLAGS.image_height, 
                                                width=FLAGS.image_width)


    self.image_mlp_layer_dims = [int(x) for x in FLAGS.image_mlp_layer_dims.split(',')]
    self.text_mlp_layer_dims = [int(x) for x in FLAGS.text_mlp_layer_dims.split(',')]

    self.scope = 'image_text_sim'

  def forward_image_layers(self, image_feature):
    if not FLAGS.pre_calc_image_feature:
      image_feature = self.image_process_fn(image_feature)

    dims = self.image_mlp_layer_dims
    return melt.slim.mlp(image_feature, 
                         dims, 
                         self.activation, 
                         weights_initializer=self.weights_initializer,
                         biases_initializer=self.biases_initialzier, 
                         scope='image_mlp')
    #return melt.layers.mlp_nobias(image_feature, FLAGS.hidden_size, FLAGS.hidden_size, self.activation, scope='image_mlp')

  def forward_text_layers(self, text_feature):
    #TODO better config like google/seq2seq us pyymal
    dims = self.text_mlp_layer_dims
    return melt.slim.mlp(text_feature, 
                         dims, 
                         self.activation, 
                         weights_initializer=self.weights_initializer,
                         biases_initializer=self.biases_initialzier, 
                         scope='text_mlp')
    #return melt.layers.mlp_nobias(text_feature, FLAGS.hidden_size, FLAGS.hidden_size, self.activation, scope='text_mlp')

  def forward_text_feature(self, text_feature):
    text_feature = self.forward_text_layers(text_feature)
    #for pointwise comment below
    text_feature = tf.nn.l2_normalize(text_feature, 1)
    return text_feature	

  def forward_text(self, text):
    """
    Args:
    text: batch text [batch_size, max_text_len]
    """
    text_feature = self.gen_text_feature(text, self.emb)
    text_feature = self.forward_text_feature(text_feature)
    return text_feature

  def forward_image_feature(self, image_feature):
    """
    Args:
      image: batch image [batch_size, image_feature_len]
    """
    image_feature = self.forward_image_layers(image_feature)
    
    #for point wise comment below
    image_feature = tf.nn.l2_normalize(image_feature, 1)

    return image_feature

  def compute_image_text_sim(self, normed_image_feature, normed_text_feature):
    #[batch_size, hidden_size]
    if FLAGS.fix_image_embedding:
      normed_image_feature = tf.stop_gradient(normed_image_feature)

    if FLAGS.fix_text_embedding:
      #not only stop internal text ebmeddding but also mlp part so fix final text embedding
      normed_text_feature = tf.stop_gradient(normed_text_feature)
    
    #for point wise comment below
    #[batch_size,1] <= [batch_size, hidden_size],[batch_size, hidden_size]
    return melt.element_wise_cosine(normed_image_feature, normed_text_feature, nonorm=True)


    #point wise
    #return -tf.losses.mean_squared_error(normed_image_feature, normed_text_feature, reduction='none')

  def build_graph(self, image_feature, text, neg_text, neg_image=None, lookup_negs_once=False):
    """
    Args:
    image_feature: [batch_size, IMAGE_FEATURE_LEN]
    text: [batch_size, MAX_TEXT_LEN]
    neg_text: [batch_size, num_negs, MAXT_TEXT_LEN]
    neg_image: None or [batch_size, num_negs, IMAGE_FEATURE_LEN]
    """
    assert (neg_text is not None) or (neg_image is not None)
    with tf.variable_scope(self.scope):
      #-------------get image feature
      #[batch_size, hidden_size] <= [batch_size, IMAGE_FEATURE_LEN] 
      image_feature = self.forward_image_feature(image_feature)
      #--------------get image text sim as pos score
      #[batch_size, emb_dim] -> [batch_size, text_MAX_WORDS, emb_dim] -> [batch_size, emb_dim]
      #text_feature = self.gen_text_feature(text, self.emb)
      text_feature = self.forward_text(text)

      pos_score = self.compute_image_text_sim(image_feature, text_feature)
      
      #--------------get image neg texts sim as neg scores
      #[batch_size, num_negs, text_MAX_WORDS, emb_dim] -> [batch_size, num_negs, emb_dim]
      tf.get_variable_scope().reuse_variables()
      neg_scores_list = []
      if neg_text is not None:
        if lookup_negs_once:
          neg_text_feature = self.forward_text(neg_text)
    
        num_negs = neg_text.get_shape()[1]
        for i in xrange(num_negs):
          if lookup_negs_once:
            neg_text_feature_i = neg_text_feature[:, i, :]
          else:
            neg_text_feature_i = self.forward_text(neg_text[:, i, :])
          neg_scores_i = self.compute_image_text_sim(image_feature, neg_text_feature_i)
          neg_scores_list.append(neg_scores_i)
      if neg_image is not None:
        num_negs = neg_image.get_shape()[1]
        for i in xrange(num_negs):
          neg_image_feature_i = self.forward_image_feature(neg_image[:, i, :])
          neg_scores_i =  self.compute_image_text_sim(neg_image_feature_i, text_feature)
          neg_scores_list.append(neg_scores_i)

      #[batch_size, num_negs]
      neg_scores = tf.concat(neg_scores_list, 1)

      #---------------rank loss
      #[batch_size, 1 + num_negs]
      scores = tf.concat([pos_score, neg_scores], 1)
      #may be turn to prob is and show is 
      #probs = tf.sigmoid(scores)

      if FLAGS.loss == 'hinge':
        loss = melt.losses.hinge(pos_score, neg_scores, FLAGS.margin)
      elif FLAGS.loss == 'cross':
        #will conver to 0-1 logits int melt.cross_entropy_loss http://www.wildml.com/2016/07/deep-learning-for-chatbots-2-retrieval-based-model-tensorflow/
        loss = melt.losses.cross_entropy(scores, num_negs)
      #point losss is bad here for you finetune both text and image embedding, all 0 vec will loss minimize..
      #if use point loss you need to fix text embedding
      elif FLAGS.loss == 'point': 
        loss = tf.reduce_mean(1.0 - pos_score)
      else: 
        loss = melt.losses.hinge_cross(pos_score, neg_scores)
      
      tf.add_to_collection('scores', scores)
    return loss

  
