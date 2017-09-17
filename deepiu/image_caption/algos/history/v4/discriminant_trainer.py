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

flags.DEFINE_string('image_mlp_dims', '1024,1024', '')
flags.DEFINE_string('text_mlp_dims', '1024,1024', '')


flags.DEFINE_boolean('fix_image_embedding', False, 'image part all fixed, so just fintune text part')
flags.DEFINE_boolean('fix_text_embedding', False, 'text part all fixed, so just fintune image part')


import functools, glob

import tensorflow.contrib.slim as slim
import melt
logging = melt.logging
import melt.slim 

from deepiu.util import vocabulary
from deepiu.seq2seq import embedding, encoder_factory
from deepiu.util.rank_loss import dot, compute_sim, pairwise_loss, normalize

class DiscriminantTrainer(object):
  def __init__(self, encoder_type='bow', is_training=True, is_predict=False):
    super(DiscriminantTrainer, self).__init__()
    self.is_training = is_training
    self.is_predict = is_predict

    logging.info('emb_dim:{}'.format(FLAGS.emb_dim))
    logging.info('margin:{}'.format(FLAGS.margin))

    self.encoder = encoder_factory.get_encoder(encoder_type, is_training, is_predict)
    self.encoder_type = encoder_type

    emb_dim = FLAGS.emb_dim
    init_width = 0.5 / emb_dim
    vocabulary.init()
    vocab_size = vocabulary.get_vocab_size() 
    self.vocab_size = vocab_size
    self.emb = embedding.get_or_restore_embedding_cpu()

    melt.visualize_embedding(self.emb, FLAGS.vocab)
    if is_training and FLAGS.monitor_level > 0:
      melt.monitor_embedding(self.emb, vocabulary.vocab, vocab_size)

    self.activation = melt.activations[FLAGS.activation]

    #TODO can consider global initiallizer like 
    # with tf.variable_scope("Model", reuse=None, initializer=initializer) 
    #https://github.com/tensorflow/models/blob/master/tutorials/rnn/ptb/ptb_word_lm.py
    self.weights_initializer = tf.random_uniform_initializer(-FLAGS.initializer_scale, FLAGS.initializer_scale)
    self.biases_initialzier = melt.slim.init_ops.zeros_initializer if FLAGS.bias else None

    self.image_process_fn = lambda x: x
    if not FLAGS.pre_calc_image_feature:
      assert melt.apps.image_processing.image_processing_fn is not None, 'forget melt.apps.image_processing.init()'
      self.image_process_fn = functools.partial(melt.apps.image_processing.image_processing_fn,
                                                height=FLAGS.image_height, 
                                                width=FLAGS.image_width,
                                                trainable=FLAGS.finetune_image_model,
                                                is_training=is_training,
                                                random_crop=FLAGS.random_crop_image,
                                                finetune_end_point=FLAGS.finetune_end_point,
                                                distort=FLAGS.distort_image)  

    self.image_mlp_dims = [int(x) for x in FLAGS.image_mlp_dims.split(',')] if FLAGS.image_mlp_dims is not '0' else None
    self.text_mlp_dims = [int(x) for x in FLAGS.text_mlp_dims.split(',')] if FLAGS.text_mlp_dims is not '0' else None

    self.scope = 'image_text_sim'

  def gen_text_feature(self, text, emb):
    """
    this common interface, ie may be lstm will use other way to gnerate text feature
    """
    text_feature = self.encoder.encode(text, emb)
    #rnn will return encode_feature, state
    if isinstance(text_feature, tuple):
      text_feature = text_feature[0]
    return text_feature

  def encoder_words_importance(self, text, emb):
    try:
      return self.encoder.words_importance_encode(text, emb=emb)
    except Exception:
      return None

  def forward_image_layers(self, image_feature):
    image_feature = self.image_process_fn(image_feature)
    dims = self.image_mlp_dims
    if not dims:
      return image_feature

    return melt.slim.mlp(image_feature, 
                         dims, 
                         self.activation, 
                         weights_initializer=self.weights_initializer,
                         biases_initializer=self.biases_initialzier, 
                         scope='image_mlp')
    #return melt.layers.mlp_nobias(image_feature, FLAGS.hidden_size, FLAGS.hidden_size, self.activation, scope='image_mlp')

  def forward_text_layers(self, text_feature):
    #TODO better config like google/seq2seq us pyymal
    dims = self.text_mlp_dims
    if not dims:
      return text_feature

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
    #must be -1 not 1 for num_negs might > 1 if lookup onece..
    text_feature = normalize(text_feature)
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
    image_feature = normalize(image_feature)

    return image_feature

  def compute_image_text_sim(self, normed_image_feature, normed_text_feature):
    #[batch_size, hidden_size]
    if FLAGS.fix_image_embedding:
      normed_image_feature = tf.stop_gradient(normed_image_feature)

    if FLAGS.fix_text_embedding:
      #not only stop internal text ebmeddding but also mlp part so fix final text embedding
      normed_text_feature = tf.stop_gradient(normed_text_feature)
    
    return compute_sim(normed_image_feature, normed_text_feature)

  def build_train_graph(self, image_feature, text, neg_image_feature, neg_text):
    if self.encoder_type == 'bow':
      return self.build_graph(image_feature, text, neg_image_feature, neg_text, lookup_negs_once=True)
    else:
      return self.build_graph(image_feature, text, neg_image_feature, neg_text)

  def build_graph(self, image_feature, text, neg_image_feature, neg_text, lookup_negs_once=False):
    """
    Args:
    image_feature: [batch_size, IMAGE_FEATURE_LEN]
    text: [batch_size, MAX_TEXT_LEN]
    neg_text: [batch_size, num_negs, MAXT_TEXT_LEN]
    neg_image_feature: None or [batch_size, num_negs, IMAGE_FEATURE_LEN]
    """
    assert (neg_text is not None) or (neg_image_feature is not None)
    with tf.variable_scope(self.scope) as scope:
      #-------------get image feature
      #[batch_size, hidden_size] <= [batch_size, IMAGE_FEATURE_LEN] 
      image_feature = self.forward_image_feature(image_feature)
      #--------------get image text sim as pos score
      #[batch_size, emb_dim] -> [batch_size, text_MAX_WORDS, emb_dim] -> [batch_size, emb_dim]
      #text_feature = self.gen_text_feature(text, self.emb)
      text_feature = self.forward_text(text)

      pos_score = self.compute_image_text_sim(image_feature, text_feature)
      
      scope.reuse_variables()
      
      #--------------get image neg texts sim as neg scores
      #[batch_size, num_negs, text_MAX_WORDS, emb_dim] -> [batch_size, num_negs, emb_dim]  
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
      if neg_image_feature is not None:
        num_negs = neg_image_feature.get_shape()[1]
        for i in xrange(num_negs):
          neg_image_feature_feature_i = self.forward_image_feature(neg_image_feature[:, i, :])
          neg_scores_i =  self.compute_image_text_sim(neg_image_feature_feature_i, text_feature)
          neg_scores_list.append(neg_scores_i)

      #[batch_size, num_negs]
      neg_scores = tf.concat(neg_scores_list, 1)
      #[batch_size, 1 + num_negs]
      scores = tf.concat([pos_score, neg_scores], 1)
      tf.add_to_collection('scores', scores)

      loss = pairwise_loss(pos_score, neg_scores)
    return loss

  
