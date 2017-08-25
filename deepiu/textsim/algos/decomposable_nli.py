#!/usr/bin/env python
# ==============================================================================
#          \file   decomposable_nli.py
#        \author   chenghuige  
#          \date   2017-08-25 20:58:08.434879
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS

import melt
logging = melt.logging

import melt.slim

import deepiu

from deepiu.textsim import conf
from deepiu.textsim.conf import TEXT_MAX_WORDS

from deepiu.util import vocabulary
from deepiu.seq2seq import embedding, encoder_factory

from deepiu.util.rank_loss import dot, compute_sim, pairwise_loss, normalize

import numpy as np

#modified version from https://gist.github.com/marekgalovic/a1a4073b917ae1b18dc7413436794dca
#original paper from https://arxiv.org/pdf/1606.01933v1.pdf 
#A Decomposable Attention Model for Natural Language Inference

class DecomposableNLI(object):
	def __init__(self, is_training=True, is_predict=False):
	 	super(DecomposableNLI, self).__init__()

		self.is_training = is_training
		self.is_predict = is_predict

		#TODO move to melt.EmbeddingTrainerBase
		emb_dim = FLAGS.emb_dim
		init_width = 0.5 / emb_dim
		vocabulary.init()
		vocab_size = vocabulary.get_vocab_size()
		self.vocab_size = vocab_size

		# cpu for adgrad optimizer
		if (not FLAGS.word_embedding_file) or glob.glob(FLAGS.model_dir + '/model.ckpt*'):
		  logging.info('Word embedding random init or from model_dir :{} and finetune=:{}'.format(
		      FLAGS.model_dir, FLAGS.finetune_word_embedding))
		  self.emb = embedding.get_embedding_cpu(
		      name='emb', trainable=FLAGS.finetune_word_embedding)
		else:
		  # https://github.com/tensorflow/tensorflow/issues/1570
		  # still adgrad must cpu..
		  # if not fintue emb this will be ok if fintune restart will ok ? must not use word embedding file? os.path.exists(FLAGS.model_dir) ? judge?
		  # or will still try to load from check point ? TODO for safe you could re run by setting word_embedding_file as None or ''
		  logging.info('Loading word embedding from :{} and finetune=:{}'.format(
		      FLAGS.word_embedding_file, FLAGS.finetune_word_embedding))
		  self.emb = melt.load_constant_cpu(
		      FLAGS.word_embedding_file, name='emb', trainable=FLAGS.finetune_word_embedding)


		melt.visualize_embedding(self.emb, FLAGS.vocab)
		if is_training and FLAGS.monitor_level > 0:
		  melt.monitor_embedding(self.emb, vocabulary.vocab, vocab_size)

		self.scope = 'decomposable_nli'
		self.build_train_graph = self.build_graph

		self._attention_output_size = 256
		self._comparison_output_size = 128

  #TODO move to melt.NegPairLossGraph
	def build_graph(self, ltext, rtext, neg_ltext, neg_rtext):
	  with tf.variable_scope(self.scope) as scope:
	    pos_score = self.compute_sim(ltext, rtext)

	    scope.reuse_variables()

	    neg_scores_list = []
	    if neg_rtext is not None:
	      num_negs = neg_rtext.get_shape()[1]
	      for i in xrange(num_negs):
	        neg_scores_i = self.compute_sim(ltext, neg_rtext[:, i, :])
	        neg_scores_list.append(neg_scores_i)
	    if neg_ltext is not None:
	      num_negs = neg_ltext.get_shape()[1]
	      for i in xrange(num_negs):
	        neg_scores_i = self.compute_sim(neg_ltext[:, i, :], rtext)
	        neg_scores_list.append(neg_scores_i)
	  
	    #[batch_size, num_negs]
	    neg_scores = tf.concat(neg_scores_list, 1)
	    #---------------rank loss
	    #[batch_size, 1 + num_negs]
	    scores = tf.concat([pos_score, neg_scores], 1)
	    tf.add_to_collection('scores', scores)

	    loss = pairwise_loss(pos_score, neg_scores)
	    return loss

  #----------below is core algo
  #for train u, v is slim already! assume input.py did this
	def compute_sim(self, u, v):
		print('ori_u,v', u, v)
		u_len = melt.length(u)
		v_len = melt.length(v)
		u, v = self.embeddings_lookup(u, v)
		alpha, beta = self.attention_layer(u, v, u_len, v_len)
		u, v = self.comparison_layer(u, v, alpha, beta, u_len, v_len)

		f = self.aggregation_layer(u, v)

		score = self.fc_layer(f)

		return score

	def fc_layer(self, f):
		return melt.slim.mlp(f,
                 [512, 1],
                 #activation_fn=tf.nn.tanh,
                 #activation_fn=None,
                 activation_fn=tf.nn.relu,
                 scope='fc')

	def embeddings_lookup(self, u, v):
		with tf.name_scope('embeddings_lookup'):
			return tf.nn.embedding_lookup(self.emb, u), tf.nn.embedding_lookup(self.emb, v)

	def attention_layer(self, u, v, u_len, v_len):
		with tf.name_scope('attention_layer'):
			print('uv', u, v)
		 	e_u = tf.layers.dense(u, self._attention_output_size, activation=tf.nn.relu, name='attention_nn')
		 	e_v = tf.layers.dense(v, self._attention_output_size, activation=tf.nn.relu, name='attention_nn', reuse=True)

		 	print('euv', e_u, e_v)
		 	e = tf.matmul(e_u, e_v, transpose_b=True, name='e')
				
			print('-----------------e', tf.transpose(e, [0,2,1]), u_len)
			alpha = tf.matmul(self.masked_softmax(tf.transpose(e, [0,2,1]), u_len), u, name='alpha')
			beta = tf.matmul(self.masked_softmax(e, v_len), v, name='beta')
			
			return alpha, beta

	def masked_softmax(self, values, lengths):
		with tf.name_scope('masked_softmax'):
			#mask = tf.expand_dims(tf.sequence_mask(lengths, tf.reduce_max(lengths), dtype=tf.float32), -2)    
			#mask = tf.expand_dims(tf.sequence_mask(lengths, TEXT_MAX_WORDS, dtype=tf.float32), -2)   
			mask = tf.expand_dims(tf.sequence_mask(lengths, dtype=tf.float32), -2)    
			print('-----mask', mask)
	  	inf_mask = (1 - mask) * -np.inf
	  	inf_mask = tf.where(tf.is_nan(inf_mask), tf.zeros_like(inf_mask), inf_mask)
	  	print('!!!!!values', values)
	  	return tf.nn.softmax(tf.multiply(values, mask) + inf_mask)

	def comparison_layer(self, u, v, alpha, beta, u_len, v_len):
		print('*******alpha,beta', alpha, beta)
		with tf.name_scope('comparison_layer'):
			u_comp = tf.layers.dense(
			    tf.concat([u, beta], 2),
			    self._comparison_output_size,
			    activation=tf.nn.relu,
			    name='comparison_nn'
			)
			u_comp = tf.multiply(
			    #tf.layers.dropout(x1_comp, rate=self.dropout, training=self.is_training),
			    u_comp,
			    #tf.expand_dims(tf.sequence_mask(u_len, tf.reduce_max(u_len), dtype=tf.float32), -1)
			    #tf.expand_dims(tf.sequence_mask(u_len, TEXT_MAX_WORDS, dtype=tf.float32), -1)
			    tf.expand_dims(tf.sequence_mask(u_len, dtype=tf.float32), -1)
			)

			v_comp = tf.layers.dense(
			    tf.concat([v, alpha], 2),
			    self._comparison_output_size,
			    activation=tf.nn.relu,
			    name='comparison_nn',
			    reuse=True
			)
			v_comp = tf.multiply(
			    #tf.layers.dropout(X2_comp, rate=self.dropout, training=self.is_training),
			    v_comp,
			    #tf.expand_dims(tf.sequence_mask(v_len, tf.reduce_max(v_len), dtype=tf.float32), -1)
			    #tf.expand_dims(tf.sequence_mask(v_len, TEXT_MAX_WORDS, dtype=tf.float32), -1)
			    tf.expand_dims(tf.sequence_mask(v_len, dtype=tf.float32), -1)
			)

	  	return u_comp, v_comp
		        
	def aggregation_layer(self, u, v):
		with tf.name_scope('aggregation_layer'):
			u_agg = tf.reduce_sum(u, 1)
			v_agg = tf.reduce_sum(v, 1)

			return tf.concat([u_agg, v_agg], 1)

#TODO move to melt.ElemetwiseTextSimPrecitor
class DecomposableNLIPredictor(DecomposableNLI, melt.PredictorBase):
  def __init__(self):
    melt.PredictorBase.__init__(self)
    DecomposableNLI.__init__(self, is_training=False, is_predict=True)

    self.ltext_feed = None
    self.rtext_feed = None

  #TODO move to TextSimPredictorBase
  #not in init for dont want to put under name space 'model_init'
  def get_ltext_feed(self, text_max_words=TEXT_MAX_WORDS):
    if self.ltext_feed is None:
      self.ltext_feed = tf.placeholder(tf.int32, [None, text_max_words], name='ltext')
    return self.ltext_feed

  def get_rtext_feed(self, text_max_words=TEXT_MAX_WORDS):
    if self.rtext_feed is None:
      self.rtext_feed = tf.placeholder(tf.int32, [None, text_max_words], name='rtext')
    return self.rtext_feed

  def init_predict(self, text_max_words=TEXT_MAX_WORDS):
    if self.ltext_feed is None:
      ltext_feed = self.get_ltext_feed(text_max_words)
      rtext_feed = self.get_rtext_feed(text_max_words)
      score = self.score = self.build_graph(ltext_feed, rtext_feed)
      tf.add_to_collection('score', score)
      tf.get_variable_scope().reuse_variables()
    return score

  def predict(self, ltexts, rtexts):
    feed_dict = {
      self.ltext_feed: ltexts,
      self.rtext_feed: rtexts
      }
    
    score = self.sess.run(self.score, feed_dict)
    score = score.reshape((len(rtexts),))

    return score

  #TODO move to melt.ElementwisePredictorBase
  def elementwise_bulk_predict(self, ltexts, rtexts):
    scores = []
    if len(rtexts) >= len(ltexts):
      for ltext in ltexts:
        stacked_ltexts = np.array([ltext] * len(rtexts))
        score = self.predict(stacked_ltexts, rtexts)
        score = np.squeeze(score) 
        scores.append(score)
    else:
      for rtext in rtexts:
        stacked_rtexts = np.array([rtext] * len(ltexts))
        score = self.predict(ltexts, stacked_rtexts)
        score = np.squeeze(score)
        scores.append(score)
    return np.array(scores)  

  def bulk_predict(self, ltexts, rtexts):
    """
    input images features [m, ] , texts feature [n,] will
    outut [m, n], for each image output n scores for each text  
    """
    return self.elementwise_bulk_predict(ltexts, rtexts)

  
  #TODO may be speedup by cocant [ltex, rtext] as one batch and co forward then split 
  def build_graph(self, ltext, rtext):
    with tf.variable_scope(self.scope):
    	ltext = melt.slim_batch(ltext)
    	rtext = melt.slim_batch(rtext)
    	score = self.compute_sim(ltext, rtext)
    	return score
