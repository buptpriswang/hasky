#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   predictor.py
#        \author   chenghuige  
#          \date   2016-09-30 15:19:35.984852
#   \Description  
# ==============================================================================
"""
This predictor will read from checkpoint and meta graph, only depend on tensorflow
no other code dpendences, so can help hadoop or online deploy,
and also can run inference without code of building net/graph
"""
  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python import debug as tf_debug

import os, sys
import numpy as np
import melt 

def get_model_dir_and_path(model_dir, model_name=None):
  model_path = model_dir
  ckpt = tf.train.get_checkpoint_state(model_dir)
  if ckpt and ckpt.model_checkpoint_path:
    #model_path = '%s/%s'%(model_dir, os.path.basename(ckpt.model_checkpoint_path)) 
    model_path = os.path.join(model_dir, os.path.basename(ckpt.model_checkpoint_path))
  else:
    model_path = model_dir if model_name is None else os.path.join(model_dir, model_name)
  #if not os.path.exists(model_path):
  #  raise ValueError(model_path)
  return os.path.dirname(model_path), model_path

def get_tensor_from_key(key, index=-1):
  if isinstance(key, str):
    try:
      ops = tf.get_collection(key)
      if len(ops) > 1:
        print('Warning: ops more then 1 for {}, ops:{}, index:{}'.format(key, ops, index))
      return ops[index]
    except Exception:
      print('Warning:', key, ' not find in graph')
      return tf.no_op()
  else:
    return key

class Predictor(object):
  def __init__(self, model_dir=None, meta_graph=None, model_name=None, debug=False, sess=None):
    super(Predictor, self).__init__()
    self.sess = sess
    if self.sess is None:
      #self.sess = tf.InteractiveSession()
      self.sess = melt.get_session() #make sure use one same global/share sess in your graph
      if debug:
        self.sess = tf_debug.LocalCLIDebugWrapperSession(self.sess)
    #ops will be map and internal list like
    #{ text : [op1, op2], text2 : [op3, op4, op5], text3 : [op6] }
    if model_dir is not None:
      self.restore(model_dir, meta_graph, model_name)

  #by default will use last one
  def inference(self, key, feed_dict=None, index=-1):
    if not isinstance(key, (list, tuple)):
      return self.sess.run(get_tensor_from_key(key, index), feed_dict=feed_dict)
    else:
      keys = key 
      if not isinstance(index, (list, tuple)):
        indexes = [index] * len(keys)
      else:
        indexes = index 
      keys = [get_tensor_from_key(key, index) for key,index in zip(keys, indexes)]
      return self.sess.run(keys, feed_dict=feed_dict)

  def predict(self, key, feed_dict=None, index=-1):
    return self.inference(key, feed_dict, index)

  def run(self, key, feed_dict=None):
    return self.sess.run(key, feed_dict)

  def restore(self, model_dir, meta_graph=None, model_name=None, random_seed=None):
    """
    do not need to create graph
    restore graph from meta file then restore values from checkpoint file
    """
    model_dir, model_path = get_model_dir_and_path(model_dir, model_name)
    self.model_path = model_path
    if meta_graph is None:
      meta_graph = '%s.meta'%model_path
    #print('restore from %s'%model_path, file=sys.stderr)
    saver = tf.train.import_meta_graph(meta_graph)
    #print('import graph ok %s'%meta_graph, file=sys.stderr)
    saver.restore(self.sess, model_path)
    print('restore meta grpah and model ok %s'%model_path, file=sys.stderr)
    if random_seed is not None:
      tf.set_random_seed(random_seed)

    #---so maybe do not use num_epochs or not save num_epochs variable!!!! can set but input producer not use, stop by my flow loop
    #---TODO not work remove can run but hang  FIXME add predictor + exact_predictor during train will face
    #@gauravsindhwani , can you still run the code successfully after you remove these two collections since they are actually part of the graph. 
    #I try your way but find the program is stuck after restoring."
    #https://github.com/tensorflow/tensorflow/issues/9747
    #tf.get_default_graph().clear_collection("queue_runners")
    #tf.get_default_graph().clear_collection("local_variables")
    #--for num_epochs not 0
    #tf.get_default_graph().clear_collection("local_variables")
    #self.sess.run(tf.local_variables_initializer())

    #https://stackoverflow.com/questions/44251666/how-to-initialize-tensorflow-variable-that-wasnt-saved-other-than-with-tf-globa
    #melt.initialize_uninitialized_vars(self.sess)

    return self.sess


class SimPredictor(object):
  def __init__(self, 
              model_dir=None, 
              lkey=None,
              rkey=None,
              key='score',
              index=0,
              meta_graph=None, 
              model_name=None, 
              debug=False, 
              sess=None):
    self._predictor = Predictor(model_dir, meta_graph, model_name, debug, sess)
    self._key = key 
    self._index = index

    if lkey is None:
      self._lkey = tf.get_collection('lfeed')[index]
    if rkey is None:
      self._rkey = tf.get_collection('rfeed')[index]

  def inference(self, ltext, rtext=None, key=None, index=None):
    if key is None:
      key = self._key
    if index is None:
      index = self._index
    if rtext is not None:
      feed_dict = {
        self._lkey: ltext,
        self._rkey: rtext
      }
      return self._predictor.inference(key, feed_dict=feed_dict, index=index)
    else:
      feed_dict = {
        self._lkey: ltext
      }
      return self._predictor.inference(key, feed_dict=feed_dict, index=index)


  def predict(self, ltext, rtext=None, key=None, index=None):
    return self.inference(ltext, rtext, key, index)

  def elementwise_predict(self, ltexts, rtexts):
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

  def top_k(self, ltext, rtext, k=1, key=None):
    feed_dict = {
      self._lkey: ltext,
      self._rkey: rtext
    }
    if key is None:
      key = 'nearby'
    try:
      values, indices = self._predictor.inference(key, feed_dict=self._feed_dict, index=self._index)
      return values[:k], indices[:k]
    except Exception:
      score = self.predict(ltext, rtext)
      indices = (-score).argsort()[:k]
      # result
      x = arr.shape[0]
      #like [0, 0, 1, 1] [1, 0, 0, 1] ->...  choose (0,1), (0, 0), (1,0), (1, 1)
      values = score[np.repeat(np.arange(x), N), indices.ravel()].reshape(x, k)
      return values, indices

class RerankSimPredictor(object):
  def __init__(self, model_dir, exact_model_dir, num_rerank=100, 
              lkey=None, rkey=None, exact_lkey=None, exact_rkey=None, key='score', exact_key='score'):
    self._predictor = SimPredictor(model_dir, index=0, lkey=lkey, rkey=rkey, key=key)
    #TODO FIXME for safe use -1, should be 1 also ok, but not sure why dual_bow has two 'score'.. 
    #[<tf.Tensor 'dual_bow/main/dual_textsim_1/dot/MatMul:0' shape=(?, ?) dtype=float32>, <tf.Tensor 'dual_bow/main/dual_textsim_1/dot/MatMul:0' shape=(?, ?) dtype=float32>,
    # <tf.Tensor 'seq2seq/main/Exp_4:0' shape=(?, 1) dtype=float32>]
    #this is becasue you use evaluator(predictor + exact_predictor) when train seq2seq, so load dual_bow will add one score..
    self._exact_predictor = SimPredictor(exact_model_dir, index=-1, lkey=exact_lkey, rkey=exact_rkey, key=exact_key)

    self._num_rerank = num_rerank

  def inference(self, ltext, rtext, ratio=1.):
    scores = self._predictor.inference(ltext, rtext)
    if not ratio:
      return scores

    exact_scores = []
    for i, score in enumerate(scores):
      index= (-score).argsort()
      top_index = index[:self._num_rerank]
      exact_rtext = rtext[top_index]
      exact_score = self._exact_predictor.elementwise_predict([ltext[i]], rtext)
      exact_score = np.squeeze(exact_score)
      if ratio < 1.:
        for j in range(len(top_index)):
          exact_score[j] = ratio * exact_score[j] + (1. - ratio) * score[top_index[j]]

      exact_scores.append(exact_score)
    return np.array(exact_score)

  def predict(self, ltext, rtext, ratio=1.):
    return self.predict(ltext, rtext, ratio)

  #TODO do numpy has top_k ? seems argpartition will get topn but not in order
  def top_k(self, ltext, rtext, k=1, ratio=1., sorted=True):
    assert k <= self._num_rerank
    #TODO speed hurt?
    ltext = np.array(ltext)
    rtext = np.array(rtext)
    scores = self._predictor.predict(ltext, rtext)
    
    top_values = []
    top_indices = []

    if not ratio:
      for i, score in enumerate(scores):
        index = (-score).argsort()
        top_values.append(score[index[:k]])
        top_indices.append(index[:k])
      return np.array(top_values), np.array(top_indices)

    for i, score in enumerate(scores):
      index = (-score).argsort()
      print(index,  np.argpartition(-score, self._num_rerank))
      if ratio:
        top_index = index[:self._num_rerank]
        exact_rtext = rtext[top_index]
        exact_score = self._exact_predictor.elementwise_predict([ltext[i]], exact_rtext)
        exact_score = np.squeeze(exact_score)
        if ratio < 1.:
          for j in range(len(top_index)):
            exact_score[j] = ratio * exact_score[j] + (1. - ratio) * score[top_index[j]]

        exact_index = (-exact_score).argsort()

        new_index = [x for x in index]
        for j in range(len(exact_index)):
          new_index[j] = index[exact_index[j]]
        index = new_index
      
      top_values.append(exact_score[exact_index[:k]])
      top_indices.append(index[:k])

    return np.array(top_values), np.array(top_indices)
  