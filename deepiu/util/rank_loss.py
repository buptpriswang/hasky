#!/usr/bin/env python
# ==============================================================================
#          \file   rank_loss.py
#        \author   chenghuige  
#          \date   2017-08-10 22:30:27.764386
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_float('margin', 0.5, 'margin for rankloss when rank_loss is hinge loss')
flags.DEFINE_string('loss', 'hinge', 'use hinge(hinge_loss) or cross(cross_entropy_loss) or hinge_cross(subtract then cross)')

import melt

def pairwise_loss(pos_score, neg_scores):
  if FLAGS.loss == 'hinge':
    loss = melt.losses.hinge(pos_score, neg_scores, FLAGS.margin)
  elif FLAGS.loss == 'cross':
    #will conver to 0-1 logits int melt.cross_entropy_loss http://www.wildml.com/2016/07/deep-learning-for-chatbots-2-retrieval-based-model-tensorflow/
    #TODO seems work but not as good as hinge, maybe should be EMS loss not norm to use cosine, then cross ?
    #[batch_size, 1 + num_negs]
    scores = tf.concat([pos_score, neg_scores], 1)
    loss = melt.losses.cross_entropy(scores, num_negs)
  #point losss is bad here for you finetune both text and image embedding, all 0 vec will loss minimize..
  #if use point loss you need to fix text embedding
  elif FLAGS.loss == 'point': 
    loss = tf.reduce_mean(1.0 - pos_score)
  elif FLAGS.loss == 'pairwise_cross': 
    loss = melt.losses.pairwise_cross(pos_score, neg_scores)
  elif FLAGS.loss == 'pairwise_exp':
    loss = melt.losses.pairwise_exp(pos_score, neg_scores)
  else:
    raise ValueError('Not supported loss')

  return loss