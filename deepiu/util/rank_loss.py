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
flags.DEFINE_bool('dist_normalize', False, 
                  """for dist based loss like constrastive or triplet by default will not l2 norm, 
                     can force l2 norm setting dist_normalize
                     for cosine same based loss will always norm""")
flags.DEFINE_bool('no_dist_normalize', False, '')
flags.DEFINE_bool('concat_sim', False, '')


import melt

class LossType:
  hinge = 'hinge'
  point = 'point'   #only positive cosine
  pairwise_cross = 'pairwise_cross'
  pairwise_exp = 'pairwise_exp'
  contrastive = 'contrastive'
  contrastive_sqrt = 'contrastive_sqrt'
  triplet = 'triplet'
  cross_entropy = 'cross_entropy'

class DistType:
  cosine = 'cosine'
  euclidean = 'euclidean' #l2 dist

def get_dist_type(loss_type):
  #TODO notice cross_entorpy also as cosine same.. since tf.sigmoid() might face big value like 76 then to 1
  if loss_type == LossType.contrastive \
    or loss_type == LossType.contrastive_sqrt \
     or loss_type == LossType.triplet:
     return DistType.euclidean
  else:
    return DistType.cosine

def concat_sim_score(u, v):
  #feature = tf.concat([left_feature, right_feature], 1)
  #feature = tf.abs(left_feature - right_feature)
  #feature = tf.multiply(left_feature, right_feature)
  #must name layers for share! otherwise dense_1 dense_2 ...
  #return tf.sigmoid(tf.layers.dense(feature, 1, name='compute_sim_dense'))
  #return melt.element_wise_dot(u, v)
  feature = tf.concat((u, v, tf.abs(u-v), u*v), 1)
  score = melt.slim.mlp(feature,
                 [512, 512, 1],
                 activation_fn=tf.nn.tanh,
                 #activation_fn=None,
                 scope='concat_same_mlp')
  #return tf.sigmoid(score)
  return score

#[batch, dim] [batch_dim] -> [batch, 1]
def compute_sim(left_feature, right_feature):
  if FLAGS.concat_sim:
    return concat_sim_score(left_feature, right_feature)
  loss_type = FLAGS.loss
  if get_dist_type(loss_type) == DistType.euclidean:
    #l2 distance the smaller the better
    return tf.reduce_sum(tf.square(left_feature - right_feature), -1, keep_dims=True)
  else:
    #cosine same, the larger the beter -1 to 1
    return melt.element_wise_dot(left_feature, right_feature)

#TODO pytorch has pairwise_distance  move to melt.ops
#https://stackoverflow.com/questions/37009647/compute-pairwise-distance-in-a-batch-without-replicating-tensor-in-tensorflow
#https://www.reddit.com/r/tensorflow/comments/58tq6j/how_to_compute_pairwise_distance_between_points/
# qexpand = tf.expand_dims(q,1) # one olumn                                                                                                         
# qTexpand = tf.expand_dims(q,0) # one row                                                                                                           
# qtile = tf.tile(qexpand,[1,N,1])                                                                                                                   
# qTtile = tf.tile(qTexpand,[N,1,1])                                                                                                                 
# deltaQ = qtile - qTtile                                                                                                                            
# deltaQ2 = deltaQ*deltaQ                                                                                                                            
# d2Q = tf.reduce_sum(deltaQ2,2) 

# def pairwise_l2_norm2(x, y, scope=None):
#   with tf.op_scope([x, y], scope, 'pairwise_l2_norm2'):
#     size_x = tf.shape(x)[0]
#     size_y = tf.shape(y)[0]
#     xx = tf.expand_dims(x, -1)
#     xx = tf.tile(xx, tf.stack([1, 1, size_y]))

#     yy = tf.expand_dims(y, -1)
#     yy = tf.tile(yy, tf.stack([1, 1, size_x]))
#     yy = tf.transpose(yy, perm=[2, 1, 0])

#     diff = tf.subtract(xx, yy)
#     square_diff = tf.square(diff)

#     square_dist = tf.reduce_sum(square_diff, 1)

#     return square_dist


# def pairwise_distance(x, y):
#   qexpand = tf.expand_dims(x,1) # one olumn                                                                                                         
#   qTexpand = tf.expand_dims(y,0) # one row                                                                                                           
#   qtile = tf.tile(qexpand,[1,N,1])                                                                                                                   
#   qTtile = tf.tile(qTexpand,[N,1,1])                                                                                                                 
#   deltaQ = qtile - qTtile                                                                                                                            
#   deltaQ2 = deltaQ*deltaQ                                                                                                                            
#   d2Q = tf.reduce_sum(deltaQ2,2)   
#   return d2Q

#-------pairwise result of compute_sim
def dot(x, y):
  if FLAGS.concat_sim:
    return concat_sim_score(x, y)

  if FLAGS.dist_normalize or get_dist_type(FLAGS.loss) == DistType.cosine:
    #has already normalized
    return melt.dot(x, y)
  else:
    #TODO ... 
    #return -tf.sqrt(pairwise_distance(x, y))
    ##not normalized before but for contrastive will it be better to use pairwise_distance like in pytorch ?

    #distance_L1 = tf.reduce_sum(tf.abs(tf.subtract(vocab, tf.expand_dims(batch,1))), axis=2)

    return melt.cosine(x, y)

def normalize(feature):
  #TODO it is better t normalize for contstive or not ?
  
  #if FLAGS.concat_same:
  #  return feature

  if FLAGS.no_dist_normalize:
    return feature

  if FLAGS.dist_normalize or get_dist_type(FLAGS.loss) == DistType.cosine:
    return tf.nn.l2_normalize(feature, -1)
  else:
    return feature

def pairwise_loss(pos_score, neg_scores):
  if FLAGS.loss == LossType.hinge:
    loss = melt.losses.hinge(pos_score, neg_scores, FLAGS.margin)
  elif FLAGS.loss == LossType.cross_entropy:
    #will conver to 0-1 logits int melt.cross_entropy_loss http://www.wildml.com/2016/07/deep-learning-for-chatbots-2-retrieval-based-model-tensorflow/
    #TODO seems work but not as good as hinge, maybe should be EMS loss not norm to use cosine, then cross ?
    #[batch_size, 1 + num_negs]
    scores = tf.concat([pos_score, neg_scores], 1)
    loss = melt.losses.cross_entropy(scores)
  #point losss is bad here for you finetune both text and image embedding, all 0 vec will loss minimize..
  #if use point loss you need to fix text embedding
  elif FLAGS.loss == LossType.point: 
    #or might be (1 - pos_score) / 2 loss always 0 -1
    #loss = tf.reduce_mean(1. - pos_score)
    loss = tf.reduce_mean((1. - pos_score) / 2.)
  elif FLAGS.loss == LossType.triplet: 
    loss = melt.losses.pairwise_cross(pos_score, neg_scores)
  elif FLAGS.loss == LossType.pairwise_exp:
    loss = melt.losses.pairwise_exp(pos_score, neg_scores)
  elif FLAGS.loss == LossType.contrastive:
    loss = melt.losses.contrastive(pos_score, neg_scores, margin=FLAGS.margin)
  elif FLAGS.loss == LossType.contrastive_sqrt:
    loss = melt.losses.contrastive(pos_score, neg_scores, margin=FLAGS.margin, use_square=False)
  elif FLAGS.loss == LossType.triplet:
    loss = melt.losses.triplet(pos_score, neg_scores, FLAGS.margin)
  else:
    raise ValueError('Not supported loss: ', FLAGS.loss)

  return loss