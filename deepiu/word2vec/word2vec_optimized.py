#!/usr/bin/env python
# -*- coding: gbk -*-
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

"""Multi-threaded word2vec unbatched skip-gram model.

Trains the model described in:
(Mikolov, et. al.) Efficient Estimation of Word Representations in Vector Space
ICLR 2013.
http://arxiv.org/abs/1301.3781
This model does true SGD (i.e. no minibatching). To do this efficiently, custom
ops are used to sequentially process data within a 'batch'.

The key ops used are:
* skipgram custom op that does input processing.
* neg_train custom op that efficiently calculates and applies the gradient using
  true SGD.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import threading
import time

from six.moves import xrange  # pylint: disable=redefined-builtin

import numpy as np
import tensorflow as tf

import gezi.nowarning
from libword_counter import Vocabulary

word2vec = tf.load_op_library(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'word2vec_ops.so'))

flags = tf.app.flags

flags.DEFINE_string("save_path", None, "Directory to write the model.")
flags.DEFINE_string(
    "train_data", None,
    "Training data. must be all ids..")
flags.DEFINE_string(
    "eval_data", None,
    "must be all ids")
flags.DEFINE_string(
    "vocab_path", None, "vocab file")
flags.DEFINE_integer("embedding_size", 256, "The embedding dimension size.")
flags.DEFINE_integer(
    "epochs_to_train", 15,
    "Number of epochs to train. Each epoch processes the training data once "
    "completely.")
flags.DEFINE_float("learning_rate", 0.025, "Initial learning rate.")
flags.DEFINE_integer("num_neg_samples", 25,
                     "Negative samples per training example.")
flags.DEFINE_integer("batch_size", 500,
                     "Numbers of training examples each step processes "
                     "(no minibatching).")
flags.DEFINE_integer("concurrent_steps", 12,
                     "The number of concurrent training steps.")
flags.DEFINE_integer("window_size", 5,
                     "The number of words to predict to the left and right "
                     "of the target word.")
flags.DEFINE_integer("min_count", 5,
                     "The minimum number of word occurrences for it to be "
                     "included in the vocabulary.")
flags.DEFINE_float("subsample", 1e-3,
                   "Subsample threshold for word occurrence. Words that appear "
                   "with higher frequency will be randomly down-sampled. Set "
                   "to 0 to disable.")
flags.DEFINE_boolean(
    "interactive", False,
    "If true, enters an IPython interactive session to play with the trained "
    "model. E.g., try model.analogy(b'france', b'paris', b'russia') and "
    "model.nearby([b'proton', b'elephant', b'maxwell'])")

FLAGS = flags.FLAGS


class Options(object):
  """Options used by our word2vec model."""

  def __init__(self):
    # Model options.

    # Embedding dimension.
    self.emb_dim = FLAGS.embedding_size

    # Training options.

    # The training text file.
    self.train_data = FLAGS.train_data

    self.vocab_path = FLAGS.vocab_path

    assert (self.train_data or FLAGS.interactive) and self.vocab_path

    # Number of negative samples per example.
    self.num_samples = FLAGS.num_neg_samples

    # The initial learning rate.
    self.learning_rate = FLAGS.learning_rate

    # Number of epochs to train. After these many epochs, the learning
    # rate decays linearly to zero and the training stops.
    self.epochs_to_train = FLAGS.epochs_to_train

    # Concurrent training steps.
    self.concurrent_steps = FLAGS.concurrent_steps

    # Number of examples for one training step.
    self.batch_size = FLAGS.batch_size

    # The number of words to predict to the left and right of the target word.
    self.window_size = FLAGS.window_size

    # The minimum number of word occurrences for it to be included in the
    # vocabulary.
    self.min_count = FLAGS.min_count

    # Subsampling threshold for word occurrence.
    self.subsample = FLAGS.subsample

    # Where to write out summaries.
    self.save_path = FLAGS.save_path
    if not os.path.exists(self.save_path):
      os.makedirs(self.save_path)

    # Eval options.

    # The text file for eval.
    self.eval_data = FLAGS.eval_data



class Word2Vec(object):
  """Word2Vec model (Skipgram)."""

  def __init__(self, options, session):
    self._options = options
    self._session = session
    self.build_graph()
    self.build_eval_graph()

  def build_graph(self):
    """Build the model graph."""
    opts = self._options

    self.vocab = Vocabulary(opts.vocab_path, 1) #num resevered ids is 1, <PAD> index 0
    opts.vocab_size = self.vocab.size()
    opts.vocab_counts = [int(self.vocab.freq(i)) for i in xrange(self.vocab.size())]
    print("Data file: ", opts.train_data)
    print("Vocab size: ", self.vocab.size())

    # The training data. A text file.
    (words_per_epoch, current_epoch, total_words_processed,
     examples, labels) = word2vec.skipgram_word2vec(filename=opts.train_data,
                                                    vocab_count=opts.vocab_counts,
                                                    batch_size=opts.batch_size,
                                                    window_size=opts.window_size,
                                                    min_count=opts.min_count,
                                                    subsample=opts.subsample)
    opts.words_per_epoch = self._session.run(words_per_epoch)

    print("Words per epoch: ", opts.words_per_epoch)

    # Declare all variables we need.
    # Input words embedding: [vocab_size, emb_dim]
    w_in = tf.Variable(
        tf.random_uniform(
            [opts.vocab_size,
             opts.emb_dim], -0.5 / opts.emb_dim, 0.5 / opts.emb_dim),
        name="w_in")

    tf.add_to_collection('word_embedding', w_in)

    # Global step: scalar, i.e., shape [].
    w_out = tf.Variable(tf.zeros([opts.vocab_size, opts.emb_dim]), name="w_out")

    # Global step: []
    global_step = tf.Variable(0, name="global_step")

    # Linear learning rate decay.
    words_to_train = float(opts.words_per_epoch * opts.epochs_to_train)
    lr = opts.learning_rate * tf.maximum(
        0.0001,
        1.0 - tf.cast(total_words_processed, tf.float32) / words_to_train)

    # Training nodes.
    inc = global_step.assign_add(1)
    with tf.control_dependencies([inc]):
      train = word2vec.neg_train_word2vec(w_in,
                                          w_out,
                                          examples,
                                          labels,
                                          lr,
                                          vocab_count=opts.vocab_counts,
                                          num_negative_samples=opts.num_samples)

    self._w_in = w_in
    self._examples = examples
    self._labels = labels
    self._lr = lr
    self._train = train
    self.global_step = global_step
    self._epoch = current_epoch
    self._words = total_words_processed


  def build_eval_graph(self):
    """Build the evaluation graph."""
    # Eval graph
    opts = self._options

    # Normalized word embeddings of shape [vocab_size, emb_dim].
    nemb = tf.nn.l2_normalize(self._w_in, 1)

    # Nodes for computing neighbors for a given word according to
    # their cosine distance.
    nearby_word = tf.placeholder(dtype=tf.int32)  # word id
    nearby_emb = tf.gather(self._w_in, nearby_word)
    nearby_emb = tf.reduce_sum(nearby_emb, 0, keep_dims=True)
    nearby_emb = tf.nn.l2_normalize(nearby_emb, 1)
    nearby_dist = tf.matmul(nearby_emb, nemb, transpose_b=True)
    tf.add_to_collection('nearby_dist', nearby_dist)
    nearby_val, nearby_idx = tf.nn.top_k(nearby_dist,
                                         min(1000, opts.vocab_size))
    tf.add_to_collection('nearby_val', nearby_dist)
    tf.add_to_collection('nearby_idx', nearby_dist)

    self._nearby_word = nearby_word
    self._nearby_val = nearby_val
    self._nearby_idx = nearby_idx

    # Properly initialize all variables.
    tf.global_variables_initializer().run()

    self.saver = tf.train.Saver()

  def _train_thread_body(self):
    initial_epoch, = self._session.run([self._epoch])
    while True:
      _, epoch = self._session.run([self._train, self._epoch])
      if epoch != initial_epoch:
        break

  def train(self):
    """Train the model."""
    opts = self._options

    initial_epoch, initial_words = self._session.run([self._epoch, self._words])

    workers = []
    for _ in xrange(opts.concurrent_steps):
      t = threading.Thread(target=self._train_thread_body)
      t.start()
      workers.append(t)

    last_words, last_time = initial_words, time.time()
    while True:
      time.sleep(5)  # Reports our progress once a while.
      (epoch, step, words, lr) = self._session.run(
          [self._epoch, self.global_step, self._words, self._lr])
      now = time.time()
      last_words, last_time, rate = words, now, (words - last_words) / (
          now - last_time)
      print("Epoch %4d Step %8d: lr = %5.3f words/sec = %8.0f\r" % (epoch, step,
                                                                    lr, rate),
            end="")
      sys.stdout.flush()
      if epoch != initial_epoch:
        break

    for t in workers:
      t.join()

  def eval(self):
    self.nearby('nike')
    self.nearby('墨镜')
    self.nearby('高 铁')
    self.nearby('我 的 家乡 惠州 越来 越 热 ， 找 一款 喜欢 的 墨镜 很 重要')

  def nearby(self, words, num=50):
    """Prints out nearby words given a list of words."""
    print(words)
    words = words.split()
    ids = np.array([self.vocab.id(x) for x in words])
    vals, idx = self._session.run(
        [self._nearby_val, self._nearby_idx], {self._nearby_word: ids})
    i = 0
    for (neighbor, distance) in zip(idx[i, :num], vals[i, :num]):
      print("%s %f" % (self.vocab.key(int(neighbor)), distance), end=' ')
    print('')

  def dump_embedding(self, ofile):
    embedding = self._session.run(self._w_in)
    np.save(ofile, embedding)

def _start_shell(local_ns=None):
  # An interactive shell is useful for debugging/development.
  import IPython
  user_ns = {}
  if local_ns:
    user_ns.update(local_ns)
  user_ns.update(globals())
  IPython.start_ipython(argv=[], user_ns=user_ns)


def main(_):
  """Train a word2vec model."""
  #if not FLAGS.train_data or not FLAGS.eval_data or not FLAGS.save_path:
  #  print("--train_data --eval_data and --save_path must be specified.")
  #  sys.exit(1)
  opts = Options()
  with tf.Graph().as_default(), tf.Session() as session:
    with tf.device("/cpu:0"):
      model = Word2Vec(opts, session)

    if FLAGS.interactive:
      print('load model from file %s %s', opts.save_path, os.path.join(opts.save_path, "model.ckpt"))
      #TODO........ why fail...
      #model.saver.restore(session, os.path.join(opts.save_path, "model.ckpt"))
      #model.saver.restore(session, '/home/gezi/new/temp/image-caption/lijiaoshou/tfrecord/seq-basic/word2vec/model.ckpt')
      model.saver.restore(session, '/home/gezi/new/temp/image-caption/lijiaoshou/tfrecord/seq-basic/word2vec2/model.ckpt-15')
      while True:
        print('input your word  like iphone')
        word = sys.stdin.readline().strip()
        print(model.nearby(word))
    else:
      for epoch in xrange(opts.epochs_to_train):
        model.train()  # Process one epoch
        model.eval()  
        model.saver.save(session, os.path.join(opts.save_path, "model.ckpt-%d"%(epoch +1)))
                         #global_step=model.global_step)
      model.dump_embedding(os.path.join(opts.save_path, 'word_embedding.npy'))



if __name__ == "__main__":
  tf.app.run()
