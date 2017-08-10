#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   eval_show.py
#        \author   chenghuige  
#          \date   2016-08-27 17:05:57.683113
#   \Description  
# ==============================================================================

"""
 this is rigt now proper for bag of word model, show eval result
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('num_text_topn', 5, '')
flags.DEFINE_integer('num_word_topn', 50, '')

import functools
import melt

from deepiu.util import evaluator
from deepiu.util import text2ids

