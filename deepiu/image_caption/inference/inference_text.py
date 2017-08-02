#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   inference.py
#        \author   chenghuige  
#          \date   2016-10-06 19:48:19.923359
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS
 
flags.DEFINE_string('vocab', '/home/gezi/new/temp/image-caption/lijiaoshou/tfrecord/seq-basic/vocab.txt', 'vocabulary file')
flags.DEFINE_string('model_dir', '/home/gezi/new/temp/image-caption/lijiaoshou/model/rnn.max.gru.bi/', '')
flags.DEFINE_string('seg_method_', 'basic', '')

import gezi
import melt 
from deepiu.util import text2ids

import numpy as np

text2ids.init(FLAGS.vocab)

predictor = melt.Predictor(FLAGS.model_dir)

def predict(text):
  timer = gezi.Timer()
  text_ids = text2ids.text2ids(text, FLAGS.seg_method_, feed_single=True)
  print('text_ids', text_ids)

  #seq_len = 50	

  #print('words', words)
  argmax_encode = predictor.inference(['text_importance'], 
                                    feed_dict= {
                                      'rnn/main/text:0': [text_ids]
                                      })
  print('argmax_encode', argmax_encode[0])

  argmax_encode = argmax_encode[0][0]

  text_ids =  text2ids.text2ids(text, FLAGS.seg_method_, feed_single=True, append_start=True, append_end=True)
  words = text2ids.ids2words(text_ids)

  seq_len = 0
  for x in words:
  	if x != 0:
  		seq_len += 1
  	else:
  		break

  print(text_ids)

   # visualize model
  import matplotlib.pyplot as plt
  argmaxs = [np.sum((argmax_encode==k)) for k in range(seq_len)]
  print('argmaxs', argmaxs, np.sum(argmaxs), seq_len)
  x = range(len(argmax_encode))
  y = [100.0*n/np.sum(argmaxs) for n in argmaxs]
  #print(words, y)
  print(text)
  for word, score in zip(words, y):
    print(word, score)
  #fig = plt.figure()
  #plt.xticks(x, words, rotation=45)
  #plt.bar(x, y)
  #plt.ylabel('%')
  #plt.title('Visualisation of words importance')
  #plt.show()
  
predict('��Ů')
predict('˧���ī��')
predict('˧�� �� ī��')
predict('���쵽�ɶ���Ҫס�Ƶ���,��Щ���޴������Ա�������')
predict('��ͷ���ֻ�?�����ͷѧӢ��')
predict('˧�磡���������������ﲢû����ô�󣡼�΢����3�걣����������һ��������')
predict('�������һ��һӢ��������,�ú��ӴӴ˰���Ӣ��!288Ԫ�����������ȡ')
predict('�򹤻���ѧ��ɶ?ѧ��ʦ���ϰ�,�ߵ��Ķ����£�')
predict('ˮ�ȼ���,�����ʦ,�ܲ����ִ�ҵ!')
predict('ˮ�ȼ���,�����ʦ,�ܲ����ִ�ҵ')
predict('���к��ӵ���Ѿ�Ʒ�Σ����ֻ����ڼ��ϣ�ʡ��')
predict('������ֵ�߲�����������Щ�ֻ��������ʵ��')
predict('�����ÿ����ڸ����㣡�����֤��������10-30�����')
predict('1890Ԫ�����ɾɽ�ɽ?����׬��,��һֱ��������!')
predict('������ϰ����ޱ��Ƶ���ȷ�򿪷�ʽ��')
