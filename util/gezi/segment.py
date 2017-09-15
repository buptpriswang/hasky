#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   segment.py
#        \author   chenghuige  
#          \date   2016-08-25 19:09:51.756927
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gezi.nowarning
import gezi
from gezi.libgezi_util import get_single_cns

#TODO need improve
def segment_gbk_char(text, cn_only=False):
  l = []
  pre_is_cn = False
  unicode_text = text.decode('gbk', 'ignore')
  for word in unicode_text:
    if u'\u4e00' <= word <= u'\u9fff':
      pre_is_cn = True
      if l:
        l.append(' ')
    else:
      if pre_is_cn:
        l.append(' ')
        pre_is_cn = False
    if not cn_only or pre_is_cn:
      l.append(word)
  text = ''.join(l)
  text = text.encode('gbk')
  l = text.split()
  return [x.strip() for x in l if x.strip()]  

def segment_utf8_char(text, cn_only=False):
  l = []
  pre_is_cn = False
  unicode_text = text.decode('utf-8', 'ignore')
  for word in unicode_text:
    if u'\u4e00' <= word <= u'\u9fff':
      pre_is_cn = True
      if l:
        l.append(' ')
    else:
      if pre_is_cn:
        l.append(' ')
        pre_is_cn = False
    if not cn_only or pre_is_cn:
      l.append(word)
  text = ''.join(l)
  text = text.encode('utf-8')
  l = text.split()
  return [x.strip() for x in l if x.strip()]    

def segment_en(text):
  l = text.strip().split()
  return [x.strip() for x in l if x.strip()]

if gezi.encoding == 'gbk':
  import libsegment
  seg = libsegment.Segmentor

  segment_char = segment_gbk_char 

  class BaiduSegmentor(object):
    def __init__(self, data='./data/wordseg', conf='./conf/scw.conf'):
      seg.Init(data_dir=data, conf_path=conf)

    #@TODO add fixed pre dict like 1500w keyword ? 
    def segment_nodupe_noseq(self, text):
      results = set()
      for word in seg.Segment(text):
        results.add(word)
      for word in seg.Segment(text, libsegment.SEG_NEWWORD):
        results.add(word)
      for word in seg.Segment(text, libsegment.SEG_BASIC):
        results.add(word)
      for word in get_single_cns(text):
        results.add(word)
      return list(results)

    def Segment_nodupe_noseq(self, text):
      return self.segment_noseq(text)

    def segment_nodupe(self, text):
      results = [word for word in seg.Segment(text)]
      results += [wrod for word in seg.Segment(text, libsegment.SEG_NEWWORD)]
      results += [wrod for word in seg.Segment(text, libsegment.SEG_BASIC)]
      results += [word for word in get_single_cns(text)]
      return gezi.dedupe_list(results)

    def Segment_nodupe(self, text):
      return self.segment(text)

    #@TODO is this best method ?
    def segment(self, text):
      results = [word for word in get_single_cns(text)]
      results_set = set(results) 
      
      for word in seg.Segment(text):
        if word not in results_set:
          results.append(word)
          results_set.add(word)
      
      for word in seg.Segment(text, libsegment.SEG_NEWWORD):
        if word not in results_set:
          results.append(word)
          results_set.add(word)
      
      for word in seg.Segment(text, libsegment.SEG_BASIC):
        if word.isdigit():
          word = '<NUM>'
        if word not in results_set:
          results.append(word)
          results_set.add(word)

      return results

    def segment_seq_all(self, text):
      results = [word for word in get_single_cns(text)]

      results.append('<SEP0>')
      for word in seg.Segment(text, libsegment.SEG_BASIC):
        results.append(word)
       
      results.append('<SEP1>')
      for word in seg.Segment(text):
        results.append(word)
      
      results.append('<SEP2>')
      for word in seg.Segment(text, libsegment.SEG_NEWWORD):
        results.append(word)
      
      return results
    
    def segment_phrase(self, text):
      return seg.Segment(text)

    def segment_basic(self, text):
      return seg.Segment(text, libsegment.SEG_BASIC)

    def segment_phrase_single(self, text):
      results = [word for word in get_single_cns(text)]
      results += [word for word in seg.Segment(text)]
      return results

    def segment_basic_single(self, text):
      results = [word for word in get_single_cns(text)]
      results += [word for word in seg.Segment(text, libsegment.SEG_BASIC)]
      return results

    def segment_merge_newword_single(self, text):
      results = [word for word in get_single_cns(text)]
      results += [word for word in seg.Segment(text, libsegment.SEG_MERGE_NEWWORD)]
      return results
    
    def Segment(self, text, method='default'):
      """
      default means all level combine
      """
      if method == 'default' or method == 'all' or method == 'full':
        return self.segment(text)
      elif method == 'phrase_single':
        return self.segment_phrase_single(text)
      elif method == 'phrase':
        return seg.Segment(text)
      elif method == 'basic':
        return seg.Segment(text, libsegment.SEG_BASIC)
      elif method == 'basic_single':
        return self.segment_basic_single(text)
      elif method == 'merge_newword':
        return seg.Segment(text, libsegment.SEG_MERGE_NEWWORD)
      elif method == 'merge_newword_single':
        return self.segment_merge_newword_single(text)
      elif method == 'seq_all':
        return self.segment_seq_all(text)
      elif method == 'en':
        return segment_en(text)
      elif method == 'char':
        return segment_char(text)
      elif method == 'tab':
        return text.strip().split('\t')
      elif method == 'white_space':
        return text.strip().split()
      else:
        raise ValueError('%s not supported'%method)


  Segmentor = BaiduSegmentor
else: #by default utf8
  import jieba 
  segment_char = segment_utf8_char

  #TODO hack how to better deal? now for c++ part must be str..
  #TODO py3
  class JiebaSegmentor(object):
    def __init__(self):
      pass

    def segment_basic_single(self, text):
      #results = [word for word in get_single_cns(text)]
      results = [word for word in segment_char(text, cn_only=True)]
      results += [word for word in jieba.cut(text)]
      return results  

    def segment_full_single(self, text):
      #results = [word for word in get_single_cns(text)]
      results = [word for word in segment_char(text, cn_only=True)]
      results += [word for word in jieba.cut_for_search(text)]
      return results  

    def Segment(self, text, method='basic'):
      """
      default means all level combine
      """
      words = None
      if method == 'default' or method == 'basic' or method == 'exact':
        words = jieba.cut(text, cut_all=False)
      elif method == 'basic_single' or method == 'exact_single':
        words = self.segment_basic_single(text)
      elif method == 'search':
        words = jieba.cut_for_search(text)
      elif method == 'cut_all':
        words = jieba.cut(text, cut_all=True)
      elif method == 'all' or method == 'full':
        words = self.segment_full_single(text)
      elif method == 'en':
        words = segment_en(text)
      elif method == 'char':
        words = segment_char(text)
      elif method == 'tab':
        words = text.strip().split('\t')
      elif method == 'white_space':
        words = text.strip().split()
      else:
        raise ValueError('%s not supported'%method)

      words = [w for w in words]
      for i in range(len(words)):
        if isinstance(words[i], unicode):
          words[i] = words[i].encode('utf-8')
          
      return words

  Segmentor = JiebaSegmentor

import threading
#looks like by this way... threads create delete too much cost @TODO
#and how to prevent this log?
#DEBUG: 08-27 13:18:50:   * 0 [seg_init_by_model]: set use ne=0
#TRACE: 08-27 13:18:50:   * 0 [clear]: tag init stat error
#DEBUG: 08-27 13:18:50:   * 0 [init_by_model]: max_nbest=1, max_word_num=100, max_word_len=100, max_y_size=6
#2>/dev/null 2>&1
#so multi thread only good for create 12 threads each do many works at parrallel, here do so little work.. slow!
def segments(texts, segmentor):
  results = [None] * len(texts)
  def segment_(i, text):
    seg.Init()
    results[i] = segmentor.segment(text)
  threads = []
  for args in enumerate(texts):
    t = threading.Thread(target=segment_, args=args) 
    threads.append(t) 
  for t in threads:
    t.start()
  for t in threads:
    t.join()
  return results

#seems multiprocessing same as above ?
#segmentor resource only init once! whyï¼? @TODO
import multiprocessing
from multiprocessing import Manager 
def segments_multiprocess(texts, segmentor):
  manager = Manager()
  dict_ = manager.dict()
  results = [None] * len(texts)

  def segment_(i, text):
    seg.Init()
    dict_[i] = segmentor.segment(text)
  record = []
  for args in enumerate(texts):
    process =  multiprocessing.Process(target=segment_, args=args) 
    process.start()
    record.append(process) 
  for process in record:
    process.join()
  for i in xrange(len(record)):
    results[i] = dict_[i]
  return results
