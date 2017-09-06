#!/usr/bin/env python
# ==============================================================================
#          \file   json2txt.py
#        \author   chenghuige  
#          \date   2017-09-06 14:25:04.959493
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os
import json 

m = json.load(open('./caption_train_annotations_20170902.json'))

out = open('./caption_train_annotations_20170902.txt', 'w')

for item in m:
  print(item['image_id'].encode('gbk'), '\x01'.join([x.encode('gbk').replace('\n', '').replace('\r', '') for x in item['caption']]), sep='\t', file=out)
