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

m = json.load(open(sys.argv[1]))

out = open(sys.argv[2], 'w')

num_captions = 0
for item in m:
  captions = [x.encode('utf-8').replace('\n', '').replace('\r', '') for x in item['caption']]
  captions = [x for x in captions if x]
  #assert len(captions) == 5, item['image_id']
  if len(captions) != 5:
    print(item['image_id'], item['caption'], file=sys.stderr)
  num_captions += len(captions)
  print(item['image_id'].encode('utf-8'), '\x01'.join(captions), sep='\t', file=out)
print('num_captions:', num_captions, file=sys.stderr)
