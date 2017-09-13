#!/usr/bin/env python
# ==============================================================================
#          \file   merge-pic-feature.py
#        \author   chenghuige  
#          \date   2017-09-02 07:11:31.677005
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os, glob

m = {}
for line in open(sys.argv[1]): 
  pic, val = line.strip().split('\t')
  if pic not in m:
    m[pic] = set([val])
  else:
    m[pic].add(val)

for line in sys.stdin:
  pic, feature = line.strip().split('\t')
  if pic in m:
    print(pic, '\x01'.join(m[pic]), feature, sep='\t')
