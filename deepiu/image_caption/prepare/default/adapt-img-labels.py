#!/usr/bin/env python
# ==============================================================================
#          \file   adjust-img-labels.py
#        \author   chenghuige  
#          \date   2017-04-13 08:02:49.600453
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os

for line in sys.stdin:
  l = line.rstrip('\n').split('\t')
  print(l[1], '\t'.join(l[1006].split('$*$')), sep='\t')
