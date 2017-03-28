#!/usr/bin/env python
# ==============================================================================
#          \file   test.py
#        \author   chenghuige  
#          \date   2017-03-28 20:21:26.220192
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os

import yaml  
  
f = open('test.yaml')  
  
x = yaml.load(f)  
  
print(x)

print('age', x['age'])
