#!/usr/bin/env python
#encoding=utf8
# ==============================================================================
#          \file   normalize.py
#        \author   chenghuige  
#          \date   2017-09-20 09:04:14.598669
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os

def norm(text):
  return text.lower().replace('。','') 

