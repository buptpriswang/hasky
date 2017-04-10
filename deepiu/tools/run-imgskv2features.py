#!/usr/bin/env python
# ==============================================================================
#          \file   run-imgskv2features.py
#        \author   chenghuige  
#          \date   2017-04-10 21:42:46.731884
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS
  
import sys, os

indir = sys.argv[1]
outdir = sys.argv[2]

os.system('mkdir -p {}'.format(outdir))
for file in os.listdir(indir):
  infile = indir + '/' + file 
  outfile = outdir + '/' + file 

  command = 'sh ./imgskv2features.sh {} {}'.format(infile, outfile)
  print(command, file=sys.stderr)
  os.system(command)