#!/usr/bin/env python
# ==============================================================================
#          \file   imgstext2imgs.py
#        \author   chenghuige  
#          \date   2017-09-02 21:23:31.974502
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os
import urllib

from melt import ImageDecoder

decoder = ImageDecoder()

odir = sys.argv[1]

for line in sys.stdin:
  l = line.rstrip().split('\t')
  pic = l[0]
  if not pic.endswith('.jpg'):
    pic = pic + '.jpg'
  pic = pic.replace('/', '_')
  img_text = l[-1]
  encoded_img = urllib.unquote_plus(img_text)
  image = decoder.decode(encoded_img)
  if image is None:
    continue
  ofile = os.path.join(odir, pic)
  out = open(ofile, 'wb')
  out.write(encoded_img)

print(decoder.num_imgs, decoder.num_bad_imgs, decoder.num_bad_imgs / decoder.num_imgs, file=sys.stderr)

