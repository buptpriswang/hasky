#!/usr/bin/env python
# ==============================================================================
#          \file   gen-feature.py
#        \author   chenghuige  
#          \date   2017-08-05 16:40:24.575681
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os

img2id_file = sys.argv[1]
img2feature_file = sys.argv[2]
label_file = sys.argv[3]
text_index = int(sys.argv[4])

img2id = {}
for line in open(img2id_file):
  img, id = line.rstrip().split('\t')
  img2id[img] = id 
print('img2id size:', len(img2id), file=sys.stderr)

id2feature = {}
for line in open(img2feature_file):
  id, feature = line.rstrip().split('\t')
  id = id.replace('.jpg', '')
  id2feature[id] = feature
print('img2feature size:', len(id2feature), file=sys.stderr)

texts_set = set()
imgs_set = set()
num_pairs = 0
num_labels = 0
missing_img_set = set()
num_missing_img_times = 0

pair_set = set()

for line in open(label_file):
  l = line.rstrip().split('\t')
  try:
    text = l[text_index].strip('\"')
  except Exception:
    print('bad line:', line, file=sys.stderr)
    continue
  texts_set.add(text)
  imgs = l[-3:]
  is_top = True
  for img in imgs:
    img = img.strip('\"')
    if img in img2id:
      id = img2id[img]
      if id not in id2feature:
        print('%s not has feature'%img, file=sys.stderr)
        missing_img_set.add(id)
        num_missing_img_times += 1
      else:
        pair = '%s\t%s'%(text, img)
        if pair in pair_set:
          continue 
        pair_set.add(pair)
        feature = id2feature[id]
        imgs_set.add(img)
        num_pairs += 1
        if is_top:
          num_labels += 1
          is_top = False
        print(img, text, feature, sep='\t')

print('num missing img:', len(missing_img_set), file=sys.stderr)
print('num missing img times:', num_missing_img_times, file=sys.stderr)
print('num labels:', num_labels, file=sys.stderr)
print('num pairs:', num_pairs, file=sys.stderr)
print('num imgs:', len(imgs_set), file=sys.stderr)
print('num texts:', len(texts_set), file=sys.stderr)
  
