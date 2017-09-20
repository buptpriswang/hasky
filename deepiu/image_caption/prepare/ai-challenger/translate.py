#!/usr/bin/env python
# ==============================================================================
#          \file   translate.py
#        \author   chenghuige  
#          \date   2017-09-19 16:42:27.866685
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os
import requests, json, md5
import urllib
import random, time
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

train_file = '/home/gezi/new2/data/MSCOCO/train2014.txt'
valid_file = '/home/gezi/new2/data/MSCOCO/val2014.txt'

cn_train_file = '/home/gezi/new2/data/MSCOCO/cn_train2014.txt'
cn_valid_file = '/home/gezi/new2/data/MSCOCO/cn_val2014.txt'

out_train = open(cn_train_file, 'a')
out_valid =  open(cn_valid_file, 'a')

appid = '20170920000084005'
key = 'VURTIxWSU4LsGSkX3qlA'

pics = set()
for line in open(cn_valid_file, 'r'):
  pic = line.strip().split('\t')[0]
  pics.add(pic)

num_lines = 0
for line in open(valid_file):
  line = line.strip()
  pic, caption = line.split('\t')
  if pic in pics:
    continue
  captions = caption.split('\x01')
  cn_captions = []
  for caption in captions:
    salt = random.randint(32768, 65536)
    sign = appid + caption + str(salt) + key
    m1 = md5.new()
    m1.update(sign)
    sign = m1.hexdigest()
    query = 'http://api.fanyi.baidu.com/api/trans/vip/translate?q=%s&from=en&to=zh&appid=%s&salt=%d&sign=%s'%(urllib.quote(caption), appid, salt, sign)
    try:
      result = requests.post(query)
    except Exception:
      time.sleep(100)
      result = requests.post(query)
    result = json.loads(result.text)
    result = result['trans_result']
    for item in result:
      cn_captions.append(item['dst'])
  print(pic, '\x01'.join(cn_captions), sep='\t', file=out_valid) 
  
  if num_lines % 100 == 0:
    print(num_lines, '\x01'.join(cn_captions), file=sys.stderr)
  num_lines += 1
