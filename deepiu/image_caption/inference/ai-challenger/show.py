#!/usr/bin/env python
# ==============================================================================
#          \file   inference.py
#        \author   chenghuige  
#          \date   2017-09-14 07:45:47.415075
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os
from deepiu.util.text_predictor import TextPredictor
from deepiu.util.sim_predictor import SimPredictor
from deepiu.util import ids2text, text2ids
import melt, gezi
import numpy as np 
import traceback
#FIXME ValueError: At least two variables have the same name: InceptionResnetV2/Repeat/block35_9/Conv2d_1x1/biases  
try:
  import conf
  from conf import TEXT_MAX_WORDS
except Exception:
  from deepiu.image_caption.conf import TEXT_MAX_WORDS

image_dir = '/home/gezi/data2/data/ai_challenger/image_caption/pic/'
image_file = '6275b5349168ac3fab6a493c509301d023cf39d3.jpg'

image_model_checkpoint_path = '/home/gezi/data/image_model_check_point/inception_resnet_v2_2016_08_30.ckpt'
model_dir = '/home/gezi/new/temp/image-caption/ai-challenger/model/showattentell/'
sim_model_dir = '/home/gezi/new/temp/image-caption/ai-challenger/model/bow/'
vocab_path = '/home/gezi/new/temp/image-caption/ai-challenger/tfrecord/seq-basic/vocab.txt'
valid_dir = '/home/gezi/new/temp/image-caption/ai-challenger/tfrecord/seq-basic/valid'

image_model_name = 'InceptionResnetV2'

feature_name = melt.image.get_features_name(image_model_name)

#if finetuned model, just  TextPredictor(model_dir, vocab_path)
if not melt.varname_in_checkpoint(image_model_name, model_dir):
  predictor = TextPredictor(model_dir, vocab_path, image_model_checkpoint_path, image_model_name=image_model_name, feature_name=feature_name)
else:
  predictor = TextPredictor(model_dir, vocab_path)
  
vocab = ids2text.vocab 

text2ids.init(vocab_path)

sim_predictor = SimPredictor(sim_model_dir, image_model_checkpoint_path, image_model_name=image_model_name, index=-1)

text_strs = np.load(os.path.join(valid_dir, 'distinct_text_strs.npy'))
img2text = np.load(os.path.join(valid_dir, 'img2text.npy')).item()

while True:
  image_file = raw_input('image_file like 6275b5349168ac3fab6a493c509301d023cf39d3.jpg:')
  if image_file == 'q':
    break

  image_path = os.path.join(image_dir, image_file)
  print('image_path:', image_path)

  if not os.path.exists(image_path):
    print('image path not find!')
    continue

  try:
    hits = img2text[image_file]
    texts = [text_strs[hit] for hit in hits]
    for text in texts:
      word_ids = text2ids.text2ids(text)
      seg_text = text2ids.ids2text(word_ids, print_end=False)
      print('label:', text, seg_text)
      words_importance = sim_predictor.words_importance([word_ids])
      words_importance = words_importance[0]
      print('word importance:')
      for i in range(len(word_ids)):
        if word_ids[i] == 0:
          break 
        print(vocab.key(int(word_ids[i])), words_importance[i], end='|')  
      print()
  except Exception:
    print(traceback.format_exc(), file=sys.stderr)    
    pass

  image = melt.read_image(image_path)
  word_ids, scores = predictor.word_ids([image])
  word_id = word_ids[0]
  score = scores[0]
  print('best predict:', ids2text.translate(word_id[0]),  score[0], '/'.join([vocab.key(int(id)) for id in word_id[0] if id != vocab.end_id()]))
  
  l = [id for id in word_id[0] if id != vocab.end_id()]
  l = gezi.pad(l, TEXT_MAX_WORDS)
  words_importance = sim_predictor.words_importance([l])
  words_importance = words_importance[0]

  print('word importance:')
  for i in range(len(word_id[0])):
    if word_id[0][i] == vocab.end_id():
      break
    print(vocab.key(int(word_id[0][i])), words_importance[i], end='|')
  print()

  for i in range(len(word_id)):
    if i > 0:
      print('top%d predict:'%i, ids2text.translate(word_id[i]),  score[i], '/'.join([vocab.key(int(id)) for id in word_id[i] if id != vocab.end_id()]))
      l = [id for id in word_id[i] if id != vocab.end_id()]
      l = gezi.pad(l, TEXT_MAX_WORDS)
      words_importance = sim_predictor.words_importance([l])
      words_importance = words_importance[0]

      print('word importance:')
      for j in range(len(word_id[i])):
        if word_id[i][j] == vocab.end_id():
          break
        print(vocab.key(int(word_id[i][j])), words_importance[j], end='|')
      print()

  scores, word_ids = sim_predictor.top_words([image], 50)
  scores = scores[0]
  word_ids = word_ids[0]
  print('topwords of image:')
  for word_id, score in zip(word_ids, scores):
    print(vocab.key(int(word_id)), score, end='|')
  print()





