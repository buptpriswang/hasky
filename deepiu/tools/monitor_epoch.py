#!/usr/bin/env python
# ==============================================================================
#          \file   monitor_epoch.py
#        \author   chenghuige  
#          \date   2017-09-19 12:14:39.924016
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('model_dir', '/home/gezi/new/temp/image-caption/ai-challenger/model/showandtell', '')
flags.DEFINE_string('vocab', '/home/gezi/new/temp/image-caption/ai-challenger/tfrecord/seq-basic/vocab.txt', '')
flags.DEFINE_integer('start_epoch', 0, '')

import sys, os, glob, time
import pickle

import gezi, melt 
from deepiu.util import evaluator
from deepiu.util.text_predictor import TextPredictor

logging = melt.logging

def main(_):
	print('eval_rank:', FLAGS.eval_rank, 'eval_translation:', FLAGS.eval_translation)
	epoch_dir = os.path.join(FLAGS.model_dir, 'epoch')

	logging.set_logging_path(gezi.get_dir(epoch_dir))

	log_dir = epoch_dir
	sess = tf.Session()
	summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
	
	Predictor = TextPredictor  

	image_model = None 
	if FLAGS.image_checkpoint_file:
		#feature_name = None, since in show and tell predictor will use gen_features not gen_feature
		image_model = melt.image.ImageModel(FLAGS.image_checkpoint_file, 
                                        FLAGS.image_model_name, 
                                        feature_name=None)

	evaluator.init(image_model)

	visited_path = os.path.join(epoch_dir, 'visited.pkl')
	if not os.path.exists(visited_path):
		visited_checkpoints = set()
	else:
		visited_checkpoints = pickle.load(open(visited_path, 'rb'))
	while True:
		suffix = '.data-00000-of-00001'
		files = glob.glob(os.path.join(epoch_dir, 'model.ckpt*.data-00000-of-00001'))
		#from epoch 1, 2, ..
		files.sort(key=os.path.getmtime)
		files = [file.replace(suffix, '') for file in files]
		for file in files:
			if file not in visited_checkpoints:
				logging.info('mointor_epoch:%d'%(len(visited_checkpoints)))
				visited_checkpoints.add(file)
				#will use predict_text in eval_translation , predict in eval_rank
				predictor = Predictor(file, image_model=image_model, feature_name=melt.get_features_name(FLAGS.image_model_name)) 
				summary = tf.Summary()
				scores, metrics = evaluator.evaluate(predictor, eval_rank=FLAGS.eval_rank, eval_translation=FLAGS.eval_translation)
				melt.add_summarys(summary, scores, metrics)
				summary_writer.add_summary(summary, len(visited_checkpoints))
				summary_writer.flush()
				pickle.dump(visited_checkpoints, open(visited_path, 'wb'))
		time.sleep(5)

 
if __name__ == '__main__':
  tf.app.run()