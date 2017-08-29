python ./evaluate-sim.py \
  --valid_resource_dir /home/gezi/new/temp/makeup/title2name/tfrecord/seq-basic/valid/ \
  --use_exact_predictor 1 \
  --model_dir /home/gezi/new/temp/makeup/title2name/model/bow/ \
  --exact_model_dir /home/gezi/new/temp/makeup/title2name/model/seq2seq/ \
  --exact_lkey seq2seq/model_init_1/input_text:0 \
  --exact_rkey seq2seq/model_init_1/text:0
