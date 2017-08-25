python ./evaluate-sim.py \
  --valid_resource_dir /home/gezi/new/temp/makeup/title2name/tfrecord/seq-basic/valid/ \
  --use_exact_predictor 1 \
  --exact_model_dir /home/gezi/new/temp/makeup/title2name/model/decomposable-nli/ \
  --exact_lkey decomposable_nli/main/ltext:0 \
  --exact_rkey decomposable_nli/main/rtext:0
