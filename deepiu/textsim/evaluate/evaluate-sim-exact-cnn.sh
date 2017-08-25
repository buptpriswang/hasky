python ./evaluate-sim.py \
 --valid_resource_dir /home/gezi/new/temp/makeup/title2name/tfrecord/seq-basic/valid/ \
 --use_exact_predict 1 \
 --model_dir /home/gezi/new/temp/makeup/title2name/model/bow/ \
 --exact_model_dir /home/gezi/new/temp/makeup/title2name/model/cnn.elementwise/ \
 --lkey dual_bow/main/ltext:0 \
 --rkey dual_bow/main/rtext:0 \
 --exact_lkey dual_cnn2/main/ltext:0 \
 --exact_rkey dual_cnn2/main/rtext:0 
