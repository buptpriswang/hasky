conf_path=./prepare/keyword2/app-conf/seq-basic.10w/
cp $conf_path/conf.py .
source $conf_path/config 

dir=/home/gezi/new/temp/image-caption/keyword/model
model_dir=$dir/showandtell
mkdir -p $model_dir

python ./train.py \
	--train_input $train_output_path/'train-*' \
	--valid_input $valid_output_path/'test-*' \
	--fixed_valid_input $fixed_valid_output_path/'test-*' \
	--valid_resource_dir $valid_output_path \
	--vocab $train_output_path/vocab.bin \
  --num_records_file  $train_output_path/num_records.txt \
  --model_dir=$model_dir \
  --algo show_and_tell \
  --num_sampled 256 \
  --log_uniform_sample 1 \
  --fixed_eval_batch_size 10 \
  --num_fixed_evaluate_examples 10 \
  --num_evaluate_examples 10 \
  --show_eval 1 \
  --train_only 0 \
  --metric_eval 1 \
  --monitor_level 2 \
  --no_log 0 \
  --batch_size 512 \
  --num_gpus 0 \
  --min_after_dequeue 500 \
  --learning_rate 0.1 \
  --eval_interval_steps 500 \
  --metric_eval_interval_steps 1000 \
  --save_interval_steps 1000 \
  --num_metric_eval_examples 1000 \
  --metric_eval_batch_size 500 \
  --margin 0.5 \
  --num_negs 0 \
  --use_neg 0 \
  --feed_dict 0 \
  --num_records 0 \
  --min_records 0 \
  --seg_method $seg_method \
  --feed_single $feed_single \
  --seq_decode_method 0 \
  --dynamic_batch_length 1 \
  --image_url_prefix 'D:\data\image-text-sim\evaluate\imgs\' \
  --image_feature_name idl4w_feature \
  --label_file $valid_output_path/'image_label.npy' \
  --img2text $valid_output_path/'img2text.npy' \
  --text2id $valid_output_path/'text2id.npy' \ 
  --image_name_bin $valid_output_path/'image_names.npy' \
  --image_feature_bin $valid_output_path/'idl4w.npy' \
  --log_device 0 \
  --work_mode full \

  #--model_dir /home/gezi/data/image-text-sim/model/model.ckpt-387000 \
  #2> ./stderr.txt 1> ./stdout.txt
