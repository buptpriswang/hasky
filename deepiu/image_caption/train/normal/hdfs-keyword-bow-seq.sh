cp ./prepare/seq-with-unk/keyword/conf.py conf.py
source ./prepare/seq-with-unk/keyword/config 

dir=/home/gezi/temp/image-caption/ 
model_dir=$dir/model.keyword.bow.seq
mkdir -p $model_dir

python ./train.py \
  --train_input=$hdfs_train_output_path \
  --valid_input=$valid_output_path/'test_*' \
  --fixed_valid_input=$fixed_valid_output_path/'test' \
  --valid_resource_dir=$valid_output_path \
  --vocab=$train_output_path/vocab.bin \
  --num_records_file=$train_output_path/num_records.hdfs.txt \
  --image_url_prefix='D:\data\image-text-sim\evaluate\imgs\' \
  --label_file=$valid_output_path/'image_labels.npy' \
  --image_feature_file=$valid_data_path/'test' \
  --image_name_bin=$valid_output_path/'image_names.npy' \
  --image_feature_bin=$valid_output_path/'image_features.npy' \
  --img2text=$valid_output_path/'img2text.npy' \
  --text2id=$valid_output_path/'text2id.npy' \
  --fixed_eval_batch_size 10 \
  --num_fixed_evaluate_examples 10 \
  --num_evaluate_examples 10 \
  --show_eval 0 \
  --train_only 1 \
  --metric_eval 1 \
  --monitor_level 2 \
  --no_log 0 \
  --batch_size 128 \
  --min_after_dequeue 1000 \
  --eval_interval_steps 1000 \
  --metric_eval_interval_steps 10000 \
  --save_interval_seconds 7200 \
  --save_interval_steps 5000 \
  --num_metric_eval_examples 1000 \
  --metric_eval_batch_size 500 \
  --debug 0 \
  --num_negs 1 \
  --interval 100 \
  --eval_batch_size 100 \
  --feed_dict 0 \
  --margin 0.5 \
  --algo bow \
  --combiner=sum \
  --exclude_zero_index 1 \
  --dynamic_batch_length 1 \
  --emb_dim 256 \
  --hidden_size 1024 \
  --model_dir $model_dir \
  --num_records 9996 \
  --min_records 9996 \
  --log_device 0 \
  --work_mode full \
 
