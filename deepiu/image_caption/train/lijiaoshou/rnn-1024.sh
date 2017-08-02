conf_path=./prepare/default/app-conf/lijiaoshou/seq-basic
cp $conf_path/conf.py .
source $conf_path/config 

model_dir=/home/gezi/new/temp/image-caption/lijiaoshou/model/rnn.1024
mkdir -p $model_dir

python ./train.py  --encode_start_mark=1 --encode_end_mark=1 \
	--train_input=$train_output_path/'train-*' \
	--valid_input=$valid_output_path/'test-*' \
	--fixed_valid_input=$fixed_valid_output_path/'test-*' \
	--valid_resource_dir=$valid_output_path \
	--vocab=$dir/vocab.txt \
  --num_records_file=$train_output_path/num_records.txt \
  --image_url_prefix='D:\data\image-text-sim\flickr\imgs\' \
  --label_file=$valid_output_path/'image_labels.npy' \
  --image_feature_file=$valid_data_path/'test' \
  --image_name_bin=$valid_output_path/'image_names.npy' \
  --image_feature_bin=$valid_output_path/'image_features.npy' \
  --img2text=$valid_output_path/'img2text.npy' \
  --text2id=$valid_output_path/'text2id.npy' \
  --show_eval 1 \
  --metric_eval 1 \
  --metric_eval_interval_steps 1000 \
  --model_dir=$model_dir \
  --show_eval 1 \
  --train_only 0 \
  --save_model 1 \
  --optimizer adagrad \
  --fixed_eval_batch_size 10 \
  --num_fixed_evaluate_examples 1 \
  --num_evaluate_examples 10 \
  --save_interval_steps 500 \
  --save_interval_epochs 1 \
  --num_negs 1 \
  --debug 0 \
  --feed_dict 0 \
  --algo rnn \
  --interval 100 \
  --eval_interval 1000 \
  --margin 0.1 \
  --learning_rate 0.01 \
  --seg_method $online_seg_method \
  --feed_single $feed_single \
  --dynamic_batch_length 1 \
  --batch_size 256 \
  --eval_batch_size 1024 \
  --rnn_method bidirectional \
  --rnn_output_method max \
  --emb_dim 256 \
  --rnn_hidden_size 1024 \
  --hidden_size 1024 \
  --bias 0 \
  --cell gru \
  --monitor_level 2 \
  --num_gpus 0 \

