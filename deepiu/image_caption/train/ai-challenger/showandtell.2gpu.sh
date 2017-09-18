conf_path=./prepare/default/app-conf/ai-challenger/seq-basic/

cp $conf_path/conf.py .
source $conf_path/config  

model_dir=/home/gezi/new/temp/image-caption/ai-challenger/model/showandtell
assistant_model_dir=/home/gezi/new/temp/image-caption/ai-challenger/model/bow
mkdir -p $model_dir

python ./train.py \
	--train_input $train_output_path/'train-*' \
  --valid_input $valid_output_path/'test-*' \
	--valid_resource_dir $valid_output_path \
	--vocab $dir/vocab.txt \
  --image_dir $win_image_dir \
  --label_file $valid_output_path/'image_label.npy' \
  --img2text $valid_output_path/'img2text.npy' \
  --text2id $valid_output_path/'text2id.npy' \
  --image_name_bin $valid_output_path/'image_names.npy' \
  --image_feature_bin $valid_output_path/'image_features.npy' \
  --num_records_file  $train_output_path/num_records.txt \
  --model_dir=$model_dir \
  --assistant_model_dir $assistant_model_dir \
  --algo show_and_tell \
  --num_sampled 0 \
  --log_uniform_sample 1 \
  --fixed_eval_batch_size 10 \
  --num_fixed_evaluate_examples 2 \
  --num_evaluate_examples 10 \
  --show_eval 1 \
  --train_only 0 \
  --metric_eval 1 \
  --monitor_level 2 \
  --no_log 0 \
  --batch_size 128 \
  --num_gpus 0 \
  --eval_batch_size 1000 \
  --min_after_dequeue 500 \
  --learning_rate 0.1 \
  --eval_interval_steps 500 \
  --metric_eval_interval_steps 1000 \
  --save_interval_steps 1000 \
  --save_interval_epochs 1 \
  --num_metric_eval_examples 500 \
  --metric_eval_batch_size 500 \
  --max_texts 20000 \
  --margin 0.5 \
  --feed_dict 0 \
  --num_records 0 \
  --min_records 0 \
  --seg_method $online_seg_method \
  --feed_single $feed_single \
  --seq_decode_method greedy \
  --length_normalization_factor 1. \
  --keep_prob 1. \
  --scheduled_sampling_probability 0. \
  --beam_size 3 \
  --emb_dim 512 \
  --rnn_hidden_size 512 \
  --dynamic_batch_length 1 \
  --log_device 0 \
  --work_mode full \

  #--model_dir /home/gezi/data/image-text-sim/model/model.ckpt-387000 \
  #2> ./stderr.txt 1> ./stdout.txt
