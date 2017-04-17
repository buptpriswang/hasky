conf_path=./prepare
cp $conf_path/conf.py .
source $conf_path/config 

#attention 1 and num_fixed_evaluate_examples 1 will eval fail, but same config can run for textsum  
model_dir=/home/gezi/new/temp/imtxt_keyword/model/imtxt2txt.fixme
mkdir -p $model_dir

#valid_output_path=/home/gezi/new/temp/image-caption/keyword/tfrecord/seq-basic/valid.lijiaoshou/
python ./train.py \
	--train_input $train_output_path/'train-*' \
  --valid_input $valid_output_path/'test-*' \
  --fixed_valid_input $fixed_valid_output_path/'test-*' \
	--valid_resource_dir $valid_output_path \
	--vocab $dir/vocab.txt \
  --num_records_file  $train_output_path/num_records.txt \
  --model_dir=$model_dir \
  --algo imtxt2txt \
  --num_sampled 0 \
  --log_uniform_sample 1 \
  --fixed_eval_batch_size 10 \
  --num_fixed_evaluate_examples 1 \
  --num_evaluate_examples 10 \
  --show_eval 1 \
  --train_only 0 \
  --metric_eval 0 \
  --monitor_level 2 \
  --no_log 0 \
  --batch_size 128 \
  --eval_batch_size 50 \
  --num_gpus 0 \
  --min_after_dequeue 500 \
  --learning_rate 0.1 \
  --eval_interval_steps 500 \
  --metric_eval_interval_steps 1000 \
  --save_interval_steps 1000 \
  --save_interval_epochs 1 \
  --num_metric_eval_examples 1000 \
  --metric_eval_batch_size 500 \
  --margin 0.5 \
  --num_negs 0 \
  --use_neg 0 \
  --feed_dict 0 \
  --rnn_method 0 \
  --emb_dim 1024 \
  --rnn_hidden_size 1024 \
  --add_text_start 1 \
  --rnn_output_method 3 \
  --use_attention 1 \
  --cell lstm_block \
  --num_records 0 \
  --min_records 0 \
  --seg_method $online_seg_method \
  --feed_single $feed_single \
  --seq_decode_method 0 \
  --dynamic_batch_length 1 \
  --log_device 0 \
  --debug 1 \
  --work_mode full \

  #--model_dir /home/gezi/data/image-text-sim/model/model.ckpt-387000 \
  #2> ./stderr.txt 1> ./stdout.txt
