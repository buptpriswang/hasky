source ./prepare/config 
cp ./prepare/conf.py .
cp ./inputs/title-desc-click/input.py .

dir=/home/gezi/new/temp/textsum/ 
model_dir=$dir/model.seq2seq.gen-copy-switch
mkdir -p $model_dir

#--train_input $train_output_path/'train_*' \
python ./train.py \
  --train_input $train_output_path/'train_*' \
  --valid_input $valid_output_path/'test_*' \
	--fixed_valid_input $fixed_valid_output_path/'test' \
	--valid_resource_dir $valid_output_path \
	--vocab $train_output_path/vocab.txt \
  --num_records_file  $train_output_path/num_records.txt \
  --image_url_prefix '' \
  --model_dir=$model_dir \
  --algo seq2seq \
  --num_sampled 256 \
  --log_uniform_sample 1 \
  --fixed_eval_batch_size 10 \
  --num_fixed_evaluate_examples 1 \
  --num_evaluate_examples 10 \
  --eval_batch_size 200 \
  --debug 0 \
  --show_eval 1 \
  --show_beam_search 1 \
  --train_only 0 \
  --metric_eval 0 \
  --gen_predict 1 \
  --legacy_rnn_decoder 0 \
  --alignment_history 0 \
  --monitor_level 2 \
  --no_log 0 \
  --batch_size 128 \
  --eval_batch_size 100 \
  --num_gpus 0 \
  --min_after_dequeue 500 \
  --learning_rate 0.1 \
  --eval_interval_steps 500 \
  --metric_eval_interval_steps 1000 \
  --save_interval_steps 1000 \
  --num_metric_eval_examples 1000 \
  --metric_eval_batch_size 500 \
  --feed_dict 0 \
  --seg_method $online_seg_method \
  --feed_single $feed_single \
  --seq_decode_method 0 \
  --beam_size 10 \
  --decode_max_words 10 \
  --dynamic_batch_length 1 \
  --rnn_method 0 \
  --emb_dim 1000 \
  --rnn_hidden_size 1024 \
  --add_text_start 1 \
  --rnn_output_method 3 \
  --use_attention 1 \
  --attention_option luong \
  --gen_copy_switch 1 \
  --encode_end_mark 1 \
  --cell lstm_block \
  --num_records 0 \
  --min_records 0 \
  --log_device 0 \
  --clip_gradients 1 \ 
  --work_mode full \

