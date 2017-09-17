conf_path=./prepare/default/app-conf/ai-challenger/seq-basic-finetune
cp $conf_path/conf.py .
source $conf_path/config 

model_dir=/home/gezi/new/temp/image-caption/ai-challenger/model/bow.finetune.later
mkdir -p $model_dir

python ./train.py \
  --train_input=$train_output_path/'train-*' \
  --valid_input=$valid_output_path/'test-*' \
  --fixed_valid_input=$fixed_valid_output_path/'test-*' \
  --valid_resource_dir=$valid_output_path \
  --vocab=$dir/vocab.txt \
  --num_records_file=$train_output_path/num_records.txt \
  --label_file=$valid_output_path/'image_labels.npy' \
  --image_feature_file=$valid_data_path/'test' \
  --image_name_bin=$valid_output_path/'image_names.npy' \
  --image_feature_bin=$valid_output_path/'image_features.npy' \
  --img2text=$valid_output_path/'img2text.npy' \
  --text2id=$valid_output_path/'text2id.npy' \
  --text2img=$valid_output_path/'text2img.npy' \
  --img2id=$valid_output_path/'img2id.npy' \
  --eval_text2img 0 \
  --fixed_eval_batch_size 10 \
  --num_fixed_evaluate_examples 1 \
  --num_evaluate_examples 10 \
  --show_eval 1 \
  --train_only 0 \
  --metric_eval 1 \
  --monitor_level 2 \
  --no_log 0 \
  --batch_size 32 \
  --num_gpus 0 \
  --eval_batch_size 200 \
  --min_after_dequeue 1000 \
  --eval_interval_steps 200 \
  --metric_eval_interval_steps 1000 \
  --save_interval_seconds 7200 \
  --save_interval_steps 1000 \
  --save_interval_epochs 10 \
  --num_epochs 2000 \
  --num_metric_eval_examples 200 \
  --metric_eval_batch_size 100 \
  --max_texts 20000 \
  --debug 0 \
  --num_negs 1 \
  --neg_left 0 \
  --neg_right 1 \
  --interval 100 \
  --feed_dict 0 \
  --margin 0.5 \
  --algo bow \
  --image_model InceptionResnetV2 \
  --image_checkpoint_file='/home/gezi/data/image_model_check_point/inception_resnet_v2_2016_08_30.ckpt' \
  --pre_calc_image_feature 0 \
  --finetune_image_model 1 \
  --learning_rate 0.01 \
  --combiner=sum \
  --exclude_zero_index 1 \
  --dynamic_batch_length 1 \
  --emb_dim 256 \
  --hidden_size 1024 \
  --model_dir $model_dir \
  --num_records 0 \
  --min_records 12 \
  --log_device 0 
 
