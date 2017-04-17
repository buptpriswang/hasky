source ./config 
mkdir -p $fixed_valid_output_path

echo $valid_data_path 

python ./gen-records.py \
  --threads 1 \
  --input_dir $valid_data_path \
  --vocab $train_output_path/vocab.txt \
  --output $fixed_valid_output_path \
  --seg_method $online_seg_method \
  --np_save 1 \
  --num_max_records 10 \
  --num_max_input 1 \
  --feed_single $feed_single \
  --name test
