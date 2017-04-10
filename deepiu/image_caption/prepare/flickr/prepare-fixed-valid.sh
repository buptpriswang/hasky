source ./config 

#rm -rf $fixed_valid_output_path
mkdir -p $fixed_valid_output_path

python ./gen-records.py \
  --image_feature $valid_data_path/img2fea.txt \
  --text $valid_data_path/results_20130124.token \
  --vocab $train_output_path/vocab.bin \
  --ori_text_index -1 \
  --out $fixed_valid_output_path \
  --write_sequence_example=$write_sequence_example \
  --name test \
  --threads 1 \
  --num_records 10 \
