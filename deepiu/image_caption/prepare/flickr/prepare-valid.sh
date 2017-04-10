source ./config 

#rm -rf $valid_output_path
mkdir -p $valid_output_path


python ./gen-records.py \
  --image_feature $valid_data_path/img2fea.txt \
  --text $valid_data_path/results_20130124.token \
  --threads 12 \
  --ori_text_index -1 \
  --write_sequence_example=$write_sequence_example \
  --vocab $train_output_path/vocab.bin \
  --out $valid_output_path \
  --name test

python ./gen-distinct-texts.py --dir $valid_output_path 
