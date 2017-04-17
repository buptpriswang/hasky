source ./config 
mkdir -p $valid_output_path

python ./gen-records.py  \
  --input_dir=$valid_data_path \
  --vocab=$train_output_path/'vocab.txt' \
  --threads=1 \
  --output=$valid_output_path \
  --seg_method $online_seg_method \
  --np_save=1 \
  --feed_single $feed_single \
  --name test

python ./gen-distinct-texts.py --dir=$valid_output_path --shuffle $shuffle_texts --max_texts $max_texts 

cat $valid_data_path/* | python ./adapt-img-labels.py | \
  python ./gen-bidirectional-label-map.py  \
  --all_distinct_text_strs=$valid_output_path/'distinct_text_strs.npy' \
  --img2text=$valid_output_path/'img2text.npy' \
  --text2id=$valid_output_path/'text2id.npy' 
 
