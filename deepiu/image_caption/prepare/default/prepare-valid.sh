source ./config 
mkdir -p $valid_output_path

python ./gen-records.py  \
  --input_dir=$valid_data_path \
  --vocab=$dir/'vocab.txt' \
  --threads=1 \
  --output=$valid_output_path \
  --seg_method $online_seg_method \
  --np_save=1 \
  --feed_single $feed_single \
  --image_dir "$image_dir" \
  --big_feature_image_dir "$big_feature_image_dir" \
  --info_dir "$info_dir" \
  --small_feature $small_feature \
  --name test

python ./gen-distinct-texts.py --dir=$valid_output_path --shuffle $shuffle_texts --max_texts $max_texts 

cat $valid_data_path/* | \
  python ./gen-bidirectional-label-map.py  \
  --all_distinct_text_strs=$valid_output_path/'distinct_text_strs.npy' \
  --all_distinct_image_names=$valid_output_path/'distinct_image_names.npy' \
  --image_names=$valid_output_path/'image_names.npy' \
  --img2text=$valid_output_path/'img2text.npy' \
  --text2id=$valid_output_path/'text2id.npy' \
  --text2img=$valid_output_path/'text2img.npy' \
  --img2id=$valid_output_path/'img2id.npy' 
 
