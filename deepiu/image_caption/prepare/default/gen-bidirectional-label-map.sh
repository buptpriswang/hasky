source ./config 
mkdir -p $valid_output_path

cat /data2/data/ai_challenger/ai_challenger_caption_validation_20170910/caption_validation_annotations_20170910.txt | \
  python ./gen-bidirectional-label-map.py  \
  --all_distinct_text_strs=$valid_output_path/'distinct_text_strs.npy' \
  --all_distinct_image_names=$valid_output_path/'distinct_image_names.npy' \
  --image_names=$valid_output_path/'image_names.npy' \
  --img2text=$valid_output_path/'img2text.npy' \
  --text2id=$valid_output_path/'text2id.npy' \
  --text2img=$valid_output_path/'text2img.npy' \
  --img2id=$valid_output_path/'img2id.npy' 
 
