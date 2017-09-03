source ./config 

mkdir -p $train_output_path

echo $train_data_path

python ./gen-records.py  \
  --input_dir $train_data_path \
  --threads 12 \
  --vocab $dir/vocab.txt \
  --output $train_output_path \
  --seg_method $online_seg_method \
  --feed_single $feed_single \
  --image_dir "$image_dir" \
  --info_dir "$info_dir" \
  --name train \
  --np_save 0

#python ./gen-distinct-texts.py --dir $train_output_path 
