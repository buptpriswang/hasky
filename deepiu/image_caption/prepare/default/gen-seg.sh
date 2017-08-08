source ./config 
mkdir -p $dir

cat $train_data_path/* | python seg.py --seg_method=$online_seg_method > $train_output_path/seg.txt
cat $valid_data_path/* | python seg.py --seg_method=$online_seg_method > $valid_output_path/seg.txt
