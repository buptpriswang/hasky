source ./config 
mkdir -p $dir

cat $train_data_path/* | python seg-ids.py --seg_method_=$online_seg_method --vocab=$dir/vocab.txt > $train_output_path/seg-id.txt
cat $valid_data_path/* | python seg-ids.py --seg_method_=$online_seg_method --vocab=$dir/vocab.txt > $valid_output_path/seg-id.txt
