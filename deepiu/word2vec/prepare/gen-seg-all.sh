source ./config 
mkdir -p $dir

mkdir -p $dir/train
mkdir -p $dir/valid

cat $train_data_path/* | python ./seg-ids-text.py --seg_method=$online_seg_method --out_text $train_output_path/seg.txt --out_id $dir/train/seg_id.txt --vocab $dir/vocab.txt
cat $valid_data_path/* | python ./seg-ids-text.py --seg_method=$online_seg_method --out_text  $valid_output_path/seg.txt --out_id $dir/valid/seg_id.txt --vocab $dir/vocab.txt
