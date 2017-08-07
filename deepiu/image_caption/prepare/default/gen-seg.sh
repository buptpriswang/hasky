source ./config 
mkdir -p $dir

cat $train_data_path/* | python seg.py > $train_output_path/seg.txt
cat $valid_data_path/* | python seg.py > $valid_output_path/seg.txt
