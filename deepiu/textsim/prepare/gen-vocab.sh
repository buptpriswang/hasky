source ./config 
mkdir -p $dir

cat $train_data_path/* | \
  python ./gen-vocab.py \
    --out_dir $dir \
    --min_count $min_count \
    --most_common $vocab_size \
    --seg_method $seg_method \
