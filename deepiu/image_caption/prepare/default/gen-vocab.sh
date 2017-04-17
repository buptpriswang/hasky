source ./config 
mkdir -p $dir

cat $train_data_path/* | \
  python ./gen-vocab.py \
    --out_dir $dir \
    --min_count 10 \
    --most_common $vocab_size \
    --seg_method $seg_method \
