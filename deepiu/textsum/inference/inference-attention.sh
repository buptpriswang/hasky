source ./prepare/config

python ./inference/inference.py \
      --model_dir '/home/gezi/new/temp/textsum/model.seq2seq.attention.luong/' \
      --debug 0 \
      --seg_method $online_seg_method \
      --feed_single $feed_single 
