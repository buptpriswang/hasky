source ./config 

python ./inference/inference.py \
      --model_dir=$1 \
      --seg_method $online_seg_method \
      --feed_single $feed_single 
