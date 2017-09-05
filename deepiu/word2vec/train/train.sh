#$1 is makeup/title2name or makeup/image_caption 
source ./prepare/app/$1/config
python ./train/word2vec_optimized.py --vocab_path=$dir/vocab.txt --train_data=$dir/train/seg_id.txt --save_path=$dir/word2vec 
