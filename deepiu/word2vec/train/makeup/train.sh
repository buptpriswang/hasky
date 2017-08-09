source ./prepare/makeup/config 
python ./train/makeup/word2vec_optimized.py --vocab_path=$dir/vocab.txt --train_data=$dir/train/seg_id.txt --save_path=$dir/word2vec 

