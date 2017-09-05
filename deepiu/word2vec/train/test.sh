source ./prepare/makeup/config 
python ./train/makeup/word2vec_optimized.py --save_path=$dir/word2vec --interactive=1 --vocab_path=$dir/vocab.txt --train_data=$dir/valid/seg_id.txt
