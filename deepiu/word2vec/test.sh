dir=/home/gezi/new/temp/image-caption/lijiaoshou/tfrecord/seq-basic
python word2vec_optimized.py --save_path=$dir/word2vec2 --interactive=1 --vocab_path=$dir/vocab.txt --train_data=$dir/valid/seg-id.txt
