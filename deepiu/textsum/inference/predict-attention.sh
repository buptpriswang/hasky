source ./config 

python ./inference/predict.py \
  --algo seq2seq \
  --model_dir /home/gezi/new/temp/makeup/title2name/model/seq2seq.attention/ \
  --vocab /home/gezi/new/temp/makeup/title2name/tfrecord/seq-basic/vocab.txt \
  --num_sampled 0 \
  --log_uniform_sample 1 \
  --seg_method $online_seg_method \
  --feed_single $feed_single \
  --seq_decode_method 0 \
  --dynamic_batch_length 1 \
  --beam_size 10 \
  --decode_max_words 10 \
  --rnn_method forward \
  --emb_dim 256 \
  --length_norm 1 \
  --rnn_hidden_size 1024 \
  --add_text_start 1 \
  --rnn_output_method 3 \
  --use_attention 1 \
  --cell lstm_block \
  --algo seq2seq
