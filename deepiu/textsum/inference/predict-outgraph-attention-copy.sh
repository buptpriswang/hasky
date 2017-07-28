source ./config 

python ./inference/predict-outgraph.py --input_text_max_words=30 \
  --algo seq2seq \
  --model_dir /home/gezi/new/temp/textsum/model.seq2seq.attention.luong/ \
  --vocab /home/gezi/temp/textsum/tfrecord/seq-basic/train/vocab.txt \
  --num_sampled 256 \
  --log_uniform_sample 1 \
  --seg_method $online_seg_method \
  --feed_single $feed_single \
  --seq_decode_method 0 \
  --dynamic_batch_length 1 \
  --beam_size 30 \
  --decode_max_words 10 \
  --rnn_method 0 \
  --emb_dim 1000 \
  --length_normalization_factor 1 \
  --rnn_hidden_size 1024 \
  --add_text_start 1 \
  --attention luong \
  --rnn_output_method 3 \
  --decode_copy 1 \
  --use_attention 1 \
  --cell lstm_block \
  --algo seq2seq
