python ./imgskv2features.py \
    --image_model_name InceptionV3 \
    --image_height 299 \
    --image_width 299 \
    --image_checkpoint_file /home/gezi/data/inceptionv3/inception_v3.ckpt \
    --batch_size 512 \
    --kv_file $1 > $2
