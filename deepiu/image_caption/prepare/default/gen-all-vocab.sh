source ./config 
mkdir -p $dir

if [ -n "$text_dir" ]
then
 echo 'has text dir ' 'text_dir:' $text_dir 
elif [ -n "$info_dir" ]
then
  echo 'has info dir '
  text_dir=$info_dir
else
  echo 'using train data path'
  text_dir=$train_data_path 
  echo 'text_dir:' $text_dir
fi

echo $text_dir 

cat $valid_data_path/* | \
  python ./gen-vocab.py \
    --out_dir $dir \
    --min_count 1 \
    --vocab_name valid_vocab_all \
    --seg_method basic  

cat $text_dir/* | \
  python ./gen-vocab.py \
    --out_dir $dir \
    --min_count 1 \
    --vocab_name vocab_all \
    --seg_method basic  


#cat $dir/vocab.txt | vocab2project.py > $dir/vocab.project 
