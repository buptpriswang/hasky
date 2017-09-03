source ./config 
mkdir -p $dir 

if [ -n $text_dir ]
then
 echo 'text_dir:' $text_dir 
elif [ -n $info_dir ]
then
  text_dir=$info_dir
else
  text_dir=$train_data_path 
  echo 'text_dir:' $text_dir
fi

echo $text_dir 

cat $text_dir/* | \
  python ./gen-vocab.py \
    --out_dir $dir \
    --min_count $min_count \
    --most_common $vocab_size \
    --seg_method $seg_method 

cat $dir/vocab.txt | vocab2project.py > $dir/vocab.project 
