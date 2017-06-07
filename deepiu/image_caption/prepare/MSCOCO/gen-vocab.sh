source ./config 
mkdir -p $dir
python gen-vocab.py --min_count 4 --out_dir $dir
