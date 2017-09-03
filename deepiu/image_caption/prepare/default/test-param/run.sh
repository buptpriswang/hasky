source ./config 
if [ -n $info_dir ]; then
  echo 'ok'
else
  echo 'wrong'
fi

python ./test.py --info_dir=$info_dir
