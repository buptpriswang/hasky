source ./config 
if [ -n "$info_dir" ]; then
  echo 'info dir ok'
else
  echo 'info dir not set'
fi

if [ -z $info_dir ]; then
  echo 'info not exists'
else
  echo 'info exists'
fi

python ./test.py --info_dir=$info_dir
