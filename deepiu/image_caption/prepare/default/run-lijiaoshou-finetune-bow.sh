cp ./app-conf/keyword/bow/lijaoshou.config config
sh ./run-local.sh 
cp ./app-conf/keyword/bow/lijaoshou.finetune.config config 
sh ./prepare-train.sh 
