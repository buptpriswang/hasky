cp ./app-conf/keyword/seq-basic/lijaoshou.config config 
sh ./run-local.sh 
cp ./app-conf/keyword/seq-basic/lijaoshou.finetune.config config 
sh ./prepare-train.sh 
