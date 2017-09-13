python ./json2txt.py ./caption_train_annotations_20170902.json  caption_train_annotations_20170902.txt 
rand.py ./caption_train_annotations_20170902.txt >  ./caption_train_annotations_20170902.rand.txt 

mkdir -p train

cd ./train/ 
ln -s ../caption_train_annotations_20170902.rand.txt .
split.py caption_train_annotations_20170902.rand.txt 
cd ..

