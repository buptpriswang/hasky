python ./json2txt.py 
rand.py ./caption_train_annotations_20170902.txt >  ./caption_train_annotations_20170902.rand.txt 
split-train-valid.py ./caption_train_annotations_20170902.rand.txt 0.1 

mkdir -p train
mkdir -p valid 
rm -rf ./train/*  
rm -rf ./valid/*

cd ./train/ 
ln -s ../caption_train_annotations_20170902.rand.train.txt .
split.py caption_train_annotations_20170902.rand.train.txt 
cd ..

cd ./valid/ 
ln -s ../caption_train_annotations_20170902.rand.valid.txt .
split.py caption_train_annotations_20170902.rand.valid.txt 
cd ..
