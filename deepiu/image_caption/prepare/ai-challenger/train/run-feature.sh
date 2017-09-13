mkdir -p train.feature

c0 py ./imgs2features.py --image_dir ./caption_train_images_20170902 | py ./merge-pic-feature.py caption_train_annotations_20170902.txt > caption_train_annotations_20170902.feature.txt
rand.py ./caption_train_annotations_20170902.feature.txt >  ./caption_train_annotations_20170902.feature.rand.txt 

cd ./train.feature
ln -s ../caption_train_annotations_20170902.feature.rand.txt .
split.py caption_train_annotations_20170902.feature.rand.txt 
cd ..

