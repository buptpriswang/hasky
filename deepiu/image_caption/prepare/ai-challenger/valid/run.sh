python ./json2txt.py ./caption_validation_annotations_20170910.json  caption_validation_annotations_20170910.txt 
rand.py ./caption_validation_annotations_20170910.txt >  ./caption_validation_annotations_20170910.rand.txt 

mkdir -p valid

cd ./valid/ 
ln -s ../caption_validation_annotations_20170910.rand.txt .
split.py caption_validation_annotations_20170910.rand.txt 
cd ..

