awk -F"\t" '{print $1}' /home/gezi/data/image-caption/flickr/train/img2fea.txt | python ./imgs2features.py > /home/gezi/data/image-caption/flickr/train/img2fea_inceptionV3.txt
