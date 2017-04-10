echo 'prepare test'
awk -F"\t" '{print $1}' /home/gezi/data/image-caption/flickr/test/img2fea.txt | python ./imgs2features.py > /home/gezi/data/image-caption/flickr/test/img2fea_inceptionV3.txt
echo 'done test'

echo 'prepare train'
awk -F"\t" '{print $1}' /home/gezi/data/image-caption/flickr/train/img2fea.txt | python ./imgs2features.py > /home/gezi/data/image-caption/flickr/train/img2fea_inceptionV3.txt 
echo 'done train'
