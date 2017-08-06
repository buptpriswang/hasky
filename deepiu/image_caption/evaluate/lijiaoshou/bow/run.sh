# text2text sim
#---tb bow 
python ./inference-score-bytextsim.py --model_dir /home/gezi/new/temp/image-caption/lijiaoshou/model/bow/ --vocab /home/gezi/new/temp/image-caption/lijiaoshou/tfrecord/bow/vocab.txt > ~/temp/tb.html 
#---tb2 bow train by ct0
python ./inference-score-bytextsim.py --model_dir /home/gezi/new/temp/image-caption/keyword/model/bow/ --vocab /home/gezi/new/temp/image-caption/keyword/tfrecord/bow/vocab.txt > ~/temp/tb2.html 
#---tb3 bow train by ct0 then finetune ad 
python ./inference-score-bytextsim.py --model_dir /home/gezi/new/temp/image-caption/keyword/model/bow.lijiaoshou/ --vocab /home/gezi/new/temp/image-caption/keyword/tfrecord/bow/vocab.txt > ~/temp/tb3.html 

# image2text sim
#---b bow 
python ./inference-score.py --model_dir /home/gezi/new/temp/image-caption/lijiaoshou/model/bow/ --vocab /home/gezi/new/temp/image-caption/lijiaoshou/tfrecord/bow/vocab.txt > ~/temp/b.html 
#---b2 bow train by ct0
python ./inference-score.py --model_dir /home/gezi/new/temp/image-caption/keyword/model/bow/ --vocab /home/gezi/new/temp/image-caption/keyword/tfrecord/bow/vocab.txt > ~/temp/b2.html 
#---b3 bow train by ct0 then finetune ad 
python ./inference-score.py --model_dir /home/gezi/new/temp/image-caption/keyword/model/bow.lijiaoshou/ --vocab /home/gezi/new/temp/image-caption/keyword/tfrecord/bow/vocab.txt > ~/temp/b3.html 
