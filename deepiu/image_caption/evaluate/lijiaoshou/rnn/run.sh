# text2text sim
#---tb bow 
#python ./inference-score-bytextsim.py --model_dir /home/gezi/new/temp/image-caption/lijiaoshou/model/rnn/ --vocab /home/gezi/new/temp/image-caption/lijiaoshou/tfrecord/seq-basic/vocab.txt > ~/temp/tr.html 
##---tb2 bow train by ct0
#python ./inference-score-bytextsim.py --model_dir /home/gezi/new/temp/image-caption/keyword/model/bow/ --vocab /home/gezi/new/temp/image-caption/keyword/tfrecord/bow/vocab.txt > ~/temp/tb2.html 
#---tb3 bow train by ct0 then finetune ad 
#python ./inference-score-bytextsim.py --model_dir /home/gezi/new/temp/image-caption/keyword/model/rnn.lijiaoshou/ --vocab /home/gezi/new/temp/image-caption/keyword/tfrecord/rnn/vocab.txt > ~/temp/tr3.html 

# image2text sim
#---b bow 
#python ./inference-score.py --model_dir /home/gezi/new/temp/image-caption/lijiaoshou/model/rnn/ --vocab /home/gezi/new/temp/image-caption/lijiaoshou/tfrecord/seq-basic/vocab.txt > ~/temp/r.html 
#---b2 bow train by ct0
#python ./inference-score.py --model_dir /home/gezi/new/temp/image-caption/keyword/model/bow/ --vocab /home/gezi/new/temp/image-caption/keyword/tfrecord/bow/vocab.txt > ~/temp/b2.html 
#---b3 bow train by ct0 then finetune ad 
python ./inference-score.py --model_dir /home/gezi/new/temp/image-caption/keyword/model/rnn.lijiaoshou/ --vocab /home/gezi/new/temp/image-caption/keyword/tfrecord/seq-basic/vocab.txt > ~/temp/r3.html 
