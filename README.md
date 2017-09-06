# current tensorflow version used  
'1.2.0-rc0' 

# tensorflow related work(nlp and image related, text classification, image caption, seq2seq, pointer-network..) 
mainly tf work, may do some pytorch related work

## incase not find dependence, make sure set PYTHONPATH to include hasky/util so we can find gezi and melt  
## ./applications/   
### ./applications/tf-record/   
show how to write and read TFRecord(tensorflow standard dataa format, Example and SequenceExample, sparse_to_dense, dynamic code)   
might move to ./exp  as this is for demo purpose  
### ./applications/text-classification  
reading libsvm format then do text classification    
### ./applications/text-regression
reading libsvm format then do text regression  
### ./applications/text-binary-classification  
reading libsvm format then do text binary classification, evaluate by auc  
### ./applications/sparse-tensor-classification/ (depreciated)  
this is self contained mlp classification example showing   
how to read sparse TFRecord and train a mlp classifier 
### ./applications/pointer-network
this is implmentaion of pointer network, dealing sorting problem 

## ./deepiu
right now actually seq2seq related applications root
### ./deepiu/image-caption
image-caption related work now support 
#### discriminant method:  
bow,  
rnn,  
cnn
#### generative method:  
show_and_tell  
show_attend_and_tell(TODO)    
Input with both image and text  

#### features
show_and_tell supported similary as im2txt, but here we also support discriminant method    
we support both ingraph/outgraph beam search  
will follow google/seq2seq method but now here just works though beam search is not dynamic  
training is dynamic also support sampled softmax  
support directly deal with image like im2txt(using inception v3, allow distort images) and also support use pre calc image feature  as image input(faster train speed but can not distort images any more, experiment show distort not very useful?)   
support using Example while im2txt use SequenceExample, also support use SequenceExample incase you want do bucket batch for rnn decode train  
use melt for training the code will be much shorter and handel all training details and auto handel summary ops  
support <train + validate(random) + fixed validate + predict evaluate> all in one mode, see below graph, will help experiment a lot, while im2txt you need seperate process to do validation  
support finetune mode, finetune from any endpoint, can finetune from pre dumped image feature model  
support recall@.. also for generative method using descriminant predictor as assistant predictor, TODO trie based beam search support for fast selection(c++)  

<div align="center">
  <img src="http://images2015.cnblogs.com/blog/61573/201704/61573-20170409001354082-1278393427.png"><br><br>
</div>  
<div align="center">
  <img src="http://images2015.cnblogs.com/blog/61573/201704/61573-20170409001550691-618821679.png"><br><br>
</div>  
<div align="center">
  <img src="http://images2015.cnblogs.com/blog/61573/201704/61573-20170409001639628-1055762157.png"><br><br>
</div>  

### ./deepiu/text-sum
app with long text as input(like image title, ct0) and predict shortter summary text(like click query)  
supporting method:  
seq2seq  
seq2seq_attetion     
seq2seq_attetion_copy   
seq2seq_attetion_gen_copy(competing softmax of gen and copy mode)  
seq2seq_attetion_gen_copy_switch(use switch to descide ratio of gen and 1- ratio of copy)  

<div align="center">
  <img src="http://images2015.cnblogs.com/blog/61573/201703/61573-20170318205837260-872722304.png"><br><br>
</div>  
<div align="center">
  <img src="http://images2015.cnblogs.com/blog/61573/201703/61573-20170318205836604-2137010595.png"><br><br>
</div>      
<div align="center">
  <img src="http://images2015.cnblogs.com/blog/61573/201703/61573-20170318205836120-181624783.png"><br><br>
</div>  

### ./deepiu/seq2seq 
common seq2seq codes used for image-caption, text-sum and other applications
### ./deepiu/util
common util for deepiu  

## ./util
### ./util/gezi
common lib 
### ./util/melt
common tensorflow related lib, you can view it similar as tf.contrib
#### ./util/melt/flow
like  tf.supervisor, make train and test flow easier,  
main functions include, application only make graph and pass train_ops and evaluate_ops to flow  
flow will do model save, log save(auto add all one dimentional shape tensors to tensorboard), show elapsed time ... 
#### ./util/melt/tfrecords  
wrapper for reading tfrcords support shuffle_then_decode and decode_then shuffle, support libsvm sparse decode  

## publish lib
### gezi ./util/gezi 
### melt ./util/melt
### deepiu ./deepiu  
### hasky ./util/hasky pytorch rlated help library TODO)
