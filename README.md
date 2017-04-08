# tensorflow related work(nlp and image related, text classification, image caption, seq2seq..) 
mainly tf work, may do some pytorch related     

<div align="center">
  <img src="http://images2015.cnblogs.com/blog/61573/201703/61573-20170318205837260-872722304.png"><br><br>
</div>  
<div align="center">
  <img src="http://images2015.cnblogs.com/blog/61573/201703/61573-20170318205836604-2137010595.png"><br><br>
</div>      
<div align="center">
  <img src="http://images2015.cnblogs.com/blog/61573/201703/61573-20170318205836120-181624783.png"><br><br>
</div>  

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

Notice some examples like text-classification might move to deepiu

## ./deepiu
right now actually seq2seq related applications root
### ./deepiu/image-caption
image-caption related work now support 
#### discriminant method:  
bow,  
rnn,  
cnn(TODO)  
#### generative method:  
show_and_tell  
show_attend_and_tell(TODO)  
Input with both image and text(TODO) 

#### features
show_and_tell supported similary as im2txt, but here we also support discriminant method    
we support both ingraph/outgraph beam search  
will follow google/seq2seq method but now they work though beam search is not dynamic  
training is dynamic also support sampled softmax  
we support directly deal with image like im2txt(using inception v3) and also support use pre calc image feature  as image input(faster traning speed)  
use melt for training the code will be much shorter and handel all training details and atuo handel summary ops  
support <train + validate(random) + fixed validate + predict evaluate> all in one model   

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
seq2seq_attetion_copy(TODO)    
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
# hasky ./util/hasky （pytorch rlated help library TODO)
