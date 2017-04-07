#dense data   
http://stackoverflow.com/questions/36917807/how-to-load-sparse-data-with-tensorflow  
as this suggested, dense data usually use decode then shuffle, but can also use shuffle then decode   
decode then shuffle will use parse_single_example  
(One might see performance advantages by batching `Example` protos with `parse_example` instead of using this function directly.)    
shuffle then decode will use parse_example (suppose this will be faster?)  

here we do experiment  

python read-records-melt.py /tmp/urate.train  
shuffle then decode? True
batch_size: 5

steps: 10000  duration: 8.26392602921 instance/s: 6050.39297584  
steps: 20000  duration: 8.21452903748 instance/s: 6086.77621954  
steps: 30000  duration: 8.20779085159 instance/s: 6091.77315846  


python read-records-melt.py /tmp/urate.train --shuffle_then_decode 0  
shuffle then decode? False  
batch_size: 5  

steps: 10000  duration: 10.7779331207 instance/s: 4639.10839304  
steps: 20000  duration: 10.6920368671 instance/s: 4676.37744064  
steps: 30000  duration: 10.6808850765 instance/s: 4681.25999314  

python read-records-melt.py /tmp/urate.train --batch_size 256 --num_test_steps 1000 
steps: 1000  duration: 30.6210808754 instance/s: 8360.25354695  
steps: 2000  duration: 30.7497649193 instance/s: 8325.26689788  
steps: 3000  duration: 30.7622931004 instance/s: 8321.87636874  

python read-records-melt.py /tmp/urate.train --batch_size 256 --num_test_steps 1000  --shuffle_then_decode 0  
steps: 1000  duration: 41.5285937786 instance/s: 6164.42736695  
steps: 2000  duration: 41.3832221031 instance/s: 6186.08187062  
steps: 3000  duration: 41.333564043 instance/s: 6193.51381684  


so for dense data prefer shuffle then decode, and for sparse data we can only use shuffle then decode(see sparse dir)  
when to use decode then shuffle ?  might be sequence example  only has parse_single_sequence_exmaple interface  
also why use sequence example, it is for dynamic length like cpation word ids, but you can also use spare data   
then sparse to dense, but what if you want bucket batching ?  seems must first decode then shuffle   


#sequence data 
notice only decode then shuffle (parse single sequence example)  
must use tf.batch, dynamic_pad=True 

now the key reason for using decode then shuffle, bucket by length, can do it by --buckets '100'

#NOTICE ids for single file do not have good random.. so random mostly affected by file shuffle, multiple file is much better!  
11889 12718 12021 12834 12925  
8376 8012 8323 8562 8555  
if use tf.batch for single file will read by sequence ... --batch_join 0 --shuffle_batch 0  
TODO check multiple files' randomness  

