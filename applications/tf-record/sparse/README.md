#shuffle then decode for sparse data ok
python ./read-records-melt.py ~/data/text-classification/tf-record/test  
steps: 10000  duration: 9.00163507462 instance/s: 5554.54643357  
steps: 20000  duration: 8.95443487167 instance/s: 5583.82530183  
steps: 30000  duration: 8.95602107048 instance/s: 5582.83635183  

#decode then shuffle not work for sparse data 
python ./read-records-melt.py ~/data/text-classification/tf-record/test --shuffle_then_decode 0  
TypeError: Expected binary or unicode string, got <tensorflow.python.framework.sparse_tensor.SparseTensor object at 0x857fc10> 
