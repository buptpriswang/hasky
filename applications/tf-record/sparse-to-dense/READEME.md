just use ../sparse/gen-records.py  
python ./read-records-melt.py ~/data/text-classification/tf-record/test 
python ./read-records-melt.py ~/data/text-classification/tf-record/test --shuffle_then_decode 0

#early version of tf can use both shuffle_then_decode 1 or 0, now only work for 1  
#so now the same as http://stackoverflow.com/questions/36917807/how-to-load-sparse-data-with-tensorflow  
For dense data usually you are doing:  

  Serialized Example (from reader) -> parse_single_example -> batch queue -> use it.  
  For sparse data you currently need to do:  

  Serialized Example (from reader) -> batch queue -> parse_example -> use it.  


but also notice early version tf has warning when sparse to dense, now not warning  
