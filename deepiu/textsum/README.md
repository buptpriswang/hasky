# train with attention   
sh ./train/seq2seq-attention.sh  
# inference with attention  
## inference output generated text/seq   
sh ./inference/inference-attention.sh    
## inference output score of in_seq -> out_seq prob  
sh ./inference/inference-score-attention.sh  

# train with pointer network (copy_only)
sh ./train/seq2seq-copy.sh 

# train with gen and copy no switch 
sh ./train/seq2seq-gen-copy.sh  

# train with gen and copy using swith prob (gen_copy_switch)  
sh ./train/seq2seq-gen-copy.sh  


