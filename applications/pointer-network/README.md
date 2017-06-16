try to use train-exp.sh first for experiments 

# davenag https://github.com/devnag/tensorflow-pointer-networks this is simple just put here but not suggest to follow

# static https://github.com/ikostrikov/TensorFlow-Pointer-Networks  this is great simple example static rnn 
I made some modification to make it work correctly using enc input to blend for feed_prev, also make attention simplier  
and easy to understand 

# static2 similar as static but using seq2seq.attention_wrapper(tf version 1.2 attention code modify a bit), this is suggested usage!!   

# static-old is similar as static2 but using tf version 1.0 attention code, this is just show usage not suggested 

# static-legacy this is same as static but make smallest change to original work so can be baseline for compare  

# dynamic https://github.com/devsisters/pointer-network-tensorflow  this is great simple example dynamic rnn
