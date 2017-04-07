python gen-sequence-records.py /home/gezi/data/urate/train /tmp/urate.train.seq   
python read-sequence-records.py /tmp/urate.train.seq  
--10, 20 can be one group
python read-sequence-records.py /tmp/urate.train.seq --buckets 21,30,60  
--10, 20 can not be one group, so means < 20, >= 20 < 30, >=60 
python read-sequence-records.py /tmp/urate.train.seq --buckets 20,30,60  
