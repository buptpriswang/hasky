python gen-records.py /home/gezi/data/urate/feature.txt /tmp/urate.train
python read-records-melt.py /tmp/urate.train
python read-records-melt.py /tmp/urate.train --shuffle_then_decode 0
