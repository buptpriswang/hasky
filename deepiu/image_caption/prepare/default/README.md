here is general version assume the input must be
image_name\ttext_list\timage_feature

text_list is texts seperated by \x01
image_feature is float features seperated by \x01

for image caption train from raw image 

for input data is image_text(urllib.qutoe_plus), you'd better  
first convert to local binary pics to 1 directory
and pre pare using this image dir   
by this way you can do evaluaton like recall@n  
