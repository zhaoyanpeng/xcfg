run `mscoco_dl.sh' to get mscoco annotation data.

`train_ids.txt' is partitioned into 4 parts so that we can parse faster by parsing the parts in parallel.

the splits seem to come from [train,test,val].txt in `coco_precomp' of `wget http://www.cs.toronto.edu/~faghri/vsepp/data.tar' 

the top 1000 image ids in the [test,val].txt are used.

see https://github.com/fartashf/vsepp#download-data 
